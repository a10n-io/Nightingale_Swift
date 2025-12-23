import MLX
import MLXNN
import Foundation

/// Streaming-optimized S3Gen wrapper that caches intermediate states
/// for incremental audio generation with low latency.
///
/// Key optimizations:
/// 1. Prompt encoding cached - compute once, reuse for all chunks
/// 2. Speaker/condition pre-computed - no redundant computation per chunk
/// 3. Incremental token processing - only encode new tokens
/// 4. Windowed ODE - process with minimal overlap for continuity
public class S3GenStreaming {

    // MARK: - Cached State

    /// The underlying S3Gen model
    private let s3gen: S3Gen

    /// Cached prompt encoding [1, promptLen*2, 512]
    private var cachedPromptEncoding: MLXArray?

    /// Cached prompt mel features [1, promptLen, 80]
    private var cachedPromptMel: MLXArray?

    /// Cached speaker conditioning [1, 80]
    private var cachedSpkCond: MLXArray?

    /// Number of prompt tokens (for offset calculations)
    private var promptTokenCount: Int = 0

    /// All tokens processed so far (for incremental encoding)
    private var processedTokens: [Int] = []

    /// Cached full encoder output [1, totalLen*2, 80] (mu projection)
    private var cachedMu: MLXArray?

    /// Audio samples generated so far
    private var generatedSamples: [Float] = []

    /// ODE solver overlap frames for continuity (larger = better quality, slower)
    private let overlapFrames: Int = 32

    /// Cross-fade samples for smooth chunk transitions
    private let crossFadeSamples: Int = 480  // 20ms at 24kHz

    /// Previous chunk's tail for cross-fading
    private var previousChunkTail: [Float] = []

    // MARK: - Configuration

    /// Number of ODE timesteps (lower = faster, higher = better quality)
    public var nTimesteps: Int = 8

    /// CFG rate for decoder
    public var cfgRate: Float = 0.7

    // MARK: - Initialization

    public init(s3gen: S3Gen) {
        self.s3gen = s3gen
    }

    // MARK: - State Management

    /// Initialize streaming state with prompt (call once before generating)
    public func initializeWithPrompt(
        promptTokens: MLXArray,
        promptFeat: MLXArray,
        speechEmbMatrix: MLXArray
    ) {
        let startTime = Date()

        // 1. Normalize speaker embedding [1, 192] -> [1, 80]
        var speechEmb = speechEmbMatrix
        let norm = sqrt(sum(speechEmb * speechEmb, axis: 1, keepDims: true)) + 1e-8
        speechEmb = speechEmb / norm
        cachedSpkCond = matmul(speechEmb, s3gen.spkEmbedAffine.weight) + s3gen.spkEmbedAffine.bias!

        // 2. Cache prompt mel features
        cachedPromptMel = promptFeat

        // 3. Encode prompt tokens (expensive - do once!)
        let vocabSize = s3gen.inputEmbedding.weight.shape[0]
        let clippedPrompt = clip(promptTokens, min: 0, max: vocabSize - 1)
        let promptEmb = s3gen.inputEmbedding(clippedPrompt)
        let promptEncoded = s3gen.encoder(promptEmb)  // [1, promptLen*2, 512]
        cachedPromptEncoding = promptEncoded

        // 4. Store prompt token count
        promptTokenCount = promptTokens.shape[1]
        processedTokens = promptTokens.asArray(Int32.self).map { Int($0) }

        // 5. Project to mel space
        cachedMu = s3gen.encoderProj(promptEncoded)  // [1, promptLen*2, 80]

        // Force evaluation
        eval(cachedSpkCond!, cachedPromptEncoding!, cachedMu!)

        let elapsed = Date().timeIntervalSince(startTime)
        print("📦 S3GenStreaming: Cached prompt (\(promptTokenCount) tokens) in \(String(format: "%.0f", elapsed * 1000))ms")
    }

    /// Cached ODE output (mel spectrogram) for already-generated regions
    private var cachedOdeOutput: MLXArray?

    /// Generate audio for new tokens incrementally using WINDOWED ODE
    /// Only processes new mel frames + small overlap for continuity
    /// Returns ONLY the new audio samples (append to previous)
    public func generateIncremental(newTokens: [Int]) -> [Float] {
        guard let cachedMu = cachedMu,
              let cachedPromptMel = cachedPromptMel,
              let cachedSpkCond = cachedSpkCond else {
            print("⚠️ S3GenStreaming: Not initialized - call initializeWithPrompt first")
            return []
        }

        let startTime = Date()

        // 1. Encode new tokens
        let newTokenArray = MLXArray(newTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let vocabSize = s3gen.inputEmbedding.weight.shape[0]
        let clippedNew = clip(newTokenArray, min: 0, max: vocabSize - 1)
        let newEmb = s3gen.inputEmbedding(clippedNew)

        // 2. Run encoder on new tokens only (approximate - for speed)
        let newEncoded = s3gen.encoder(newEmb)  // [1, newLen*2, 512]
        let newMu = s3gen.encoderProj(newEncoded)  // [1, newLen*2, 80]

        // 3. Update cached mu (concatenate new mu)
        let updatedMu = concatenated([cachedMu, newMu], axis: 1)
        self.cachedMu = updatedMu
        processedTokens.append(contentsOf: newTokens)

        let encoderTime = Date().timeIntervalSince(startTime)

        // 4. WINDOWED ODE: Only process new frames + small overlap
        let odeStart = Date()

        let L_pm = cachedPromptMel.shape[1]
        let L_total = updatedMu.shape[1]
        let L_new = newTokens.count * 2  // Upsampled 2x

        // Window parameters
        let contextFrames = min(overlapFrames, L_pm)  // Context from prompt
        let windowStart = max(0, L_total - L_new - contextFrames)
        let windowLen = L_total - windowStart

        // Extract window from mu [B, T, 80]
        let muWindow = updatedMu[0..., windowStart..<L_total, 0...]

        // Create conditions for window
        let windowCondStart = max(0, windowStart - L_pm)
        let promptInWindow = windowStart < L_pm ? cachedPromptMel[0..., windowStart..<L_pm, 0...] : nil
        let zerosNeeded = windowLen - (promptInWindow?.shape[1] ?? 0)

        let condsWindow: MLXArray
        if let prompt = promptInWindow, zerosNeeded > 0 {
            let zeros = MLXArray.zeros([1, zerosNeeded, 80], dtype: cachedPromptMel.dtype)
            condsWindow = concatenated([prompt, zeros], axis: 1)
        } else if let prompt = promptInWindow {
            condsWindow = prompt
        } else {
            condsWindow = MLXArray.zeros([1, windowLen, 80], dtype: cachedPromptMel.dtype)
        }

        // Initialize noise for WINDOW only
        let xtWindow = s3gen.fixedNoise[0..., 0..., 0..<windowLen]

        // Transpose for decoder [B, C, T]
        let condsT = condsWindow.transposed(0, 2, 1)
        let muT = muWindow.transposed(0, 2, 1)
        let mask = MLXArray.ones([1, 1, windowLen], dtype: muT.dtype)

        // Run ODE solver on WINDOW
        var xtVar = xtWindow

        // Cosine time scheduling
        var tSpan: [Float] = []
        for i in 0...nTimesteps {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        // Pre-allocate CFG tensors (smaller - for window only)
        let zeroMu = MLXArray.zeros(like: muT)
        let zeroSpk = MLXArray.zeros(like: cachedSpkCond)
        let zeroCond = MLXArray.zeros(like: condsT)

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // CFG batch
            let xIn = concatenated([xtVar, xtVar], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([muT, zeroMu], axis: 0)
            let spkIn = concatenated([cachedSpkCond, zeroSpk], axis: 0)
            let condIn = concatenated([condsT, zeroCond], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            let vBatch = s3gen.decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskIn)
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            xtVar = xtVar + v * dt
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        let odeTime = Date().timeIntervalSince(odeStart)

        // 5. Vocoder - process full window, then extract new portion with cross-fade
        let vocoderStart = Date()

        // Vocoder the FULL window (including overlap for continuity)
        let wav = s3gen.vocoder(xtVar)
        eval(wav)

        var windowSamples = wav.asArray(Float.self)

        // Calculate where new audio starts (skip overlap context in audio domain)
        // Each mel frame ≈ 256 samples at 24kHz (hop size)
        let overlapAudioSamples = contextFrames * 256

        // Extract new portion (after overlap)
        var newSamples: [Float]
        if overlapAudioSamples < windowSamples.count {
            newSamples = Array(windowSamples[overlapAudioSamples...])
        } else {
            newSamples = windowSamples
        }

        // Apply cross-fade with previous chunk for smooth transition
        if !previousChunkTail.isEmpty && !newSamples.isEmpty {
            let fadeLen = min(crossFadeSamples, previousChunkTail.count, newSamples.count)
            for i in 0..<fadeLen {
                let fadeOut = Float(fadeLen - i) / Float(fadeLen)  // 1.0 -> 0.0
                let fadeIn = Float(i) / Float(fadeLen)            // 0.0 -> 1.0
                newSamples[i] = previousChunkTail[previousChunkTail.count - fadeLen + i] * fadeOut + newSamples[i] * fadeIn
            }
        }

        // Save tail for next chunk's cross-fade
        if newSamples.count > crossFadeSamples {
            previousChunkTail = Array(newSamples.suffix(crossFadeSamples))
            // Don't include the cross-fade tail in output (will be blended with next chunk)
            newSamples = Array(newSamples.dropLast(crossFadeSamples))
        }

        generatedSamples.append(contentsOf: newSamples)

        let vocoderTime = Date().timeIntervalSince(vocoderStart)
        let totalTime = Date().timeIntervalSince(startTime)

        print("🔊 S3GenStreaming: \(newTokens.count) tokens → \(newSamples.count) samples (\(String(format: "%.0f", Double(newSamples.count) / 24.0))ms audio) [window: \(windowLen) frames]")
        print("   Encoder: \(String(format: "%.0f", encoderTime * 1000))ms | ODE: \(String(format: "%.0f", odeTime * 1000))ms | Vocoder: \(String(format: "%.0f", vocoderTime * 1000))ms | Total: \(String(format: "%.0f", totalTime * 1000))ms")

        return newSamples
    }

    /// Reset streaming state (call before new utterance)
    public func reset() {
        cachedPromptEncoding = nil
        cachedPromptMel = nil
        cachedSpkCond = nil
        cachedMu = nil
        cachedOdeOutput = nil
        previousChunkTail = []
        promptTokenCount = 0
        processedTokens = []
        generatedSamples = []
        print("🔄 S3GenStreaming: State reset")
    }

    /// Flush any remaining audio (call after last chunk)
    public func flush() -> [Float] {
        // Return any remaining cross-fade tail
        let remaining = previousChunkTail
        previousChunkTail = []
        generatedSamples.append(contentsOf: remaining)
        return remaining
    }

    /// Get total audio generated so far
    public var totalAudioSamples: [Float] {
        return generatedSamples
    }

    /// Get total tokens processed
    public var totalTokensProcessed: Int {
        return processedTokens.count - promptTokenCount
    }
}


// MARK: - ChatterboxEngine Extension for Streaming

extension ChatterboxEngine {

    /// Create a streaming S3Gen wrapper using the loaded model
    public func createStreamingS3Gen() async -> S3GenStreaming? {
        guard let s3gen = await self.getS3Gen() else {
            print("⚠️ S3Gen not loaded")
            return nil
        }
        return S3GenStreaming(s3gen: s3gen)
    }
}
