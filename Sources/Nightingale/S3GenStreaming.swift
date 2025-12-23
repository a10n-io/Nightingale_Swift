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

    /// Track if this is the first chunk (for fast mode)
    private var isFirstChunk: Bool = true

    /// ODE solver overlap frames for continuity (larger = better quality, slower)
    /// 64 frames = ~680ms of context at 256 samples/frame
    private let overlapFrames: Int = 64

    /// Cross-fade samples for smooth chunk transitions
    /// Larger = smoother but more latency. 2400 = 100ms at 24kHz
    private let crossFadeSamples: Int = 2400

    /// Previous chunk's tail for cross-fading
    private var previousChunkTail: [Float] = []

    // MARK: - Configuration

    /// Number of ODE timesteps (lower = faster, higher = better quality)
    /// 8 = best quality, 4 = fast, 2 = fastest
    public var nTimesteps: Int = 8

    /// CFG rate for decoder (0 = no CFG = 2x faster)
    public var cfgRate: Float = 0.7

    /// Use fast mode for ALL chunks (fewer ODE steps, no CFG)
    /// Set to true for speed, false for quality
    public var fastMode: Bool = true

    /// ODE steps for fast mode (used when fastMode=true)
    public var fastOdeSteps: Int = 4

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
    /// Re-encodes ALL tokens for quality, but only runs ODE on new frames
    /// Returns ONLY the new audio samples (append to previous)
    public func generateIncremental(newTokens: [Int]) -> [Float] {
        guard let cachedPromptMel = cachedPromptMel,
              let cachedSpkCond = cachedSpkCond else {
            print("⚠️ S3GenStreaming: Not initialized - call initializeWithPrompt first")
            return []
        }

        let startTime = Date()

        // 1. Accumulate all tokens
        processedTokens.append(contentsOf: newTokens)

        // 2. Re-encode ALL tokens with full bidirectional context
        // This is key for quality - the encoder uses self-attention
        let allTokenArray = MLXArray(processedTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let vocabSize = s3gen.inputEmbedding.weight.shape[0]
        let clippedAll = clip(allTokenArray, min: 0, max: vocabSize - 1)
        let allEmb = s3gen.inputEmbedding(clippedAll)

        // Run encoder on ALL tokens for proper bidirectional attention
        let allEncoded = s3gen.encoder(allEmb)  // [1, allLen*2, 512]
        let updatedMu = s3gen.encoderProj(allEncoded)  // [1, allLen*2, 80]
        self.cachedMu = updatedMu

        let encoderTime = Date().timeIntervalSince(startTime)

        // 4. FULL ODE: Run on entire sequence for consistency
        // This ensures no representation drift between chunks
        let odeStart = Date()

        let L_pm = cachedPromptMel.shape[1]
        let L_total = updatedMu.shape[1]
        let L_new = newTokens.count * 2  // Upsampled 2x

        // Create full conditions [B, T, 80]
        let zerosNeeded = L_total - L_pm
        let fullConds: MLXArray
        if zerosNeeded > 0 {
            let zeros = MLXArray.zeros([1, zerosNeeded, 80], dtype: cachedPromptMel.dtype)
            fullConds = concatenated([cachedPromptMel, zeros], axis: 1)
        } else {
            fullConds = cachedPromptMel[0..., 0..<L_total, 0...]
        }

        // Transpose for decoder [B, C, T]
        let condsT = fullConds.transposed(0, 2, 1)
        let muT = updatedMu.transposed(0, 2, 1)
        let mask = MLXArray.ones([1, 1, L_total], dtype: muT.dtype)

        // Run ODE solver on FULL sequence
        // Use fixed noise sliced to full length
        var xtVar = s3gen.fixedNoise[0..., 0..., 0..<L_total]

        // Use fast mode for all chunks if enabled (consistent quality)
        let actualSteps = fastMode ? fastOdeSteps : nTimesteps
        let useCFG = !fastMode  // Skip CFG in fast mode for 2x speedup

        // Cosine time scheduling
        var tSpan: [Float] = []
        for i in 0...actualSteps {
            let linearT = Float(i) / Float(actualSteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        if useCFG {
            // Full CFG path (slower but higher quality)
            let zeroMu = MLXArray.zeros(like: muT)
            let zeroSpk = MLXArray.zeros(like: cachedSpkCond)
            let zeroCond = MLXArray.zeros(like: condsT)

            for step in 1...actualSteps {
                let t = MLXArray([currentT])

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
                if step < actualSteps {
                    dt = tSpan[step + 1] - currentT
                }
            }
        } else {
            // Fast path: no CFG (2x fewer decoder calls)
            for step in 1...actualSteps {
                let t = MLXArray([currentT])
                let v = s3gen.decoder(x: xtVar, mu: muT, t: t, speakerEmb: cachedSpkCond, cond: condsT, mask: mask)
                xtVar = xtVar + v * dt
                currentT = currentT + dt
                if step < actualSteps {
                    dt = tSpan[step + 1] - currentT
                }
            }
        }

        let odeTime = Date().timeIntervalSince(odeStart)

        // 5. Vocoder with context overlap to avoid edge artifacts
        let vocoderStart = Date()

        // Include some context frames before new frames for vocoder
        // Vocoder needs receptive field context to avoid clicking at boundaries
        let vocoderContextFrames = 32  // ~340ms context at 256 samples/frame
        let melStartWithContext = max(0, L_total - L_new - vocoderContextFrames)
        let contextUsed = L_total - L_new - melStartWithContext  // Actual context frames used

        // Extract mel frames with context [B, C, T]
        let melFramesWithContext = xtVar[0..., 0..., melStartWithContext...]

        // Vocoder the extended region
        let wav = s3gen.vocoder(melFramesWithContext)
        eval(wav)

        let allSamples = wav.asArray(Float.self)

        // Discard context audio samples to get only new audio
        let contextAudioSamples = contextUsed * 256
        var newSamples: [Float]
        if contextAudioSamples < allSamples.count {
            newSamples = Array(allSamples[contextAudioSamples...])
        } else {
            newSamples = allSamples
        }

        // Simple approach: trim the end of each chunk to remove clicking artifacts
        // 512 samples = ~21ms at 24kHz - removes vocoder edge effects
        let trimSamples = 512
        if newSamples.count > trimSamples {
            newSamples = Array(newSamples.dropLast(trimSamples))
        }

        generatedSamples.append(contentsOf: newSamples)

        let vocoderTime = Date().timeIntervalSince(vocoderStart)
        let totalTime = Date().timeIntervalSince(startTime)

        print("🔊 S3GenStreaming: \(newTokens.count) tokens → \(newSamples.count) samples (\(String(format: "%.0f", Double(newSamples.count) / 24.0))ms audio) [full ODE: \(L_total) frames, new: \(L_new)]")
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
        isFirstChunk = true  // Reset for next utterance
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
