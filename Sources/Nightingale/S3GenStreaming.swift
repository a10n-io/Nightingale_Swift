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
    }

    /// Cached ODE output (mel spectrogram) for already-generated regions
    private var cachedOdeOutput: MLXArray?

    /// Generate audio for new tokens incrementally using WINDOWED ODE
    /// Re-encodes ALL tokens for quality, but only runs ODE on new frames
    /// Returns ONLY the new audio samples (append to previous)
    /// Generate audio for new tokens incrementally using WINDOWED ODE
    /// Re-encodes ALL tokens for quality, but only runs ODE on new frames
    /// Returns ONLY the new audio samples (append to previous)
    public func generateIncremental(newTokens: [Int]) -> [Float] {
        guard let cachedPromptMel = cachedPromptMel,
              let cachedSpkCond = cachedSpkCond else {
            return []
        }

        let startTime = Date()

        // 1. Accumulate all tokens
        processedTokens.append(contentsOf: newTokens)

        // 2. Re-encode ALL tokens with full bidirectional context
        // This is key for quality - the encoder uses self-attention
        // We run this on the full sequence to get consistent global prosody
        let allTokenArray = MLXArray(processedTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let vocabSize = s3gen.inputEmbedding.weight.shape[0]
        let clippedAll = clip(allTokenArray, min: 0, max: vocabSize - 1)
        let allEmb = s3gen.inputEmbedding(clippedAll)

        let allEncoded = s3gen.encoder(allEmb)  // [1, allLen*2, 512]
        let updatedMu = s3gen.encoderProj(allEncoded)  // [1, allLen*2, 80]
        self.cachedMu = updatedMu

        let encoderTime = Date().timeIntervalSince(startTime)

        // 4. WINDOWED ODE: Run only on relevant window for speed
        let odeStart = Date()

        let L_pm = cachedPromptMel.shape[1]
        let L_total = updatedMu.shape[1]
        let L_new = newTokens.count * 2  // Upsampled 2x
        
        // Define decoding window: Overlap + New Frames
        // We need enough overlap for the diffusion solver to stabilize boundaries
        let overlap = overlapFrames
        
        // Calculate window start (ensure we don't go before start of sequence)
        let previousTotal = L_total - L_new
        let windowStart = max(0, previousTotal - overlap)
        let windowEnd = L_total
        let windowLen = windowEnd - windowStart
        
        // Slice mu for the window [1, windowLen, 80]
        // Note: muT is [B, C, T] usually in decoder, but updatedMu is [B, T, C] from encoderProj?
        // Let's check s3gen.encoderProj(allEncoded).
        // encoder output is [B, T, 512]. encoderProj is Linear(512, 80).
        // So updatedMu is [B, T, 80].
        let muWindow = updatedMu[0..., windowStart..<windowEnd, 0...]
        
        // Prepare conditions for the window
        // condition = promptMel followed by zeros.
        let condWindow: MLXArray
        if windowStart < L_pm {
            if windowEnd <= L_pm {
                // Window entirely within prompt
                condWindow = cachedPromptMel[0..., windowStart..<windowEnd, 0...]
            } else {
                // Window spans prompt end
                let promptPart = cachedPromptMel[0..., windowStart..<L_pm, 0...]
                let zeroLen = windowEnd - L_pm
                let zeroPart = MLXArray.zeros([1, zeroLen, 80], dtype: cachedPromptMel.dtype)
                condWindow = concatenated([promptPart, zeroPart], axis: 1)
            }
        } else {
            // Window entirely in generated region
            condWindow = MLXArray.zeros([1, windowLen, 80], dtype: cachedPromptMel.dtype)
        }

        // Transpose for decoder [B, C, T]
        let condsT = condWindow.transposed(0, 2, 1)
        let muT = muWindow.transposed(0, 2, 1)
        let mask = MLXArray.ones([1, 1, windowLen], dtype: muT.dtype)

        // Slice noise for the window
        // MUST use the same noise positions as full generation for consistency
        var xtVar = s3gen.fixedNoise[0..., 0..., windowStart..<windowEnd]

        // Use fast mode logic
        let actualSteps = fastMode ? fastOdeSteps : nTimesteps
        let useCFG = !fastMode
        
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
            // Full CFG path
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
            // Fast path: no CFG
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

        // 5. Vocoder with context
        let vocoderStart = Date()
        
        // XTVar contains the generated mel spec for `windowStart..<windowEnd`.
        // We want to vocode it.
        // For the Vocoder to produce clean audio for the NEW region, it needs context.
        // `xtVar` already covers `windowStart = previousTotal - overlap`.
        // If overlap (64) is > vocoder context (32), then vocoding `xtVar` is sufficient.
        
        let wav = s3gen.vocoder(xtVar) // [1, 1, windowSamples]
        eval(wav)

        let allSamples = wav.asArray(Float.self)
        
        // Calculate offset to new samples
        // `xtVar` duration is `windowLen` frames = `windowLen * 256` samples (approx).
        // New content starts at frame index `overlap` relative to `xtVar` start.
        // But wait.
        // `windowStart = previousTotal - overlap`
        // New tokens start at `previousTotal` globally.
        // So offset inside `xtVar` is `overlap` frames.
        
        let offsetFrames = (previousTotal - windowStart)
        let offsetSamples = offsetFrames * 256
        
        var newSamples: [Float]
        if offsetSamples < allSamples.count {
             newSamples = Array(allSamples[offsetSamples...])
        } else {
             newSamples = []
        }

        // Trim end artifacts (same as before)
        let trimSamples = 512
        if newSamples.count > trimSamples {
            newSamples = Array(newSamples.dropLast(trimSamples))
        }

        generatedSamples.append(contentsOf: newSamples)

        let vocoderTime = Date().timeIntervalSince(vocoderStart)
        let totalTime = Date().timeIntervalSince(startTime)


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
            return nil
        }
        return S3GenStreaming(s3gen: s3gen)
    }
}
