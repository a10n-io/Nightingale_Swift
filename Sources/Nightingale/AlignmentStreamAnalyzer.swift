import MLX
import Foundation

// MARK: - Alignment Analysis Result

/// Result from alignment analysis for each generation step
public struct AlignmentAnalysisResult {
    /// Was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    public let falseStart: Bool
    /// Was this frame detected as being part of a long tail with potential hallucinations?
    public let longTail: Bool
    /// Was this frame detected as repeating existing text content?
    public let repetition: Bool
    /// Was the alignment position of this frame too far from the previous frame?
    public let discontinuity: Bool
    /// Has inference reached the end of the text tokens?
    public let complete: Bool
    /// Approximate position in the text token sequence
    public let position: Int
}

// MARK: - Alignment Stream Analyzer

/// Monitors attention patterns during T3 generation to enforce text-speech alignment.
///
/// Some transformer TTS models implicitly solve text-speech alignment in specific self-attention
/// heads. This module exploits this to perform online integrity checks while streaming.
///
/// Key heads tracked (layer_idx, head_idx):
/// - Layer 9, Head 2
/// - Layer 12, Head 15
/// - Layer 13, Head 11
///
/// The analyzer acts as a "guardrail" during the generation loop. It does not change the model
/// architecture; it changes the sampling probabilities (logits) based on what the model looked
/// at in the past.
public class AlignmentStreamAnalyzer {

    /// Heads that track text-speech alignment: (layer_index, head_index)
    public static let alignedHeads: [(layer: Int, head: Int)] = [
        (12, 15),  // Layer 12, head 15
        (13, 11),  // Layer 13, head 11
        (9, 2)     // Layer 9, head 2
    ]

    /// Slice of text tokens in the input sequence (start, end)
    let textTokensSlice: (start: Int, end: Int)

    /// Index of the EOS token
    let eosIdx: Int

    /// Accumulated alignment matrix [T, S] where T = speech frames, S = text length
    var alignment: MLXArray?

    /// Current frame position in speech generation
    var currFramePos: Int = 0

    /// Current tracked position in text sequence
    var textPosition: Int = 0

    /// Has generation properly started (not false-starting)?
    var started: Bool = false

    /// Frame position where generation started
    var startedAt: Int?

    /// Has generation reached completion?
    var complete: Bool = false

    /// Frame position where completion was detected
    var completedAt: Int?

    /// Recently generated tokens for repetition detection
    var generatedTokens: [Int] = []

    /// Number of text tokens
    let textLength: Int

    /// Initialize the analyzer
    /// - Parameters:
    ///   - textTokensSlice: (start, end) indices of text tokens in the full sequence
    ///   - eosIdx: Index of the end-of-speech token (default: 6562)
    public init(textTokensSlice: (Int, Int), eosIdx: Int = 6562) {
        self.textTokensSlice = textTokensSlice
        self.eosIdx = eosIdx
        self.textLength = textTokensSlice.1 - textTokensSlice.0

        print("AlignmentStreamAnalyzer: Initialized with text slice (\(textTokensSlice.0), \(textTokensSlice.1)), length=\(textLength)")
    }

    /// Process one generation step
    /// - Parameters:
    ///   - attentionWeights: Dictionary mapping layer indices to attention weight tensors [B, H, T, S]
    ///   - logits: Current logits to potentially modify [vocab_size]
    ///   - nextToken: Last generated token for repetition tracking
    /// - Returns: Potentially modified logits
    public func step(
        attentionWeights: [Int: MLXArray],
        logits: MLXArray,
        nextToken: Int? = nil
    ) -> MLXArray {
        var modifiedLogits = logits

        // Extract and average attention across alignment heads
        // Each attention weight is [B, H, T_q, T_kv]
        // We want specific heads from each layer
        var attentionChunks: [MLXArray] = []

        if currFramePos == 0 {
            print("DEBUG Analyzer: First step - extracting attention from \(attentionWeights.count) layers")
            for (layer, attn) in attentionWeights {
                print("  Layer \(layer): shape \(attn.shape)")
            }
        }

        for (layerIdx, headIdx) in Self.alignedHeads {
            guard let layerAttn = attentionWeights[layerIdx] else {
                print("WARNING: Missing attention for layer \(layerIdx)")
                continue
            }

            // Extract specific head: [B, H, T_q, T_kv] -> [T_q, T_kv]
            // Take batch 0, head headIdx
            let headAttn = layerAttn[0, headIdx, 0..., 0...]  // [T_q, T_kv]
            attentionChunks.append(headAttn)
        }

        guard !attentionChunks.isEmpty else {
            // No attention weights available, skip analysis
            currFramePos += 1
            return modifiedLogits
        }

        // Average across aligned heads
        let stackedAttn = stacked(attentionChunks, axis: 0)  // [3, T_q, T_kv]
        let alignedAttn = mean(stackedAttn, axis: 0)  // [T_q, T_kv]

        // Extract text portion of attention
        let (i, j) = textTokensSlice

        let chunkSlice: MLXArray
        if currFramePos == 0 {
            // First chunk: attention from speech frames to text tokens
            // alignedAttn is [total_seq_len, total_seq_len], we want [speech:, text_start:text_end]
            let speechStart = j  // Speech frames start after text
            if alignedAttn.shape[0] > speechStart {
                chunkSlice = alignedAttn[speechStart..., i..<j]  // [T_speech, S_text]
            } else {
                // Not enough context yet
                currFramePos += 1
                return modifiedLogits
            }
        } else {
            // Subsequent chunks: single frame due to KV-cache
            // alignedAttn is [1, total_seq_len] after KV-cache
            chunkSlice = alignedAttn[0..., i..<j]  // [1, S_text]
        }

        // Monotonic masking: prevent looking at future text positions
        // Zero out attention to positions after current frame position
        let S = chunkSlice.shape[chunkSlice.ndim - 1]
        let monotonic: MLXArray
        if currFramePos + 1 < S {
            // Create mask: [0, 0, 0, ..., -inf, -inf] for positions > currFramePos
            var maskArray = [Float](repeating: 1.0, count: S)
            for idx in (currFramePos + 1)..<S {
                maskArray[idx] = 0.0
            }
            let mask = MLXArray(maskArray).reshaped([1, S])
            monotonic = chunkSlice * broadcast(mask, to: chunkSlice.shape)
        } else {
            monotonic = chunkSlice
        }

        // Accumulate alignment matrix
        if alignment == nil {
            alignment = monotonic
        } else {
            alignment = concatenated([alignment!, monotonic], axis: 0)
        }

        let A = alignment!
        let T = A.shape[0]  // Speech frames

        // Find current text position from attention peak
        // Use RAW (unmasked) attention for position tracking - the masking was causing issues
        let rawLastRow = chunkSlice[(chunkSlice.shape[0]-1)..<chunkSlice.shape[0], 0...]
        let curTextPosn = Int(argMax(rawLastRow, axis: -1).item(Int32.self))

        // Detect discontinuity (jumping too far in text) - more lenient range
        // Allow larger forward jumps since attention can skip ahead
        let discontinuity = !(-4 < curTextPosn - textPosition && curTextPosn - textPosition < 15)

        // DEBUG: Log discontinuity triggers (disabled for clean output)
        // if discontinuity && currFramePos > 3 {
        //     print("⚠️  DISCONTINUITY at frame \(currFramePos): curTextPosn=\(curTextPosn), textPosition=\(textPosition), delta=\(curTextPosn - textPosition)")
        // }

        if !discontinuity {
            textPosition = curTextPosn
        }

        // False start detection: hallucinations at the beginning
        // Check if attention is focused on end of sequence (wrong) vs beginning (correct)
        let falseStart: Bool
        if !started && T >= 2 {
            let lastTwoRows = A[(T-2)..<T, 0...]  // Last 2 frames
            let lastTwoCols = lastTwoRows[0..., (max(0, S-2))..<S]  // Last 2 text positions
            let firstFourCols = A[0..., 0..<min(4, S)]  // First 4 text positions

            let maxAtEnd = Float(lastTwoCols.max().item(Float.self))
            let maxAtStart = Float(firstFourCols.max().item(Float.self))

            falseStart = maxAtEnd > 0.1 || maxAtStart < 0.5
        } else {
            falseStart = false
        }

        started = !falseStart && started || !falseStart
        if started && startedAt == nil {
            startedAt = T
        }

        // Completion check: has model reached end of text?
        if !complete && textPosition >= S - 3 {
            complete = true
            completedAt = T
        }

        // Long tail detection: hallucinations at the end
        var longTail = false
        if complete, let compAt = completedAt, compAt < T {
            let tailSlice = A[compAt..<T, max(0, S-3)..<S]  // After completion, last 3 text positions
            let summed = sum(tailSlice, axis: 0)  // Sum over speech frames
            let maxSum = Float(summed.max().item(Float.self))
            longTail = maxSum >= 5.0  // ~200ms threshold
        }

        // Alignment repetition: attention jumping back to earlier text
        var alignmentRepetition = false
        if complete, let compAt = completedAt, compAt < T && S > 5 {
            let tailSlice = A[compAt..<T, 0..<(S-5)]  // After completion, early text positions
            let maxPerRow = max(tailSlice, axis: 1)  // Max attention per row
            let sumMax = Float(sum(maxPerRow).item(Float.self))
            alignmentRepetition = sumMax > 5.0
        }

        // Token repetition tracking
        if let token = nextToken {
            generatedTokens.append(token)
            // Keep only last 8 tokens
            if generatedTokens.count > 8 {
                generatedTokens = Array(generatedTokens.suffix(8))
            }
        }

        // Check for excessive token repetition (infinite loops)
        // CRITICAL FINDING: Python FP32 model generates up to 5x consecutive repetitions at temp=0.8
        // Test: "Hello, this is a test of the Swift text to speech system."
        //   - Python: 131 tokens, token 6405 repeats 5x at positions 125-129
        //   - This is NORMAL behavior for high-temperature sampling
        // Q4 quantization causes ~1.8 logit drift, leading to different token sequences
        // Q4 can get stuck in repetition patterns that FP32 doesn't experience
        // Threshold: 8 consecutive identical tokens (generous margin to allow Q4 to potentially break out)
        // If still stuck at 8x, the Q4 model is genuinely in an infinite loop
        let tokenRepetition: Bool
        if generatedTokens.count >= 8 {
            let lastEight = Array(generatedTokens.suffix(8))
            tokenRepetition = Set(lastEight).count == 1
        } else {
            tokenRepetition = false
        }

        if tokenRepetition {
            print("🚨 AlignmentStreamAnalyzer: Detected 8x repetition of token \(generatedTokens.last ?? -1)")
            print("   Token sequence (\(generatedTokens.count) tokens): \(generatedTokens)")
        }

        // ============================================
        // LOGIT MODIFICATION
        // ============================================

        // DISABLED: EOS suppression causes issues when textPos tracking isn't working.
        // For English models, we let the model generate EOS naturally and only use
        // token repetition detection as a fallback safety mechanism.
        //
        // Original code (for multilingual):
        // if curTextPosn < S - 3 && S > 5 {
        //     var logitsArray = modifiedLogits.asArray(Float.self)
        //     logitsArray[eosIdx] = -32768.0  // -2^15
        //     modifiedLogits = MLXArray(logitsArray)
        // }

        // Force EOS on detected errors
        if longTail || alignmentRepetition || tokenRepetition {
            print("⚠️ AlignmentStreamAnalyzer: Forcing EOS - longTail=\(longTail), alignmentRepetition=\(alignmentRepetition), tokenRepetition=\(tokenRepetition)")
            // Set all logits to -inf except EOS
            var logitsArray = [Float](repeating: -32768.0, count: modifiedLogits.shape[0])
            logitsArray[eosIdx] = 32768.0  // +2^15
            modifiedLogits = MLXArray(logitsArray)
        }

        currFramePos += 1

        return modifiedLogits
    }

    /// Reset the analyzer for a new generation
    public func reset() {
        alignment = nil
        currFramePos = 0
        textPosition = 0
        started = false
        startedAt = nil
        complete = false
        completedAt = nil
        generatedTokens = []
    }

    /// Get the current analysis result
    public func getResult() -> AlignmentAnalysisResult {
        return AlignmentAnalysisResult(
            falseStart: !started,
            longTail: complete && completedAt != nil,
            repetition: generatedTokens.count >= 2 && Set(generatedTokens.suffix(2)).count == 1,
            discontinuity: false,
            complete: complete,
            position: textPosition
        )
    }
}
