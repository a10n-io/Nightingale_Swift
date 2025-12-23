import MLX
import MLXNN
import MLXRandom
import Foundation

// MARK: - RMS Normalization

public class RMSNorm: Module, UnaryLayer {
    var weight: MLXArray  // var to allow weight loading
    let eps: Float
    let epsArray: MLXArray  // Pre-computed eps as MLXArray for precision

    public init(dims: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
        self.epsArray = MLXArray(eps)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Match PyTorch's LlamaRMSNorm exactly:
        // 1. Convert to Float32
        // 2. Compute variance and normalize
        // 3. Convert back before weight multiply
        let inputDtype = x.dtype
        var hidden = x.asType(.float32)

        let variance = hidden.pow(2).mean(axis: -1, keepDims: true)
        hidden = hidden * rsqrt(variance + epsArray)
        hidden = hidden.asType(inputDtype)

        // Debug: print dtypes for first call
        #if DEBUG
        struct DebugState {
            static var firstCall = true
        }
        if DebugState.firstCall {
            print("[RMSNorm DEBUG] input dtype: \(inputDtype), variance dtype: \(variance.dtype), weight dtype: \(weight.dtype), eps: \(eps)")
            DebugState.firstCall = false
        }
        #endif

        return weight * hidden
    }
}

// MARK: - Rotary Position Embedding (Llama-style)

public class RoPE: Module {
    let dims: Int
    let traditional: Bool
    let base: Float
    let scale: Float

    // Pre-computed frequency tables for Llama3 RoPE
    public let cosTable: MLXArray?
    public let sinTable: MLXArray?

    public init(dims: Int, traditional: Bool = false, base: Float = 10000, scale: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.cosTable = nil
        self.sinTable = nil
        super.init()
    }

    /// Initialize with pre-computed Llama3 RoPE frequency tables
    public init(loadFrequenciesFrom url: URL) throws {
        // Load pre-computed cos/sin tables
        let tables = try MLX.loadArrays(url: url)

        guard let cosTable = tables["rope_cos_table"],
              let sinTable = tables["rope_sin_table"] else {
            throw NSError(domain: "RoPE", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing cos/sin tables in file"])
        }

        self.cosTable = cosTable
        self.sinTable = sinTable

        // Extract metadata
        let headDimArray = tables["head_dim"]?.asArray(Int32.self) ?? [64]
        self.dims = Int(headDimArray[0])

        let thetaArray = tables["rope_theta"]?.asArray(Float.self) ?? [500000.0]
        self.base = thetaArray[0]

        self.traditional = false
        self.scale = 1.0

        super.init()

        print("[RoPE] Loaded pre-computed Llama3 frequencies:")
        print("  cos_table: \(cosTable.shape)")
        print("  sin_table: \(sinTable.shape)")
        print("  theta (base): \(self.base)")
        print("  dims: \(self.dims)")
    }

    /// Apply rotary position embedding to input tensor
    /// x shape: [Batch, Heads, SeqLen, HeadDim]
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let shape = x.shape
        let seqLen = shape[shape.count - 2]
        let headDim = shape[shape.count - 1]

        // Get cos/sin values for the sequence
        let cosTheta: MLXArray
        let sinTheta: MLXArray

        if let cosTable = self.cosTable, let sinTable = self.sinTable {
            // Use pre-computed tables (Llama3 RoPE with scaling)
            // cosTable and sinTable are [MaxPositions, HeadDim]
            // Extract rows for positions [offset, offset+1, ..., offset+seqLen-1]

            let positionIndices = Array(offset..<(offset + seqLen)).map { Int32($0) }
            let positionArray = MLXArray(positionIndices)

            // Index into tables to get [SeqLen, HeadDim]
            cosTheta = cosTable[positionArray]
            sinTheta = sinTable[positionArray]
        } else {
            // Fallback: Compute on-the-fly (original method, may not match Llama3 scaling)
            // Generate position indices: [SeqLen]
            let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Float($0) })

            // Generate inverse frequencies: [HeadDim/2]
            // invFreq[i] = 1.0 / (base ^ (2i / dim))
            let halfDim = headDim / 2
            var invFreqArray: [Float] = []
            for i in 0..<halfDim {
                let freq = 1.0 / pow(base, Float(2 * i) / Float(headDim))
                invFreqArray.append(freq)
            }
            let invFreq = MLXArray(invFreqArray)

            // Compute angles: positions outer product with invFreq
            // positions: [SeqLen] -> [SeqLen, 1]
            // invFreq: [HalfDim] -> [1, HalfDim]
            // angles: [SeqLen, HalfDim]
            let posExpanded = positions.expandedDimensions(axis: 1)
            let freqExpanded = invFreq.expandedDimensions(axis: 0)
            let angles = posExpanded * freqExpanded * scale

            // Compute cos and sin: [SeqLen, HalfDim]
            let cosAngles = MLX.cos(angles)
            let sinAngles = MLX.sin(angles)

            // Duplicate to full dim: [SeqLen, HeadDim]
            // cos/sin are repeated for both halves: [cos, cos] pattern for interleaved
            cosTheta = concatenated([cosAngles, cosAngles], axis: -1)
            sinTheta = concatenated([sinAngles, sinAngles], axis: -1)
        }

        // Reshape for broadcasting: [1, 1, SeqLen, HeadDim]
        let cosBroadcast = cosTheta.reshaped([1, 1, seqLen, headDim])
        let sinBroadcast = sinTheta.reshaped([1, 1, seqLen, headDim])

        // Apply rotation: (x * cos) + (rotate_half(x) * sin)
        let rotatedX = rotateHalf(x)
        return (x * cosBroadcast) + (rotatedX * sinBroadcast)
    }

    /// Rotate half: splits tensor and rotates
    /// Input [x1, x2] -> Output [-x2, x1]
    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let lastDim = x.shape[x.ndim - 1]
        let halfDim = lastDim / 2

        // Split into two halves along last dimension
        // x1 = x[..., :halfDim], x2 = x[..., halfDim:]
        let x1 = x[0..., 0..., 0..., ..<halfDim]
        let x2 = x[0..., 0..., 0..., halfDim...]

        // Concatenate [-x2, x1]
        return concatenated([-x2, x1], axis: -1)
    }
}

// MARK: - Attention Debug State

// Global flags to ensure Layer 0 attention probes run only once
// MARK: - Attention

public class Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo public var qProj: Linear
    @ModuleInfo public var kProj: Linear
    @ModuleInfo public var vProj: Linear
    @ModuleInfo public var oProj: Linear
    public let rope: RoPE

    /// Initialize with weights dictionary for quantized loading
    public init(config: T3Config, layerPrefix: String, weights: [String: MLXArray], rope: RoPE? = nil) {
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        // Use LinearFactory to detect quantized vs FP16 weights
        self.qProj = LinearFactory.load("\(layerPrefix).selfAttn.qProj", inputDim: hiddenSize, outputDim: nHeads * headDim, weights: weights, bias: false)
        self.kProj = LinearFactory.load("\(layerPrefix).selfAttn.kProj", inputDim: hiddenSize, outputDim: nKVHeads * headDim, weights: weights, bias: false)
        self.vProj = LinearFactory.load("\(layerPrefix).selfAttn.vProj", inputDim: hiddenSize, outputDim: nKVHeads * headDim, weights: weights, bias: false)
        self.oProj = LinearFactory.load("\(layerPrefix).selfAttn.oProj", inputDim: nHeads * headDim, outputDim: hiddenSize, weights: weights, bias: false)

        // Use shared RoPE if provided, otherwise create new one
        self.rope = rope ?? RoPE(dims: headDim, base: config.ropeTheta)

        super.init()
    }

    /// Legacy init without weights (random initialization)
    public init(config: T3Config) {
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize
        self.qProj = Linear(hiddenSize, nHeads * headDim, bias: false)
        self.kProj = Linear(hiddenSize, nKVHeads * headDim, bias: false)
        self.vProj = Linear(hiddenSize, nKVHeads * headDim, bias: false)
        self.oProj = Linear(nHeads * headDim, hiddenSize, bias: false)
        self.rope = RoPE(dims: headDim, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape to [B, L, nHeads, headDim] then transpose to [B, nHeads, L, headDim]
        queries = queries.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.offset ?? 0

        // 🔬 ROPE DIAGNOSTIC: Check Q, K, V before and after RoPE
        if offset == 0 && L == 80 {  // Only for initial prefill pass
            eval(queries, keys, values)
            let q_pre = queries[0, 0, L-1, 0..<5]  // Batch 0, Head 0, Last token, First 5
            let k_pre = keys[0, 0, L-1, 0..<5]
            let k_0_pre = keys[0, 0, 0, 0..<5]  // First token
            let v_pre = values[0, 0, L-1, 0..<5]
            eval(q_pre, k_pre, k_0_pre, v_pre)
            print("\n🔬 PRE-ROPE (Batch 0, Head 0, Last Token [79]):")
            print("   Q [:5]: \(q_pre.asArray(Float.self))")
            print("   K [:5]: \(k_pre.asArray(Float.self))")
            print("   V [:5]: \(v_pre.asArray(Float.self))")
            print("\n🔬 PRE-ROPE (Batch 0, Head 0, First Token [0]):")
            print("   K [:5]: \(k_0_pre.asArray(Float.self))")
        }

        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update cache
        if let cache = cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Repeat KV heads if needed (GQA)
        if nKVHeads < nHeads {
            let repeats = nHeads / nKVHeads
            keys = MLX.repeated(keys, count: repeats, axis: 1)
            values = MLX.repeated(values, count: repeats, axis: 1)
        }

        // Compute attention
        // Match Python exactly: (Q * scale) @ K^T
        // Python: scores_pre = (q * scale) @ k.transpose(-1, -2)
        var scores = (queries * scale).matmul(keys.transposed(0, 1, 3, 2))

        // 🔬 PRECISION DIAGNOSTIC: Check dtypes
        if offset == 0 && L == 80 {
            eval(queries, keys, scores)
            print("\n🔬 PRECISION CHECK:")
            print("   Q dtype: \(queries.dtype)")
            print("   K dtype: \(keys.dtype)")
            print("   scores dtype: \(scores.dtype)")
            print("   scale value: \(scale)")
        }

        // 🔬 MANUAL DOT PRODUCT VERIFICATION
        if offset == 0 && L == 80 {
            // Extract q[79] and k[0] for manual computation
            let q_vec = queries[0, 0, L-1]  // [64] - Batch 0, Head 0, Token 79
            let k_vec = keys[0, 0, 0]       // [64] - Batch 0, Head 0, Token 0
            eval(q_vec, k_vec)

            let q_arr = q_vec.asArray(Float.self)
            let k_arr = k_vec.asArray(Float.self)

            // Manual dot product: sum(q[i] * k[i])
            var manual_dot: Float = 0.0
            for i in 0..<headDim {
                manual_dot += q_arr[i] * k_arr[i]
            }
            let manual_score = manual_dot * scale

            let mlx_score = scores[0, 0, L-1, 0].item(Float.self)

            print("\n🔬 MANUAL DOT PRODUCT VERIFICATION (Token 79, Pos 0):")
            print("   q[79][:5]: \(Array(q_arr.prefix(5)))")
            print("   k[0][:5]:  \(Array(k_arr.prefix(5)))")
            print("   Manual dot (unscaled): \(manual_dot)")
            print("   Manual score (scaled):  \(manual_score)")
            print("   MLX matmul result:      \(mlx_score)")
            print("   Difference: \(abs(manual_score - mlx_score))")
            print("   Python reference: 1.4127315283")
        }

        // 🔬 MASK DIAGNOSTIC: Check mask and scores
        if offset == 0 && L == 80 {
            eval(scores)
            let scores_pre = scores[0, 0, L-1, 0..<5]  // Batch 0, Head 0, Last token, First 5
            eval(scores_pre)
            print("\n🔬 SCORES (PRE-MASK) [B0, H0, T79, :5]:")
            print("   \(scores_pre.asArray(Float.self))")

            if let m = mask {
                eval(m)
                print("\n🔬 MASK:")
                print("   Shape: \(m.shape)")
                let mask_first = m[0, 0, 0, 0..<5]
                let mask_last = m[0, 0, L-1, 0..<5]
                eval(mask_first, mask_last)
                print("   First row [0,0,0,:5]: \(mask_first.asArray(Float.self))")
                print("   Last row [0,0,79,:5]: \(mask_last.asArray(Float.self))")
            } else {
                print("\n🔬 MASK: nil (no mask applied)")
            }
        }

        if let mask = mask {
            scores = scores + mask
        }

        // 🔬 MASK DIAGNOSTIC: Check scores after mask
        if offset == 0 && L == 80 {
            eval(scores)
            let scores_post = scores[0, 0, L-1, 0..<5]
            eval(scores_post)
            print("\n🔬 SCORES (POST-MASK) [B0, H0, T79, :5]:")
            print("   \(scores_post.asArray(Float.self))")
        }

        // Check scores for Step 1, Layer 0

        let weights = softmax(scores, axis: -1)

        // 🔬 SOFTMAX DIAGNOSTIC: Check attention weights (probabilities)
        if offset == 0 && L == 80 {
            eval(weights)
            let weights_sample = weights[0, 0, L-1, 0..<5]  // Batch 0, Head 0, Last token, First 5
            let weights_row = weights[0, 0, L-1]  // All 80 positions
            let weights_sum = weights_row.sum()  // Should be 1.0
            eval(weights_sample, weights_sum)
            print("\n🔬 ATTENTION WEIGHTS (POST-SOFTMAX) [B0, H0, T79, :5]:")
            print("   \(weights_sample.asArray(Float.self))")
            print("   Sum over all positions: \(weights_sum.item(Float.self))")
        }

        if offset == 80 && L == 1 {
        }

        var output = weights.matmul(values)

        // 🔬 CHECKPOINT 5: After value aggregation (before reshape)
        if offset == 0 && L == 80 {
            eval(output)
            let agg_sample = output[0, 0, L-1, 0..<5]  // [B, nHeads, L, headDim]
            eval(agg_sample)
            print("\n🔬 CHECKPOINT 5: After Weights @ Values [B0, H0, T79, :5]:")
            print("   \(agg_sample.asArray(Float.self))")
        }

        // Reshape back
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, nHeads * headDim])

        // 🔬 CHECKPOINT 6: After reshape (before oProj)
        if offset == 0 && L == 80 {
            eval(output)
            let reshaped_sample = output[0, L-1, 0..<5]  // [B, L, hiddenDim]
            eval(reshaped_sample)
            print("\n🔬 CHECKPOINT 6: After Reshape [B0, T79, :5]:")
            print("   \(reshaped_sample.asArray(Float.self))")
        }

        let finalOutput = oProj(output)

        // 🔬 CHECKPOINT 7: After oProj (final attention output)
        if offset == 0 && L == 80 {
            eval(finalOutput)
            let final_sample = finalOutput[0, L-1, 0..<5]
            eval(final_sample)
            print("\n🔬 CHECKPOINT 7: After oProj (Final) [B0, T79, :5]:")
            print("   \(final_sample.asArray(Float.self))")
        }

        return finalOutput
    }
}

// MARK: - Inspectable Attention (for alignment tracking)

/// Attention layer that captures attention weights for alignment analysis.
/// Used for layers 9, 12, 13 which track text-speech alignment.
public class InspectableAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let layerIdx: Int

    @ModuleInfo public var qProj: Linear
    @ModuleInfo public var kProj: Linear
    @ModuleInfo public var vProj: Linear
    @ModuleInfo public var oProj: Linear
    public let rope: RoPE

    /// Last captured attention weights [B, nHeads, T_q, T_kv]
    public var lastAttentionWeights: MLXArray?

    /// Initialize with weights dictionary for quantized loading
    public init(config: T3Config, layerPrefix: String, weights: [String: MLXArray], layerIdx: Int, rope: RoPE? = nil) {
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.layerIdx = layerIdx

        let hiddenSize = config.hiddenSize

        // Use LinearFactory to detect quantized vs FP16 weights
        self.qProj = LinearFactory.load("\(layerPrefix).selfAttn.qProj", inputDim: hiddenSize, outputDim: nHeads * headDim, weights: weights, bias: false)
        self.kProj = LinearFactory.load("\(layerPrefix).selfAttn.kProj", inputDim: hiddenSize, outputDim: nKVHeads * headDim, weights: weights, bias: false)
        self.vProj = LinearFactory.load("\(layerPrefix).selfAttn.vProj", inputDim: hiddenSize, outputDim: nKVHeads * headDim, weights: weights, bias: false)
        self.oProj = LinearFactory.load("\(layerPrefix).selfAttn.oProj", inputDim: nHeads * headDim, outputDim: hiddenSize, weights: weights, bias: false)

        // Use shared RoPE if provided, otherwise create new one
        self.rope = rope ?? RoPE(dims: headDim, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape to [B, L, nHeads, headDim] then transpose to [B, nHeads, L, headDim]
        queries = queries.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        // Apply RoPE
        let offset = cache?.offset ?? 0

        // 🔬 ROPE DIAGNOSTIC: Check Q, K, V before and after RoPE
        if offset == 0 && L == 80 {  // Only for initial prefill pass
            eval(queries, keys, values)
            let q_pre = queries[0, 0, L-1, 0..<5]  // Batch 0, Head 0, Last token, First 5
            let k_pre = keys[0, 0, L-1, 0..<5]
            let k_0_pre = keys[0, 0, 0, 0..<5]  // First token
            let v_pre = values[0, 0, L-1, 0..<5]
            eval(q_pre, k_pre, k_0_pre, v_pre)
            print("\n🔬 PRE-ROPE (Batch 0, Head 0, Last Token [79]):")
            print("   Q [:5]: \(q_pre.asArray(Float.self))")
            print("   K [:5]: \(k_pre.asArray(Float.self))")
            print("   V [:5]: \(v_pre.asArray(Float.self))")
            print("\n🔬 PRE-ROPE (Batch 0, Head 0, First Token [0]):")
            print("   K [:5]: \(k_0_pre.asArray(Float.self))")
        }

        // DEBUG: Log offset every 10 steps and around the suspected freeze point (frame 70-90)
        let shouldLog = (offset % 10 == 0) || (offset >= 70 && offset <= 90)
        if shouldLog {
            print("🔍 RoPE offset=\(offset), seqLen=\(L)")
        }

        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // 🔬 ROPE DIAGNOSTIC: Check Q, K after RoPE
        if offset == 0 && L == 80 {
            eval(queries, keys)
            let q_post = queries[0, 0, L-1, 0..<5]
            let k_post = keys[0, 0, L-1, 0..<5]
            let k_0_post = keys[0, 0, 0, 0..<5]  // CRITICAL: Check position 0 too!
            eval(q_post, k_post, k_0_post)
            print("\n🔬 POST-ROPE (Batch 0, Head 0, Last Token [79]):")
            print("   Q [:5]: \(q_post.asArray(Float.self))")
            print("   K [:5]: \(k_post.asArray(Float.self))")
            print("\n🔬 POST-ROPE (Batch 0, Head 0, First Token [0]):")
            print("   K [:5]: \(k_0_post.asArray(Float.self))")
        }

        // Update cache
        if let cache = cache {
            let oldOffset = cache.offset
            (keys, values) = cache.update(keys: keys, values: values)
            let newOffset = cache.offset

            if shouldLog {
                print("🔍 KV cache: oldOffset=\(oldOffset) → newOffset=\(newOffset) (delta=\(newOffset - oldOffset))")
            }
        }

        // Repeat KV heads if needed (GQA)
        var expandedKeys = keys
        var expandedValues = values
        if nKVHeads < nHeads {
            let repeats = nHeads / nKVHeads
            expandedKeys = MLX.repeated(keys, count: repeats, axis: 1)
            expandedValues = MLX.repeated(values, count: repeats, axis: 1)
        }

        // Compute attention scores (NOT using fast SDPA - we need the weights!)
        var scores = (queries * scale).matmul(expandedKeys.transposed(0, 1, 3, 2))

        if let mask = mask {
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1)

        // CAPTURE attention weights for alignment analysis
        // Shape: [B, nHeads, T_q, T_kv]
        self.lastAttentionWeights = weights

        var output = weights.matmul(expandedValues)

        // Reshape back
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, nHeads * headDim])
        return oProj(output)
    }

    /// Clear captured attention (call after each generation step to free memory)
    public func clearAttention() {
        lastAttentionWeights = nil
    }
}

// MARK: - MLP

public class MLP: Module {
    @ModuleInfo public var gateProj: Linear
    @ModuleInfo public var upProj: Linear
    @ModuleInfo public var downProj: Linear

    /// Initialize with weights dictionary for quantized loading
    public init(config: T3Config, layerPrefix: String, weights: [String: MLXArray]) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self.gateProj = LinearFactory.load("\(layerPrefix).mlp.gateProj", inputDim: hiddenSize, outputDim: intermediateSize, weights: weights, bias: false)
        self.upProj = LinearFactory.load("\(layerPrefix).mlp.upProj", inputDim: hiddenSize, outputDim: intermediateSize, weights: weights, bias: false)
        self.downProj = LinearFactory.load("\(layerPrefix).mlp.downProj", inputDim: intermediateSize, outputDim: hiddenSize, weights: weights, bias: false)

        super.init()
    }

    /// Legacy init without weights (random initialization)
    public init(config: T3Config) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self.gateProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.upProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.downProj = Linear(intermediateSize, hiddenSize, bias: false)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SiLU(gate) * up
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

// MARK: - Transformer Block Protocol

/// Protocol for transformer blocks (regular and inspectable)
public protocol TransformerBlockProtocol {
    func forward(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray
}

// MARK: - Transformer Block

public class TransformerBlock: Module, TransformerBlockProtocol {
    public let selfAttn: Attention
    public let mlp: MLP
    public let inputLayernorm: RMSNorm
    public let postAttentionLayernorm: RMSNorm

    /// Initialize with weights dictionary for quantized loading
    public init(config: T3Config, layerIndex: Int, weights: [String: MLXArray], rope: RoPE? = nil) {
        let prefix = "layers.\(layerIndex)"
        self.selfAttn = Attention(config: config, layerPrefix: prefix, weights: weights, rope: rope)
        self.mlp = MLP(config: config, layerPrefix: prefix, weights: weights)
        self.inputLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.postAttentionLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        // Load norm weights (FP16, not quantized)
        if let inputNormWeight = weights["\(prefix).inputLayernorm.weight"] {
            self.inputLayernorm.weight = inputNormWeight
        }
        if let postNormWeight = weights["\(prefix).postAttentionLayernorm.weight"] {
            self.postAttentionLayernorm.weight = postNormWeight
        }

        super.init()
    }

    /// Legacy init without weights (random initialization)
    public init(config: T3Config) {
        self.selfAttn = Attention(config: config)
        self.mlp = MLP(config: config)
        self.inputLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.postAttentionLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Pre-norm attention
        let normedInput = inputLayernorm(x)

        // 🔬 CHECKPOINT 1: LayerNorm Output (only for Layer 0, initial pass)
        if cache?.offset == 0 && x.shape[1] == 80 {
            eval(normedInput)
            let norm_sample = normedInput[0, 0, 0..<5]  // Batch 0, Token 0 (position 0), First 5 dims
            eval(norm_sample)
            print("\n🔬 CHECKPOINT 1: LayerNorm Output [B0, T0, :5]:")
            print("   \(norm_sample.asArray(Float.self))")
        }

        let attnOut = selfAttn(normedInput, mask: mask, cache: cache)
        let h = x + attnOut  // First residual connection

        // Pre-norm MLP
        let mlpInput = postAttentionLayernorm(h)

        // 🔬 CHECKPOINT 8: MLP Input (post-norm)
        if cache?.offset == 0 && x.shape[1] == 80 {
            eval(mlpInput)
            let mlp_in_sample = mlpInput[0, 79, 0..<5]  // Batch 0, Token 79, First 5
            eval(mlp_in_sample)
            print("\n🔬 CHECKPOINT 8: MLP Input (Post-Norm) [B0, T79, :5]:")
            print("   \(mlp_in_sample.asArray(Float.self))")
        }

        let mlpOutput = mlp(mlpInput)

        // 🔬 CHECKPOINT 9: MLP Output (before residual)
        if cache?.offset == 0 && x.shape[1] == 80 {
            eval(mlpOutput)
            let mlp_out_sample = mlpOutput[0, 79, 0..<5]
            eval(mlp_out_sample)
            print("\n🔬 CHECKPOINT 9: MLP Output (Before Residual) [B0, T79, :5]:")
            print("   \(mlp_out_sample.asArray(Float.self))")
        }

        let finalOutput = h + mlpOutput  // Second residual connection

        // 🔬 CHECKPOINT 10: Layer 0 Final Output
        if cache?.offset == 0 && x.shape[1] == 80 {
            eval(finalOutput)
            let final_sample = finalOutput[0, 79, 0..<5]
            eval(final_sample)
            print("\n🔬 CHECKPOINT 10: Layer 0 Final Output [B0, T79, :5]:")
            print("   \(final_sample.asArray(Float.self))")
        }

        return finalOutput
    }

    public func forward(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        return callAsFunction(x, mask: mask, cache: cache)
    }
}

// MARK: - Inspectable Transformer Block (for alignment tracking)

/// Transformer block that uses InspectableAttention for capturing attention weights.
/// Used for layers 9, 12, 13 which track text-speech alignment.
public class InspectableTransformerBlock: Module, TransformerBlockProtocol {
    let selfAttn: InspectableAttention
    let mlp: MLP
    let inputLayernorm: RMSNorm
    let postAttentionLayernorm: RMSNorm
    let layerIdx: Int

    /// Initialize with weights dictionary for quantized loading
    public init(config: T3Config, layerIndex: Int, weights: [String: MLXArray], rope: RoPE? = nil) {
        self.layerIdx = layerIndex
        let prefix = "layers.\(layerIndex)"
        self.selfAttn = InspectableAttention(config: config, layerPrefix: prefix, weights: weights, layerIdx: layerIndex, rope: rope)
        self.mlp = MLP(config: config, layerPrefix: prefix, weights: weights)
        self.inputLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.postAttentionLayernorm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        // Load norm weights (FP16, not quantized)
        if let inputNormWeight = weights["\(prefix).inputLayernorm.weight"] {
            self.inputLayernorm.weight = inputNormWeight
        }
        if let postNormWeight = weights["\(prefix).postAttentionLayernorm.weight"] {
            self.postAttentionLayernorm.weight = postNormWeight
        }

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Pre-norm attention
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        // Pre-norm MLP
        return h + mlp(postAttentionLayernorm(h))
    }

    public func forward(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        return callAsFunction(x, mask: mask, cache: cache)
    }

    /// Get last captured attention weights
    public var lastAttentionWeights: MLXArray? {
        return selfAttn.lastAttentionWeights
    }

    /// Clear captured attention to free memory
    public func clearAttention() {
        selfAttn.clearAttention()
    }
}

// MARK: - KV Cache with Sink Token Support (StreamingLLM)

public class KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int = 0

    // Sink Token / Rolling Cache Configuration
    let sinkTokens: Int      // Number of anchor tokens to always keep (attention sink)
    let maxCacheSize: Int    // Maximum total cache size before eviction

    public init(sinkTokens: Int = 4, maxCacheSize: Int = 512) {
        self.sinkTokens = sinkTokens
        self.maxCacheSize = maxCacheSize
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, newKeys], axis: 2)
            values = concatenated([existingValues, newValues], axis: 2)

            // Apply rolling eviction if cache exceeds max size
            let currentSize = keys!.shape[2]

            if currentSize > maxCacheSize {
                // Keep sink tokens (0..sinkTokens) + recent tokens
                let recentStart = currentSize - (maxCacheSize - sinkTokens)

                // Sink tokens (the "anchor" - first N tokens)
                let sinkKeys = keys![0..., 0..., 0..<sinkTokens, 0...]
                let sinkValues = values![0..., 0..., 0..<sinkTokens, 0...]

                // Recent tokens (rolling window)
                let recentKeys = keys![0..., 0..., recentStart..<currentSize, 0...]
                let recentValues = values![0..., 0..., recentStart..<currentSize, 0...]

                // Concatenate: [sink | recent]
                keys = concatenated([sinkKeys, recentKeys], axis: 2)
                values = concatenated([sinkValues, recentValues], axis: 2)

                print("KVCache: Evicted \(currentSize - maxCacheSize) tokens, keeping \(sinkTokens) sink + \(maxCacheSize - sinkTokens) recent")
            }
        } else {
            keys = newKeys
            values = newValues
        }
        offset += newKeys.shape[2]
        return (keys!, values!)
    }

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }
}

// MARK: - Learned Position Embeddings

public class LearnedPositionEmbeddings: Module {
    public let embedding: Embedding

    public init(maxLen: Int, dims: Int) {
        self.embedding = Embedding(embeddingCount: maxLen, dimensions: dims)
        super.init()
    }

    public func callAsFunction(_ positions: MLXArray) -> MLXArray {
        return embedding(positions)
    }

    public func getFixedEmbedding(_ position: Int) -> MLXArray {
        return embedding(MLXArray([Int32(position)])).expandedDimensions(axis: 0)
    }
}

// MARK: - T3 Model

public class T3Model: Module {
    public let config: T3Config

    // Transformer layers - using Module array with protocol conformance
    // Layers 9, 12, 13 are InspectableTransformerBlock, others are TransformerBlock
    public let layers: [Module]
    public let norm: RMSNorm

    // Track inspectable layer indices for alignment analysis
    // Key: layer index, Value: position in layers array (same in this case)
    public static let inspectableLayerIndices: Set<Int> = [9, 12, 13]

    // Embeddings
    public let textEmb: Embedding
    public let speechEmb: Embedding
    public let textPosEmb: LearnedPositionEmbeddings
    public let speechPosEmb: LearnedPositionEmbeddings

    // Output heads
    @ModuleInfo public var speechHead: Linear

    // Conditioning encoder (T3CondEnc equivalent)
    @ModuleInfo public var speakerProj: Linear      // Projects 256-dim speaker embedding to model hidden size
    public let perceiver: Perceiver?    // Compresses 150 speech tokens to 32 tokens
    @ModuleInfo public var emotionAdvFC: Linear?    // Emotion adversarial conditioning (1 -> 1024)

    // Special tokens
    public let startSpeechToken: Int = 6561  // Fixed: was 6560 (incorrect)
    public let stopSpeechToken: Int = 6562

    /// Initialize with weights dictionary for quantized loading
    /// This loads weights directly during construction, supporting both quantized (4-bit) and FP16 layers.
    public init(config: T3Config, weights: [String: MLXArray], ropeFreqsURL: URL? = nil) {
        self.config = config

        // Analyze weights
        let stats = LinearFactory.analyzeWeights(weights)
        print("T3Model: Loading weights - \(stats.quantized) quantized, \(stats.fp16) FP16, \(stats.other) other")

        // Create shared RoPE instance with pre-computed Llama3 frequencies
        let sharedRoPE: RoPE?
        if let ropeURL = ropeFreqsURL {
            do {
                sharedRoPE = try RoPE(loadFrequenciesFrom: ropeURL)
                print("T3Model: Using pre-computed Llama3 RoPE frequencies from \(ropeURL.lastPathComponent)")
            } catch {
                print("⚠️  Failed to load RoPE frequencies: \(error)")
                print("   Falling back to on-the-fly computation (may not match Python)")
                sharedRoPE = nil
            }
        } else {
            print("⚠️  No RoPE frequency table provided - using on-the-fly computation")
            print("   This may not match Python's Llama3 RoPE scaling!")
            sharedRoPE = nil
        }

        // Build transformer with weights
        // Use InspectableTransformerBlock for layers 9, 12, 13 (alignment tracking)
        self.layers = (0..<config.numHiddenLayers).map { i in
            if Self.inspectableLayerIndices.contains(i) {
                print("T3Model: Using InspectableTransformerBlock for layer \(i)")
                return InspectableTransformerBlock(config: config, layerIndex: i, weights: weights, rope: sharedRoPE) as Module
            } else {
                return TransformerBlock(config: config, layerIndex: i, weights: weights, rope: sharedRoPE) as Module
            }
        }
        self.norm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        // Load final norm weight
        print("🔍 Loading final norm weight..."); fflush(stdout)
        print("   Available keys containing 'norm': \(weights.keys.filter { $0.contains("norm") })")
        fflush(stdout)
        print("   About to access weights[\"norm.weight\"]..."); fflush(stdout)
        if let normWeight = weights["norm.weight"] {
            print("   Successfully got normWeight"); fflush(stdout)
            print("   ✅ Found 'norm.weight' key, shape: \(normWeight.shape), dtype: \(normWeight.dtype)")
            eval(normWeight)
            print("   normWeight range: [\(normWeight.min().item(Float.self)), \(normWeight.max().item(Float.self))]")
            print("   Current norm.weight shape: \(self.norm.weight.shape)")

            // Assign the weight
            self.norm.weight = normWeight
            print("   ✅ Weight assigned successfully")

            eval(self.norm.weight)
            print("   ✅ Eval completed")

            // Try to access the data safely
            do {
                let firstFive = try self.norm.weight[0..<min(5, self.norm.weight.shape[0])].asArray(Float.self)
                print("   After assignment - norm.weight[:5]: \(firstFive)")
                let meanVal = self.norm.weight.mean().item(Float.self)
                let sumVal = self.norm.weight.sum().item(Float.self)
                print("   After assignment - mean: \(meanVal), sum: \(sumVal)")
            } catch {
                print("   ⚠️  Error accessing norm weight values: \(error)")
            }
        } else {
            print("   ❌ 'norm.weight' key NOT FOUND in weights dictionary!")
        }

        // Embeddings - these are FP16 (not quantized)
        self.textEmb = Embedding(embeddingCount: config.textVocabSize, dimensions: config.hiddenSize)
        self.speechEmb = Embedding(embeddingCount: config.speechVocabSize, dimensions: config.hiddenSize)
        self.textPosEmb = LearnedPositionEmbeddings(maxLen: config.maxTextTokens + 2, dims: config.hiddenSize)
        self.speechPosEmb = LearnedPositionEmbeddings(maxLen: config.maxSpeechTokens + 4, dims: config.hiddenSize)

        // Load embedding weights using update(parameters:) - MLX Embedding.weight is immutable
        if let w = weights["textEmb.weight"] {
            self.textEmb.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  Loaded textEmb.weight")
        }
        if let w = weights["speechEmb.weight"] {
            self.speechEmb.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  Loaded speechEmb.weight")
        }
        if let w = weights["textPosEmb.embedding.weight"] {
            self.textPosEmb.embedding.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  Loaded textPosEmb.embedding.weight")
        }
        if let w = weights["speechPosEmb.embedding.weight"] {
            self.speechPosEmb.embedding.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  Loaded speechPosEmb.embedding.weight")
        }

        // Output head - Use FP32 weights from t3_fp32.safetensors for perfect precision
        self.speechHead = LinearFactory.load("speechHead", inputDim: config.hiddenSize, outputDim: config.speechVocabSize, weights: weights, bias: false)

        // Speaker projection - may be quantized
        self.speakerProj = LinearFactory.load("speakerProj", inputDim: 256, outputDim: config.hiddenSize, weights: weights, bias: true)

        // Perceiver module (compresses 150 speech tokens to 32)
        if config.usePerceiverResampler {
            self.perceiver = Perceiver(
                queryTokens: 32,
                channels: config.hiddenSize,
                numHeads: 4,
                weights: weights,
                prefix: "perceiver"
            )
            print("  Loaded Perceiver module")
        } else {
            self.perceiver = nil
        }

        // Emotion adversarial conditioning
        if let emotionWeight = weights["emotionAdvFC.weight"] {
            self.emotionAdvFC = Linear(1, config.hiddenSize, bias: false)
        } else {
            self.emotionAdvFC = nil
        }

        super.init()

        // Load emotionAdvFC weights after super.init()
        if let emotionWeight = weights["emotionAdvFC.weight"] {
            self.emotionAdvFC!.update(parameters: ModuleParameters.unflattened(["weight": emotionWeight]))
            print("  Loaded emotionAdvFC.weight")
        }

        self.train(false) // Set eval mode
        print("T3Model: Initialization complete")
    }

    /// Legacy init without weights (random initialization)
    public init(config: T3Config) {
        self.config = config

        // Build transformer (all regular TransformerBlocks for legacy init)
        self.layers = (0..<config.numHiddenLayers).map { _ in TransformerBlock(config: config) as Module }
        self.norm = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        // Embeddings
        self.textEmb = Embedding(embeddingCount: config.textVocabSize, dimensions: config.hiddenSize)
        self.speechEmb = Embedding(embeddingCount: config.speechVocabSize, dimensions: config.hiddenSize)
        self.textPosEmb = LearnedPositionEmbeddings(maxLen: config.maxTextTokens + 2, dims: config.hiddenSize)
        self.speechPosEmb = LearnedPositionEmbeddings(maxLen: config.maxSpeechTokens + 4, dims: config.hiddenSize)

        // Output head
        self.speechHead = Linear(config.hiddenSize, config.speechVocabSize, bias: false)

        // Speaker projection: 256-dim speaker embedding -> 1024-dim hidden
        self.speakerProj = Linear(256, config.hiddenSize)

        // No perceiver or emotion in legacy init
        self.perceiver = nil
        self.emotionAdvFC = nil

        super.init()
    }

    /// Forward pass through transformer
    public func forward(
        _ embeddings: MLXArray,
        cache: [KVCache]? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        var h = embeddings

        // Check if this is the initial forward pass (cache offset = 0 or nil)
        let cacheOffset = cache?[0].offset ?? -1
        let isInitialPass = (cache == nil) || cacheOffset == 0

        if cacheOffset <= 0 {
            print("DEBUG forward(): cacheOffset=\(cacheOffset), isInitialPass=\(isInitialPass)")
        }

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            // Cast to appropriate type and call forward
            if let inspectable = layer as? InspectableTransformerBlock {
                h = inspectable(h, mask: mask, cache: layerCache)
            } else if let regular = layer as? TransformerBlock {
                h = regular(h, mask: mask, cache: layerCache)
            }

            // DEBUG: Capture Layer 0 output for BOTH initial and incremental passes
            if i == 0 {
                let currentOffset = cacheOffset

                // 🔍 NEW: Debug incremental pass (Step 1)
                if currentOffset == 80 {
                    eval(h)
                    print("\n" + String(repeating: "=", count: 60))
                    print("🔍 SWIFT LAYER 0 INCREMENTAL (STEP 1) DEBUG:")
                    print(String(repeating: "=", count: 60))
                    print("  Layer 0 incremental output shape: \(h.shape)")
                    print("  Cache offset: \(currentOffset) (should be 80)")

                    let b0First5 = h[0, 0, 0..<5]
                    let b1First5 = h[1, 0, 0..<5]
                    eval(b0First5, b1First5)

                    print("  Batch 0, first 5:")
                    print("    \(b0First5.asArray(Float.self))")
                    print("  Batch 1, first 5:")
                    print("    \(b1First5.asArray(Float.self))")
                    print(String(repeating: "=", count: 60))
                    print("Python Reference (from test_layer0_full.py):")
                    print("  Batch 0: [-0.04825843, 0.09446631, -0.10146948, 0.04380099, -0.04365392]")
                    print("  Batch 1: [-0.04825843, 0.09446631, -0.10146948, 0.04380099, -0.04365392]")
                    print("✅ If Swift matches → Layer 0 is PERFECT (Attention + MLP + Residuals)!")
                    print("❌ If Swift diverges → Bug still exists (check MLP or residual connections)")
                    print(String(repeating: "=", count: 60) + "\n")
                }
            }

            if i == 0 && isInitialPass {
                eval(h)  // Force computation
                print("\n" + String(repeating: "=", count: 60))
                print("🔍 SWIFT LAYER 0 DEBUG:")
                print(String(repeating: "=", count: 60))
                print("  Layer 0 output shape: \(h.shape)")

                // Get last token (position 79 = BOS token for 2-batch, or last position)
                let lastIdx = h.shape[1] - 1
                let batchSize = h.shape[0]
                let b0Last10 = h[0, lastIdx, 0..<10]
                eval(b0Last10)

                print("  Batch 0, last token, first 10:")
                print("    \(b0Last10.asArray(Float.self))")

                if batchSize >= 2 {
                    let b1Last10 = h[1, lastIdx, 0..<10]
                    eval(b1Last10)
                    print("  Batch 1, last token, first 10:")
                    print("    \(b1Last10.asArray(Float.self))")
                }

                // Stats
                let mean = h.mean()
                let variance = ((h - mean) * (h - mean)).mean()
                let std = sqrt(variance)
                eval(mean, variance, std)

                let stats = (
                    h.min().item(Float.self),
                    h.max().item(Float.self),
                    mean.item(Float.self),
                    std.item(Float.self)
                )
                print("  Range: [\(String(format: "%.6f", stats.0)), \(String(format: "%.6f", stats.1))]")
                print("  Mean: \(String(format: "%.6f", stats.2)), Std: \(String(format: "%.6f", stats.3))")
                print(String(repeating: "=", count: 60) + "\n")
            }

            // DEBUG: Capture Layer 15 output (index 14) for binary search
            if i == 14 && isInitialPass {
                eval(h)  // Force computation
                print("\n" + String(repeating: "=", count: 60))
                print("🔍 SWIFT LAYER 15 DEBUG (Binary Search):")
                print(String(repeating: "=", count: 60))
                print("  Layer 15 output shape: \(h.shape)")

                // Get last token for batches
                let lastIdx = h.shape[1] - 1
                let batchSize = h.shape[0]
                let b0Last10 = h[0, lastIdx, 0..<10]
                eval(b0Last10)

                print("  Batch 0, last token, first 10:")
                print("    \(b0Last10.asArray(Float.self))")

                if batchSize >= 2 {
                    let b1Last10 = h[1, lastIdx, 0..<10]
                    eval(b1Last10)
                    print("  Batch 1, last token, first 10:")
                    print("    \(b1Last10.asArray(Float.self))")
                }

                // Stats
                let mean = h.mean()
                let variance = ((h - mean) * (h - mean)).mean()
                let std = sqrt(variance)
                eval(mean, variance, std)

                let stats = (
                    h.min().item(Float.self),
                    h.max().item(Float.self),
                    mean.item(Float.self),
                    std.item(Float.self)
                )
                print("  Range: [\(String(format: "%.6f", stats.0)), \(String(format: "%.6f", stats.1))]")
                print("  Mean: \(String(format: "%.6f", stats.2)), Std: \(String(format: "%.6f", stats.3))")
                print(String(repeating: "=", count: 60) + "\n")
            }
        }

        // 🔬 CHECKPOINT: Final Block Output (before norm)
        if isInitialPass {
            eval(h)
            print("\n" + String(repeating: "=", count: 60))
            print("🔬 CHECKPOINT: Final Block Output (Before Norm)")
            print(String(repeating: "=", count: 60))
            print("  Shape: \(h.shape)")

            let lastIdx = h.shape[1] - 1
            let batchSize = h.shape[0]
            let b0_last_5 = h[0, lastIdx, 0..<5]
            eval(b0_last_5)

            print("  [B0, T\(lastIdx), :5]: \(b0_last_5.asArray(Float.self))")
            if batchSize >= 2 {
                let b1_last_5 = h[1, lastIdx, 0..<5]
                eval(b1_last_5)
                print("  [B1, T\(lastIdx), :5]: \(b1_last_5.asArray(Float.self))")
            }

            let mean = h.mean()
            let variance = ((h - mean) * (h - mean)).mean()
            let std = sqrt(variance)
            eval(mean, variance, std)

            let stats = (
                h.min().item(Float.self),
                h.max().item(Float.self),
                mean.item(Float.self),
                std.item(Float.self)
            )
            print("  Range: [\(String(format: "%.6f", stats.0)), \(String(format: "%.6f", stats.1))]")
            print("  Mean: \(String(format: "%.6f", stats.2)), Std: \(String(format: "%.6f", stats.3))")
            print(String(repeating: "=", count: 60) + "\n")
        }

        // 🔬 FINAL NORM DIAGNOSTICS (Before applying norm)
        if isInitialPass {
            print("\n" + String(repeating: "=", count: 60))
            print("🔬 FINAL NORM DIAGNOSTICS")
            print(String(repeating: "=", count: 60))

            // 1. Check Weights
            eval(norm.weight)
            let normWeightFirst5 = norm.weight[0..<5]
            eval(normWeightFirst5)
            print("  Final Norm Weight (First 5): \(normWeightFirst5.asArray(Float.self))")

            // 2. Check Weight Stats
            let wMean = norm.weight.mean()
            let wSum = norm.weight.sum()
            eval(wMean, wSum)
            print("  Final Norm Weight Mean: \(wMean.item(Float.self))")
            print("  Final Norm Weight Sum: \(wSum.item(Float.self))")

            // 3. Check Epsilon
            print("  Final Norm Epsilon: \(norm.eps)")

            // 4. Manual RMS Calculation
            let x = h  // The input (Checkpoint 1)
            let x2 = x * x
            let mse = x2.mean(axis: -1, keepDims: true)
            let variance = mse + MLXArray(norm.eps)
            let rms = rsqrt(variance)
            let manualNorm = x * rms * norm.weight

            eval(manualNorm)
            let lastIdx = manualNorm.shape[1] - 1
            let manualNormSample = manualNorm[0, lastIdx, 0..<5]
            eval(manualNormSample)
            print("\n  Manual RMS Calculation:")
            print("    Input [B0, T79, :5]: \(h[0, lastIdx, 0..<5].asArray(Float.self))")
            print("    Manual Norm [B0, T79, :5]: \(manualNormSample.asArray(Float.self))")

            print(String(repeating: "=", count: 60) + "\n")
        }

        let normedOutput = norm(h)

        // 🔬 CHECKPOINT: Final Norm Output
        if isInitialPass {
            eval(normedOutput)
            print("\n" + String(repeating: "=", count: 60))
            print("🔬 CHECKPOINT: Final Norm Output")
            print(String(repeating: "=", count: 60))
            print("  Shape: \(normedOutput.shape)")

            let lastIdx = normedOutput.shape[1] - 1
            let batchSize = normedOutput.shape[0]
            let b0_last_5 = normedOutput[0, lastIdx, 0..<5]
            eval(b0_last_5)

            print("  Actual Norm Output [B0, T\(lastIdx), :5]: \(b0_last_5.asArray(Float.self))")
            if batchSize >= 2 {
                let b1_last_5 = normedOutput[1, lastIdx, 0..<5]
                eval(b1_last_5)
                print("  [B1, T\(lastIdx), :5]: \(b1_last_5.asArray(Float.self))")
            }
            print("\n  Python Reference [B0, T79, :5]: [-0.02582952, 0.70925778, 0.39189163, -1.67706203, -2.14925456]")

            let mean = normedOutput.mean()
            let variance = ((normedOutput - mean) * (normedOutput - mean)).mean()
            let std = sqrt(variance)
            eval(mean, variance, std)

            let stats = (
                normedOutput.min().item(Float.self),
                normedOutput.max().item(Float.self),
                mean.item(Float.self),
                std.item(Float.self)
            )
            print("  Range: [\(String(format: "%.6f", stats.0)), \(String(format: "%.6f", stats.1))]")
            print("  Mean: \(String(format: "%.6f", stats.2)), Std: \(String(format: "%.6f", stats.3))")
            print(String(repeating: "=", count: 60) + "\n")
        }

        return normedOutput
    }

    /// Get captured attention weights from inspectable layers
    /// Returns: Dictionary mapping layer index to attention weights [B, nHeads, T_q, T_kv]
    public func getCapturedAttentionWeights() -> [Int: MLXArray] {
        var result: [Int: MLXArray] = [:]
        var inspectableCount = 0
        var capturedCount = 0
        for (i, layer) in layers.enumerated() {
            if let inspectable = layer as? InspectableTransformerBlock {
                inspectableCount += 1
                if let weights = inspectable.lastAttentionWeights {
                    result[i] = weights
                    capturedCount += 1
                } else {
                    print("WARNING: Layer \(i) is inspectable but has nil lastAttentionWeights")
                }
            }
        }
        if inspectableCount > 0 && capturedCount < inspectableCount {
            print("DEBUG: Found \(inspectableCount) inspectable layers but only \(capturedCount) captured attention")
        }
        return result
    }

    /// Clear captured attention weights from all inspectable layers
    public func clearCapturedAttention() {
        for layer in layers {
            if let inspectable = layer as? InspectableTransformerBlock {
                inspectable.clearAttention()
            }
        }
    }

    /// Generate speech tokens from text with Classifier-Free Guidance (CFG)
    /// Python conditioning structure: [speaker(1) | perceiver(32) | emotion(1)] = 34 tokens
    /// CFG: logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
    public func generate(
        textTokens: MLXArray,
        speakerEmb: MLXArray,
        condTokens: MLXArray,
        maxTokens: Int = 1000,           // Python: max_new_tokens=1000
        temperature: Float = 0.0001,     // Near-zero for deterministic (PyTorch crashes at 0)
        emotionValue: Float = 0.5,       // Default emotion value
        cfgWeight: Float = 0.5,          // Python: cfg_weight=0.5
        repetitionPenalty: Float = 2.0,  // Python: repetition_penalty=2.0
        topP: Float = 1.0,               // Python: top_p=1.0
        minP: Float = 0.05               // Python: min_p=0.05
    ) -> [Int] {
        print("🚨🚨🚨 LAYER 0 DEBUG BUILD \(Date()) 🚨🚨🚨")
        print("DEBUG T3: generate() called with CFG weight: \(cfgWeight)")
        print("DEBUG T3: textTokens shape: \(textTokens.shape)")
        print("DEBUG T3: speakerEmb shape: \(speakerEmb.shape)")
        print("DEBUG T3: condTokens shape: \(condTokens.shape)")

        // ============================================
        // BUILD CONDITIONING: [speaker | perceiver | emotion]
        // Python: cond_embeds = torch.cat((cond_spkr, cond_clap, cond_prompt_speech_emb, cond_emotion_adv), dim=1)
        // ============================================

        // 1. Speaker embedding projection: [1, 256] -> [1, 1, 1024] (single token prefix!)
        let spkProj = speakerProj(speakerEmb)  // [1, 1024]
        let spkToken = spkProj.expandedDimensions(axis: 1)  // [1, 1, 1024]
        print("DEBUG T3: spkToken shape: \(spkToken.shape)")

        // 2. Embed conditioning speech tokens
        let condLen = condTokens.shape[1]
        let condPositions = MLXArray(Array(0..<condLen).map { Int32($0) }).reshaped([1, condLen])
        let condSpeechEmb = speechEmb(condTokens) + speechPosEmb(condPositions)  // [1, 150, 1024]

        // 3. Apply Perceiver to compress: [1, 150, 1024] -> [1, 32, 1024]
        let perceiverOutput: MLXArray
        if let perc = perceiver {
            perceiverOutput = perc(condSpeechEmb)
            print("DEBUG T3: Perceiver compressed \(condLen) -> \(perceiverOutput.shape[1]) tokens")
        } else {
            // Fallback if no perceiver: use raw conditioning (not recommended)
            perceiverOutput = condSpeechEmb
            print("DEBUG T3: WARNING - No Perceiver, using raw \(condLen) tokens")
        }

        // 4. Emotion conditioning: [1] -> [1, 1, 1024]
        let emotionToken: MLXArray
        if let emotionFC = emotionAdvFC {
            let emotionInput = MLXArray([emotionValue]).reshaped([1, 1, 1])
            emotionToken = emotionFC(emotionInput)  // [1, 1, 1024]
            print("DEBUG T3: emotionToken shape: \(emotionToken.shape)")
        } else {
            // No emotion conditioning
            emotionToken = MLXArray.zeros([1, 0, config.hiddenSize])
        }

        // 5. Concatenate: [speaker(1) | perceiver(32) | emotion(1)]
        let condEmb = concatenated([spkToken, perceiverOutput, emotionToken], axis: 1)
        let finalCondLen = condEmb.shape[1]
        print("DEBUG T3: Final conditioning length: \(finalCondLen) (expected ~34)")

        // 🔬 CONDITIONING DEBUG: Check what we're feeding to transformer
        eval(condEmb)
        let cond_pos0 = condEmb[0, 0, 0..<5]
        let cond_pos1 = condEmb[0, 1, 0..<5]
        eval(cond_pos0, cond_pos1)
        print("🔬 CONDITIONING EMBEDDINGS:")
        print("   Position 0 (speaker): \(cond_pos0.asArray(Float.self))")
        print("   Position 1 (perceiver[0]): \(cond_pos1.asArray(Float.self))")
        print("   Python Pos 0 reference: [-0.006982, -0.016019, -0.012832, 0.014774, -0.018492]")

        // Text positions start from 0 (text has its own position embedding space)
        let textLen = textTokens.shape[1]
        let textPositions = MLXArray(Array(0..<textLen).map { Int32($0) }).reshaped([1, textLen])
        let textEmbedding = textEmb(textTokens) + textPosEmb(textPositions)

        // --- DEBUG: TEXT EMBEDDINGS ---
        eval(textEmbedding)
        print("\n🔍 TEXT EMBEDDINGS (after textEmb + textPosEmb):")
        print("   Shape: \(textEmbedding.shape)")
        print("   Text tokens: \(textTokens.asArray(Int32.self))")
        let firstTextEmb = textEmbedding[0, 0, 0..<10]
        eval(firstTextEmb)
        print("   Token 0 (284) embedding [:10]: \(firstTextEmb.asArray(Float.self))")
        print()
        // --------------------------------

        // ============================================
        // CFG SETUP: Create conditioned and unconditioned batches
        // Python: Doubles batch [cond | uncond], uncond has TOKEN embeddings zeroed but KEEPS positional embeddings
        // CRITICAL: PyTorch does:
        //   1. text_emb = token_emb(tokens)
        //   2. text_emb[1].zero_()  # Zero ONLY token embeddings for batch 1
        //   3. text_emb = text_emb + pos_emb(tokens)  # Add positional embeddings to BOTH batches
        // So unconditional has: zeros (token) + positional = positional embeddings only
        // ============================================
        let useCFG = cfgWeight > 0

        // For unconditioned: zero token embeddings BUT keep positional embeddings (matching PyTorch!)
        let (B, T_text, C) = (textEmbedding.shape[0], textEmbedding.shape[1], textEmbedding.shape[2])
        let uncondTokenEmb = MLXArray.zeros([B, T_text, C], dtype: textEmbedding.dtype)
        let uncondTextEmbedding = uncondTokenEmb + textPosEmb(textPositions)  // Add positional embeddings!
        print("DEBUG T3: Created uncondTextEmbedding: zeros (token) + positional = \(uncondTextEmbedding.shape), dtype \(uncondTextEmbedding.dtype)")

        // BOS token uses speech position 0 (Python: get_fixed_embedding(0))
        var currentToken = startSpeechToken
        let bosTokenArray = MLXArray([Int32(currentToken)]).reshaped([1, 1])
        let bosEmb = speechEmb(bosTokenArray) + speechPosEmb.getFixedEmbedding(0)  // Position 0

        // Build initial sequences for both conditioned and unconditioned
        // Conditioned: [cond | text | bos]
        // Unconditioned: [cond | zeros | bos]
        let condInputEmb = concatenated([condEmb, textEmbedding, bosEmb], axis: 1)
        let uncondInputEmb = concatenated([condEmb, uncondTextEmbedding, bosEmb], axis: 1)

        // CRITICAL FIX: PyTorch adds BOS token TWICE!
        // 1. First BOS added above in prepare_input_embeds (initial_speech_tokens)
        // 2. Second BOS added in inference() line 319: torch.cat([embeds, bos_embed], dim=1)
        // This creates a 46-token sequence: [cond(34) + text(10) + bos + bos]
        let condWithDoubleBos = concatenated([condInputEmb, bosEmb], axis: 1)
        let uncondWithDoubleBos = concatenated([uncondInputEmb, bosEmb], axis: 1)

        // Stack into batch [cond, uncond] = [2, seqLen, hidden]
        let inputEmb: MLXArray
        if useCFG {
            inputEmb = concatenated([condWithDoubleBos, uncondWithDoubleBos], axis: 0)
            print("DEBUG T3: Using CFG - batch size 2 (with double BOS fix)")
        } else {
            inputEmb = condWithDoubleBos
            print("DEBUG T3: No CFG - batch size 1 (with double BOS fix)")
        }

        // Initialize cache (one set per layer, handles batch internally)
        let cache = layers.map { _ in KVCache() }

        // ============================================
        // ALIGNMENT STREAM ANALYZER SETUP
        // Text tokens are positioned at: [cond(34) | text(textLen) | bos1]
        // So text starts at finalCondLen and ends at finalCondLen + textLen
        // Note: The second BOS (at position 45) is not part of the text slice
        // ============================================
        let textStart = finalCondLen
        let textEnd = finalCondLen + textLen
        let analyzer = AlignmentStreamAnalyzer(
            textTokensSlice: (textStart, textEnd),
            eosIdx: stopSpeechToken
        )
        print("DEBUG T3: AlignmentStreamAnalyzer initialized - text slice (\(textStart), \(textEnd))")

        // ============================================
        // CREATE CAUSAL MASK (fully causal like Python's LlamaModel)
        // ============================================
        // CRITICAL FIX: Python's LlamaModel uses fully causal attention for all tokens.
        // The previous "hybrid mask" (bidirectional for conditioning) was incorrect and caused
        // logit divergence from Python. Now using standard causal mask where no token can
        // attend to future positions.
        // Sequence structure: [cond(34) | text(10) | bos1 | bos2] = 46 tokens
        let seqLen = inputEmb.shape[1]
        let causalMask = T3Model.createCausalMask(seqLen: seqLen)
            .reshaped([1, 1, seqLen, seqLen])
        eval(causalMask)
        print("✅ Created causal mask: \(causalMask.shape), seqLen=\(seqLen) (fully causal like Python's LlamaModel)\n")

        // ============================================
        // DEBUG: CHECK INPUT TO LAYER 0 FOR BOTH BATCHES
        // Focus on TEXT region (positions 34-43) as this is where conditional/unconditional differ
        // ============================================
        print("\n🔍 === INPUT TO LAYER 0 (Before any processing) ===")
        eval(inputEmb)
        print("Input shape: \(inputEmb.shape)")
        print("Sequence structure: [cond(34) | text(10) | bos1 | bos2] = 46 tokens")
        print("Text region: positions 34-43 (should be ZEROS for Batch 1 unconditional)\n")

        // Check TEXT region (positions 34-43)
        let textStartDebug = 34
        let textEndDebug = 43
        print("Batch 0 (Conditional), TEXT positions \(textStartDebug)-\(textEndDebug), first 5 dims:")
        for pos in textStartDebug...textEndDebug {
            let textEmb = inputEmb[0, pos, 0..<5]
            eval(textEmb)
            print("  Pos \(pos): \(textEmb.asArray(Float.self))")
        }

        print("\nBatch 1 (Unconditional), TEXT positions \(textStartDebug)-\(textEndDebug), first 5 dims:")
        print("(These should all be ZEROS for unconditional)")
        for pos in textStartDebug...textEndDebug {
            let textEmb = inputEmb[1, pos, 0..<5]
            eval(textEmb)
            let vals = textEmb.asArray(Float.self)
            let allZeros = vals.allSatisfy { abs($0) < 1e-6 }
            print("  Pos \(pos): \(vals) \(allZeros ? "✓ ZEROS" : "✗ NOT ZEROS!")")
        }

        // Also check last token (BOS) - should be identical
        let lastInputIdxCheck = inputEmb.shape[1] - 1
        print("\nLast Token (BOS at position \(lastInputIdxCheck)):")
        let b0BOS = inputEmb[0, lastInputIdxCheck, 0..<10]
        let b1BOS = inputEmb[1, lastInputIdxCheck, 0..<10]
        eval(b0BOS, b1BOS)
        print("  Batch 0 [:10]: \(b0BOS.asArray(Float.self))")
        print("  Batch 1 [:10]: \(b1BOS.asArray(Float.self))")
        let diffCheck = abs(b0BOS - b1BOS).max()
        eval(diffCheck)
        print("  Max difference: \(diffCheck.item(Float.self)) (should be 0.0)")
        print(String(repeating: "=", count: 60) + "\n")

        // ============================================
        // DEBUG INSTRUMENTATION - REMOVED TO AVOID CACHE CONTAMINATION
        // The first Layer 0 forward pass was contaminating cache[0], causing
        // the full forward pass at line 1823 to see 46+46=92 tokens in the cache.
        // More detailed Layer 0 debugging happens later using fresh caches.
        // ============================================

        // ============================================
        // LAYER 1 DETAILED INSTRUMENTATION
        // ============================================
        print("🔍 RUNNING DETAILED LAYER 1 INSTRUMENTATION...")
        print(String(repeating: "=", count: 60))

        // Fresh caches for clean debug
        let freshCache = layers.map { _ in KVCache() }

        // CRITICAL FIX: Create CAUSAL mask (matching Python's LlamaModel!)
        // Python uses fully causal attention - no special bidirectional for conditioning.
        let seqLenDebug = inputEmb.shape[1]
        let causalMaskDebug = T3Model.createCausalMask(seqLen: seqLenDebug)
            .reshaped([1, 1, seqLenDebug, seqLenDebug])  // Add batch and head dimensions
        eval(causalMaskDebug)
        print("✅ Created causal mask: \(causalMaskDebug.shape), seqLen=\(seqLenDebug) (fully causal like Python)\n")

        // ======== LAYER 0 ========
        print("LAYER 0:")
        print(String(repeating: "=", count: 60))

        var layer0Input = inputEmb
        print("  Input shape: \(layer0Input.shape)")
        let layer0InB0 = layer0Input[0, layer0Input.shape[1]-1, 0..<5]
        let layer0InB1 = layer0Input[1, layer0Input.shape[1]-1, 0..<5]
        eval(layer0InB0, layer0InB1)
        print("  Input Batch 0, last token, first 5: \(layer0InB0.asArray(Float.self))")
        print("  Input Batch 1, last token, first 5: \(layer0InB1.asArray(Float.self))")

        var layer0Output = inputEmb
        if let block0 = layers[0] as? TransformerBlock {
            layer0Output = block0.callAsFunction(layer0Output, mask: causalMask, cache: freshCache[0])
        } else if let block0 = layers[0] as? InspectableTransformerBlock {
            layer0Output = block0.callAsFunction(layer0Output, mask: causalMask, cache: freshCache[0])
        }
        eval(layer0Output)

        let layer0OutB0 = layer0Output[0, layer0Output.shape[1]-1, 0..<5]
        let layer0OutB1 = layer0Output[1, layer0Output.shape[1]-1, 0..<5]
        eval(layer0OutB0, layer0OutB1)
        print("  Output Batch 0, last token, first 5: \(layer0OutB0.asArray(Float.self))")
        print("  Output Batch 1, last token, first 5: \(layer0OutB1.asArray(Float.self))")
        print()

        // ======== LAYER 1 DETAILED ========
        print(String(repeating: "=", count: 60))
        print("LAYER 1 DETAILED:")
        print(String(repeating: "=", count: 60))

        if let layer1Block = layers[1] as? TransformerBlock {
            let layer1Input = layer0Output
            let lastToken = layer1Input.shape[1] - 1

            print("1️⃣ Layer 1 Input:")
            print("   Shape: \(layer1Input.shape)")
            let step1B0 = layer1Input[0, lastToken, 0..<5]
            let step1B1 = layer1Input[1, lastToken, 0..<5]
            eval(step1B0, step1B1)
            print("   Batch 0, last token, first 5: \(step1B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step1B1.asArray(Float.self))")
            print()

            // Step 1: Input layernorm
            let normedInput = layer1Block.inputLayernorm(layer1Input)
            eval(normedInput)
            print("2️⃣ After Input LayerNorm:")
            let step2B0 = normedInput[0, lastToken, 0..<5]
            let step2B1 = normedInput[1, lastToken, 0..<5]
            eval(step2B0, step2B1)
            print("   Batch 0, last token, first 5: \(step2B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step2B1.asArray(Float.self))")
            print()

            // Step 2: Self-attention (with causal mask)
            let attnOutput = layer1Block.selfAttn(normedInput, mask: causalMask, cache: freshCache[1])
            eval(attnOutput)
            print("3️⃣ After Self-Attention (before residual):")
            let step3B0 = attnOutput[0, lastToken, 0..<5]
            let step3B1 = attnOutput[1, lastToken, 0..<5]
            eval(step3B0, step3B1)
            print("   Batch 0, last token, first 5: \(step3B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step3B1.asArray(Float.self))")
            print()

            // Step 3: First residual
            let h = layer1Input + attnOutput
            eval(h)
            print("4️⃣ After First Residual (h = x + attn):")
            let step4B0 = h[0, lastToken, 0..<5]
            let step4B1 = h[1, lastToken, 0..<5]
            eval(step4B0, step4B1)
            print("   Batch 0, last token, first 5: \(step4B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step4B1.asArray(Float.self))")
            print()

            // Step 4: Post-attention layernorm
            let normedH = layer1Block.postAttentionLayernorm(h)
            eval(normedH)
            print("5️⃣ After Post-Attention LayerNorm:")
            let step5B0 = normedH[0, lastToken, 0..<5]
            let step5B1 = normedH[1, lastToken, 0..<5]
            eval(step5B0, step5B1)
            print("   Batch 0, last token, first 5: \(step5B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step5B1.asArray(Float.self))")
            print()

            // Step 5: MLP
            let mlpOutput = layer1Block.mlp(normedH)
            eval(mlpOutput)
            print("6️⃣ After MLP (before residual):")
            let step6B0 = mlpOutput[0, lastToken, 0..<5]
            let step6B1 = mlpOutput[1, lastToken, 0..<5]
            eval(step6B0, step6B1)
            print("   Batch 0, last token, first 5: \(step6B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step6B1.asArray(Float.self))")
            print()

            // Step 6: Second residual
            let layer1Output = h + mlpOutput
            eval(layer1Output)
            print("7️⃣ Final Layer 1 Output (h + mlp):")
            let step7B0 = layer1Output[0, lastToken, 0..<5]
            let step7B1 = layer1Output[1, lastToken, 0..<5]
            eval(step7B0, step7B1)
            print("   Batch 0, last token, first 5: \(step7B0.asArray(Float.self))")
            print("   Batch 1, last token, first 5: \(step7B1.asArray(Float.self))")
            print()

            print(String(repeating: "=", count: 60))
            print("COMPARE WITH PYTHON OUTPUT")
            print(String(repeating: "=", count: 60))
            print("Find where the first divergence appears:")
            print("  - If after input layernorm → RMSNorm bug")
            print("  - If after attention → RoPE or attention bug")
            print("  - If after MLP → MLP bug")
            print(String(repeating: "=", count: 60) + "\n")
        } else {
            print("❌ Layer 1 is not a TransformerBlock - skipping detailed instrumentation\n")
        }

        // CRITICAL FIX: Create causal mask for initial forward pass
        // This prevents future tokens from attending to past tokens
        let initialSeqLen = inputEmb.shape[1]
        let initialMask = T3Model.createCausalMask(seqLen: initialSeqLen)
            .reshaped([1, 1, initialSeqLen, initialSeqLen])
        eval(initialMask)

        // Run forward pass with causal mask
        var hidden = forward(inputEmb, cache: cache, mask: causalMask)

        var generatedTokens: [Int] = [currentToken]

        for step in 0..<maxTokens {
            // CRITICAL: Wrap in autoreleasepool to prevent memory graph buildup
            let nextToken: Int = autoreleasepool {
                // Get logits for last position - get last token's hidden state
                let lastHidden = hidden[0..., (hidden.shape[1]-1)..<hidden.shape[1], 0...]
                let allLogits = speechHead(lastHidden)  // [B, 1, vocab] where B=2 with CFG

                // ============================================
                // CFG: Combine conditioned and unconditioned logits
                // Python: logits = uncond_logits + cfg_weight * (cond_logits - uncond_logits)
                // ============================================
                let logits: MLXArray
                if useCFG {
                    // allLogits is [2, 1, vocab]
                    let condLogits = allLogits[0, 0..., 0...]    // [1, vocab]
                    let uncondLogits = allLogits[1, 0..., 0...]  // [1, vocab]

                    // CFG formula: cond + cfg * (cond - uncond)
                    logits = condLogits + MLXArray(cfgWeight) * (condLogits - uncondLogits)
                } else {
                    logits = allLogits[0, 0..., 0...]  // [1, vocab]
                }

                // Sample with temperature, repetition penalty, and top-p + invalid token filtering
                // Also filter out tokens >= 6561 (invalid for S3Gen)
                // topP and minP are now passed as function parameters

                var logitsFlat = logits.reshaped([-1])  // [vocabSize] - apply temp after filtering

                // ============================================
                // ALIGNMENT STREAM ANALYZER INTERVENTION
                // The analyzer monitors attention patterns and modifies logits to:
                // - Suppress EOS when text isn't complete (prevents early termination)
                // - Force EOS on repetition/hallucination detection (prevents loops)
                // ============================================
                let capturedAttention = getCapturedAttentionWeights()
                let analyzerEnabled = true
                if analyzerEnabled && !capturedAttention.isEmpty {
                    let lastGeneratedToken = generatedTokens.last
                    logitsFlat = analyzer.step(
                        attentionWeights: capturedAttention,
                        logits: logitsFlat,
                        nextToken: lastGeneratedToken
                    )
                    // Clear attention to free memory
                    clearCapturedAttention()
                } else {
                    // Still clear attention to prevent memory buildup
                    clearCapturedAttention()
                }

                // Apply repetition penalty to previously generated tokens
                if repetitionPenalty != 1.0 && !generatedTokens.isEmpty {
                    // Get unique tokens - filter to vocab size
                    let uniqueTokens = Array(Set(generatedTokens).filter { $0 < logitsFlat.shape[0] })
                    if !uniqueTokens.isEmpty {
                        // Create index array for all tokens at once (GPU)
                        let tokenIndices = MLXArray(uniqueTokens.map { Int32($0) })

                        // Get all logits for generated tokens at once (GPU)
                        let selectedLogits = logitsFlat[tokenIndices]

                        // Apply penalty using MLX operations (GPU)
                        let penalty = MLXArray(repetitionPenalty)
                        let penalizedLogits = `where`(
                            selectedLogits .> 0,
                            selectedLogits / penalty,
                            selectedLogits * penalty
                        )

                        // Convert penalized logits to CPU once
                        let penalizedArray = penalizedLogits.asArray(Float.self)

                        // Build update arrays on CPU (single pass, no GPU sync per iteration)
                        var updateMaskArray = [Float](repeating: 0.0, count: logitsFlat.shape[0])
                        var penalizedFullArray = [Float](repeating: 0.0, count: logitsFlat.shape[0])

                        for (i, token) in uniqueTokens.enumerated() {
                            updateMaskArray[token] = 1.0
                            penalizedFullArray[token] = penalizedArray[i]
                        }

                        // Convert back to GPU once
                        let updateMask = MLXArray(updateMaskArray)
                        let penalizedFull = MLXArray(penalizedFullArray)

                        // Apply update using where (GPU)
                        logitsFlat = `where`(updateMask .== 1.0, penalizedFull, logitsFlat)
                    }
                }

                // IMPORTANT: Python does NOT mask invalid tokens during generation
                // Invalid tokens are filtered AFTER generation via drop_invalid_tokens()
                // This allows the model to naturally generate SOS/EOS tokens

                // Handle temperature=0.0 (greedy/argmax) specially to avoid division by zero
                // Use argmax for very low temperatures to ensure deterministic output
                let nextToken: Int
                if temperature <= 0.01 {  // Temperature <= 0.01 uses greedy decoding for determinism
                    let tokenIdx = Int(argMax(logitsFlat, axis: -1).item(Int32.self))
                    nextToken = tokenIdx
                } else {
                    // Apply temperature scaling
                    let scaledLogits = logitsFlat / MLXArray(temperature)

                // ============================================
                // TOP-P (NUCLEUS) SAMPLING + MIN-P
                // Python: TopPLogitsWarper(top_p=0.95), MinPLogitsWarper(min_p=0.05)
                // ============================================

                // Get probabilities for filtering
                let probs = softmax(scaledLogits, axis: -1)

                // Min-P filtering: mask tokens with prob < min_p * max_prob
                let maxProb = probs.max()
                let minPThreshold = MLXArray(minP) * maxProb
                let minPMask = probs .< minPThreshold
                var filteredLogits = which(minPMask, MLXArray(-Float.infinity), scaledLogits)

                // Top-P (nucleus) filtering - optimized to reduce GPU-CPU transfers
                // Transfer probs to CPU once for sorting and cumsum (unavoidable for top-p)
                let probsArray = probs.asArray(Float.self)

                // CPU: Create (index, prob) pairs and sort descending
                var sortedPairs = probsArray.enumerated().map { ($0.offset, $0.element) }
                sortedPairs.sort { $0.1 > $1.1 }

                // CPU: Compute cumulative sum and build keep mask in single pass
                var cumSum: Float = 0.0
                var keepMaskArray = [Float](repeating: 0.0, count: probsArray.count)

                for (idx, prob) in sortedPairs {
                    cumSum += prob
                    keepMaskArray[idx] = 1.0
                    if cumSum > topP {
                        break
                    }
                }

                // GPU: Convert mask back once and apply
                let keepMask = MLXArray(keepMaskArray)
                filteredLogits = `where`(
                    keepMask .== 1.0,
                    filteredLogits,
                    MLXArray(-Float.infinity)
                )

                // Sample from the filtered distribution
                // MLXRandom.categorical has issues with -inf values
                // Use a workaround: replace -inf with very large negative value
                    let logitsForSampling = which(
                        filteredLogits .== MLXArray(-Float.infinity),
                        MLXArray(-1e10),  // Large negative but not -inf
                        filteredLogits
                    )
                    let sampled = MLXRandom.categorical(logitsForSampling.expandedDimensions(axis: 0), axis: -1)
                    eval(sampled)
                    nextToken = Int(sampled.item(Int32.self))
                }  // end if-else for temperature

                return nextToken
            }

            // Check EOS
            if nextToken == stopSpeechToken {
                break
            }

            generatedTokens.append(nextToken)
            currentToken = nextToken

            // Prepare next input (also in autoreleasepool)
            // Speech position embeddings are SEPARATE from text/conditioning context
            // BOS is at position 0, first generated token at position 1, etc.
            autoreleasepool {
                let nextTokenArray = MLXArray([Int32(nextToken)]).reshaped([1, 1])
                let nextPosition = step + 1  // Position 1, 2, 3, ... (BOS was at 0)
                let nextEmb = speechEmb(nextTokenArray) + speechPosEmb.getFixedEmbedding(nextPosition)

                // For CFG: both conditioned and unconditioned use same speech token
                let nextInputEmb: MLXArray
                if useCFG {
                    // Stack same embedding twice for batch [cond, uncond]
                    nextInputEmb = concatenated([nextEmb, nextEmb], axis: 0)
                } else {
                    nextInputEmb = nextEmb
                }

                // Forward with cache (mask defaults to nil for incremental generation)
                hidden = forward(nextInputEmb, cache: cache)
            }
        }

        return generatedTokens
    }

    /// Drop invalid tokens (SOS/EOS) from generated speech tokens
    /// Matches Python's drop_invalid_tokens function:
    /// - Removes SOS (6561) from start if present
    /// - Removes EOS (6562) from end if present
    /// - Returns trimmed token sequence
    public static func dropInvalidTokens(_ tokens: [Int]) -> [Int] {
        guard !tokens.isEmpty else { return tokens }

        let SOS = 6561
        let EOS = 6562

        var startIdx = 0
        var endIdx = tokens.count

        // Find SOS and skip past it
        if let sosIndex = tokens.firstIndex(of: SOS) {
            startIdx = sosIndex + 1
        }

        // Find EOS and trim there
        if let eosIndex = tokens.firstIndex(of: EOS) {
            endIdx = eosIndex
        }

        // Return trimmed array
        guard startIdx < endIdx else { return [] }
        return Array(tokens[startIdx..<endIdx])
    }

    /// Create causal mask
    public static func createCausalMask(seqLen: Int) -> MLXArray {
        // CRITICAL FIX: Use true -infinity (not -1e9) to match Python's behavior
        // Python: torch.triu(torch.ones(...) * float('-inf'), diagonal=1)
        // This ensures softmax(exp(-inf)) = exactly 0, not a tiny non-zero value
        let mask = MLXArray.full([seqLen, seqLen], values: MLXArray(-.infinity))
        return triu(mask, k: 1)
    }

    /// Create hybrid mask: ONLY conditioning tokens (speaker/perceiver/emotion) attend bidirectionally,
    /// text and speech tokens are causal
    /// Python implementation in verify_v2_step3_transformer.py:
    ///   cond_len = cond_emb.shape[1]  # Just conditioning, NOT including text!
    ///   mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
    ///   mask_rows = mx.arange(seq_len)[:, None]
    ///   mask = mx.where(mask_rows < cond_len, mx.zeros_like(mask), mask)
    public static func createHybridMask(seqLen: Int, condLen: Int) -> MLXArray {
        // Start with causal mask (upper triangle = -inf)
        var mask = MLXArray.full([seqLen, seqLen], values: MLXArray(-.infinity))
        mask = triu(mask, k: 1)

        // Make conditioning rows bidirectional (all zeros)
        // mask_rows = mx.arange(seq_len)[:, None]
        let maskRows = MLXArray(0..<seqLen).expandedDimensions(axis: 1)  // [seqLen, 1]

        // mask = mx.where(mask_rows < cond_len, mx.zeros_like(mask), mask)
        let condition = MLX.less(maskRows, MLXArray(Int32(condLen)))
        let zeros = MLXArray.zeros([seqLen, seqLen])
        mask = MLX.where(condition, zeros, mask)

        return mask
    }
}
