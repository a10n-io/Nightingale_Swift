import MLX
import MLXNN
import Foundation

// MARK: - Perceiver Attention Block

/// Attention block for Perceiver - supports both cross-attention and self-attention
public class PerceiverAttention: Module {
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public let norm: LayerNorm
    public let toQ: Linear
    public let toK: Linear
    public let toV: Linear
    public let projOut: Linear

    public init(channels: Int, numHeads: Int = 4, weights: [String: MLXArray], prefix: String) {
        self.numHeads = numHeads
        self.headDim = channels / numHeads
        self.scale = pow(Float(headDim), -0.5)

        // Layer norm
        self.norm = LayerNorm(dimensions: channels)

        // Q, K, V projections
        self.toQ = Linear(channels, channels)
        self.toK = Linear(channels, channels)
        self.toV = Linear(channels, channels)

        // Output projection
        self.projOut = Linear(channels, channels)

        super.init()

        // Helper to convert weights to FP32 for maximum precision
        func ensureFP32(_ array: MLXArray, name: String) -> MLXArray {
            if array.dtype == .float32 {
                return array
            } else {
                print("  ⚠️ Converting \(name) from \(array.dtype) to FP32")
                return array.asType(.float32)
            }
        }

        // Load weights using update(parameters:) with explicit FP32 conversion
        print("PerceiverAttention: Loading weights with FP32 precision...")

        if let w = weights["\(prefix).norm.weight"], let b = weights["\(prefix).norm.bias"] {
            let wFP32 = ensureFP32(w, name: "\(prefix).norm.weight")
            let bFP32 = ensureFP32(b, name: "\(prefix).norm.bias")
            norm.update(parameters: ModuleParameters.unflattened(["weight": wFP32, "bias": bFP32]))
        }

        if let w = weights["\(prefix).to_q.weight"] {
            let wFP32 = ensureFP32(w, name: "\(prefix).to_q.weight")
            var params: [String: MLXArray] = ["weight": wFP32]
            if let b = weights["\(prefix).to_q.bias"] {
                params["bias"] = ensureFP32(b, name: "\(prefix).to_q.bias")
            }
            toQ.update(parameters: ModuleParameters.unflattened(params))
        }

        if let w = weights["\(prefix).to_k.weight"] {
            let wFP32 = ensureFP32(w, name: "\(prefix).to_k.weight")
            var params: [String: MLXArray] = ["weight": wFP32]
            if let b = weights["\(prefix).to_k.bias"] {
                params["bias"] = ensureFP32(b, name: "\(prefix).to_k.bias")
            }
            toK.update(parameters: ModuleParameters.unflattened(params))
        }

        if let w = weights["\(prefix).to_v.weight"] {
            let wFP32 = ensureFP32(w, name: "\(prefix).to_v.weight")
            var params: [String: MLXArray] = ["weight": wFP32]
            if let b = weights["\(prefix).to_v.bias"] {
                params["bias"] = ensureFP32(b, name: "\(prefix).to_v.bias")
            }
            toV.update(parameters: ModuleParameters.unflattened(params))
        }

        if let w = weights["\(prefix).proj_out.weight"] {
            let wFP32 = ensureFP32(w, name: "\(prefix).proj_out.weight")
            var params: [String: MLXArray] = ["weight": wFP32]
            if let b = weights["\(prefix).proj_out.bias"] {
                params["bias"] = ensureFP32(b, name: "\(prefix).proj_out.bias")
            }
            projOut.update(parameters: ModuleParameters.unflattened(params))
        }

        print("PerceiverAttention: All weights loaded with FP32 precision")
    }

    /// Forward pass with cross-attention (query attends to key/value)
    /// x1: query input [B, T1, C]
    /// x2: key/value input [B, T2, C]
    public func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        let (B, T1, C) = (x1.shape[0], x1.shape[1], x1.shape[2])
        let T2 = x2.shape[1]

        // Ensure FP32 precision for all computations
        let x1FP32 = x1.dtype == .float32 ? x1 : x1.asType(.float32)
        let x2FP32 = x2.dtype == .float32 ? x2 : x2.asType(.float32)

        // Normalize inputs
        let x1Norm = norm(x1FP32)
        let x2Norm = norm(x2FP32)

        // Project to Q, K, V
        var q = toQ(x1Norm)  // [B, T1, C]
        var k = toK(x2Norm)  // [B, T2, C]
        var v = toV(x2Norm)  // [B, T2, C]

        // Reshape to [B, numHeads, T, headDim]
        q = q.reshaped([B, T1, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, T2, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, T2, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        // scores: [B, numHeads, T1, T2]
        // CRITICAL: Match Python order - matmul THEN scale (not scale THEN matmul)
        let scores = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        let attnWeights = softmax(scores, axis: -1)

        // Apply attention to values
        var out = attnWeights.matmul(v)  // [B, numHeads, T1, headDim]

        // Reshape back to [B, T1, C]
        out = out.transposed(0, 2, 1, 3).reshaped([B, T1, C])

        // Output projection and residual
        let h = projOut(out)
        return x1FP32 + h
    }
}

// MARK: - Perceiver Module

/// Perceiver resampler that compresses speech conditioning tokens
/// Uses learned queries to attend to speech embeddings and compress them
public class Perceiver: Module {
    public var preAttentionQuery: MLXArray  // [1, 32, 1024] - learned queries
    public let attn: PerceiverAttention

    public init(queryTokens: Int = 32, channels: Int = 1024, numHeads: Int = 4, weights: [String: MLXArray], prefix: String) {
        // Load pre-attention query (learned queries that will attend to speech)
        if let query = weights["\(prefix).pre_attention_query"] {
            // Ensure FP32 precision for learned queries
            if query.dtype == .float32 {
                self.preAttentionQuery = query
                print("Perceiver: Loaded pre_attention_query shape: \(query.shape), dtype: FP32")
            } else {
                print("Perceiver: Converting pre_attention_query from \(query.dtype) to FP32")
                self.preAttentionQuery = query.asType(.float32)
                print("Perceiver: Loaded pre_attention_query shape: \(self.preAttentionQuery.shape), dtype: FP32")
            }
        } else {
            // Initialize with small random values (shouldn't happen if weights loaded correctly)
            let variance = sqrt(3.0) * sqrt(2.0 / Float(queryTokens + queryTokens))
            self.preAttentionQuery = MLXRandom.uniform(
                low: MLXArray(-variance),
                high: MLXArray(variance),
                [1, queryTokens, channels]
            )
            print("Perceiver: WARNING - pre_attention_query not found, using random init")
        }

        // Attention block for cross-attention and self-attention (with FP32 weights)
        self.attn = PerceiverAttention(
            channels: channels,
            numHeads: numHeads,
            weights: weights,
            prefix: "\(prefix).attn"
        )

        super.init()
    }

    /// Forward pass
    /// h: speech embeddings [B, T, C] where T can be 150 tokens
    /// Returns: compressed embeddings [B, 32, C]
    public func callAsFunction(_ h: MLXArray) -> MLXArray {
        let B = h.shape[0]

        // Ensure FP32 precision for input
        let hFP32 = h.dtype == .float32 ? h : h.asType(.float32)

        // Expand query to batch size: [1, 32, C] -> [B, 32, C]
        let query: MLXArray
        if B > 1 {
            query = broadcast(preAttentionQuery, to: [B, preAttentionQuery.shape[1], preAttentionQuery.shape[2]])
        } else {
            query = preAttentionQuery
        }

        // Cross-attention: queries attend to speech embeddings (both FP32)
        let preAtt = attn(query, hFP32)  // [B, 32, C]

        // Self-attention: result attends to itself (both FP32)
        let out = attn(preAtt, preAtt)  // [B, 32, C]

        return out
    }
}
