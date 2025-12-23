import MLX
import MLXNN
import MLXFFT
import MLXRandom
import MLXFast
import Foundation

// MARK: - Configuration

public struct S3GenConfig {
    public let hiddenDim: Int
    public let melChannels: Int
    public let numMidBlocks: Int
    public let timeEmbDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let inputDim: Int
    
    public init(
        hiddenDim: Int = 256,
        melChannels: Int = 80,
        numMidBlocks: Int = 12,
        timeEmbDim: Int = 1024,
        numHeads: Int = 8,
        headDim: Int = 64,
        vocabSize: Int = 6561,
        inputDim: Int = 512
    ) {
        self.hiddenDim = hiddenDim
        self.melChannels = melChannels
        self.numMidBlocks = numMidBlocks
        self.timeEmbDim = timeEmbDim
        self.numHeads = numHeads
        self.headDim = headDim
        self.vocabSize = vocabSize
        self.inputDim = inputDim
    }
}

// MARK: - Helper Functions

private func mish(_ x: MLXArray) -> MLXArray {
    return x * tanh(softplus(x))
}

// Debug helper for tensor statistics
private func debugStats(_ x: MLXArray, name: String) {
    eval(x)
    let mean = x.mean().item(Float.self)
    let variance = x.variance().item(Float.self)
    let std = sqrt(variance)
    let minVal = x.min().item(Float.self)
    let maxVal = x.max().item(Float.self)
    print(String(format: "🔍 [%@] Shape: %@ | Mean: %.4f | Std: %.4f | Range: [%.4f, %.4f]",
                 name, "\(x.shape)", mean, std, minVal, maxVal))
}

/// Reflection padding for time dimension (axis 1)
/// Input: [Batch, Time, Channels]
/// Output: [Batch, Time + 2*padAmt, Channels]
private func reflectionPad1D(_ x: MLXArray, padAmt: Int) -> MLXArray {
    guard padAmt > 0 else { return x }

    let T = x.shape[1]

    // Left reflection: mirror positions [1...padAmt]
    let leftReflect = x[0..., 1...(padAmt), 0...]
    let leftPad = leftReflect[0..., (.stride(to: 0, by: -1)), 0...]

    // Right reflection: mirror positions [T-padAmt-1..<T-1]
    let rightReflect = x[0..., (T - padAmt - 1)..<(T - 1), 0...]
    let rightPad = rightReflect[0..., (.stride(to: 0, by: -1)), 0...]

    // Concatenate: [leftPad, x, rightPad]
    return concatenated([leftPad, x, rightPad], axis: 1)
}

// MARK: - Layers

public class CausalConv1d: Module {
    let conv: Conv1d
    let padAmount: Int
    let stride: Int

    public init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, bias: Bool = true) {
        self.stride = stride
        self.conv = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            bias: bias
        )
        self.padAmount = (kernelSize - 1) * dilation
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input x: [B, C, T] (channels-first)
        // MLX Conv1d expects: [B, T, C] (channels-last)
        var h = x.transposed(0, 2, 1) // [B, T, C]
        if padAmount > 0 {
            // Pad time dimension (axis 1 in channels-last format)
            h = padded(h, widths: [[0,0], [padAmount, 0], [0,0]])
        }
        h = conv(h) // [B, T, C_out]
        return h.transposed(0, 2, 1) // [B, C_out, T]
    }
}

public class Snake: Module, UnaryLayer {
    public let alpha: MLXArray
    let logScale: Bool

    public init(channels: Int, logScale: Bool = false) {
        self.logScale = logScale
        if logScale {
            self.alpha = MLXArray.zeros([channels])
        } else {
            self.alpha = MLXArray.ones([channels])
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: [B, L, C] (channels-last for MLX Conv1d compatibility)
        // alpha shape: [C] -> reshape to [1, 1, C] for broadcasting
        var a = alpha.reshaped([1, 1, -1])
        if logScale { a = exp(a) }
        
        // Match Python HiFi-GAN implementation:
        // alpha_clamped = alpha_sign * mx.maximum(alpha_abs, min_alpha) where min_alpha = 1e-4
        // This avoids division by zero while preserving sign and preventing exploding gradients for small alphas
        let minAlpha: Float = 1e-4
        let aSign = sign(a)
        let aAbs = abs(a)
        // Use sign * max(|a|, 1e-4). 
        // If a is exactly 0, sign is 0, so we need to handle that like Python
        // Python: alpha_clamped = mx.where(alpha_abs < 1e-9, min_alpha, alpha_clamped)
        
        var aClamped = aSign * maximum(aAbs, MLXArray(minAlpha))
        // If |a| < 1e-9, just use minAlpha (positive)
        let mask = aAbs .< 1e-9
        aClamped = MLX.where(mask, MLXArray(minAlpha), aClamped)
        
        let sinPart = sin(a * x)
        return x + (1.0 / aClamped) * sinPart * sinPart
    }
}

public class TimeMLP: Module {
    public let linear1: FixedLinear
    public let linear2: FixedLinear
    let inputDim: Int

    public static var debugEnabled: Bool = false

    public init(inputDim: Int = 320, embDim: Int = 1024) {
        self.inputDim = inputDim
        self.linear1 = FixedLinear(inputDim, embDim, name: "TimeMLP.linear1")
        self.linear2 = FixedLinear(embDim, embDim, name: "TimeMLP.linear2")
        super.init()
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        if TimeMLP.debugEnabled {
            eval(t)
            print("TimeMLP input t:"); fflush(stdout)
            print("  shape: \(t.shape)"); fflush(stdout)
            print("  values: \(t.asArray(Float.self))"); fflush(stdout)
        }

        let emb = sinusoidalEmbedding(t, dim: inputDim, scale: 1000.0)

        if TimeMLP.debugEnabled {
            eval(emb)
            print("TimeMLP sinusoidal embedding:"); fflush(stdout)
            print("  shape: \(emb.shape)"); fflush(stdout)
            print("  min/max: [\(emb.min().item(Float.self)), \(emb.max().item(Float.self))]"); fflush(stdout)
            print("  first 10: \(emb[0, 0..<10].asArray(Float.self))"); fflush(stdout)
            print("  last 10: \(emb[0, emb.shape[1]-10..<emb.shape[1]].asArray(Float.self))"); fflush(stdout)
        }

        let afterLinear1 = linear1(emb)
        if TimeMLP.debugEnabled {
            eval(afterLinear1)
            print("TimeMLP after Linear 1:"); fflush(stdout)
            print("  shape: \(afterLinear1.shape)"); fflush(stdout)
            print("  min/max: [\(afterLinear1.min().item(Float.self)), \(afterLinear1.max().item(Float.self))]"); fflush(stdout)
            print("  first 5: \(afterLinear1[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            print("  linear1.weight[0,:5]: \(linear1.weight[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            if let b = linear1.bias { print("  linear1.bias[:5]: \(b[0..<5].asArray(Float.self))"); fflush(stdout) }
        }

        let afterSiLU = silu(afterLinear1)
        if TimeMLP.debugEnabled {
            eval(afterSiLU)
            print("TimeMLP after SiLU:"); fflush(stdout)
            print("  min/max: [\(afterSiLU.min().item(Float.self)), \(afterSiLU.max().item(Float.self))]"); fflush(stdout)
            print("  first 5: \(afterSiLU[0, 0..<5].asArray(Float.self))"); fflush(stdout)
        }

        let result = linear2(afterSiLU)
        if TimeMLP.debugEnabled {
            eval(result)
            print("TimeMLP after Linear 2:"); fflush(stdout)
            print("  shape: \(result.shape)"); fflush(stdout)
            print("  min/max: [\(result.min().item(Float.self)), \(result.max().item(Float.self))]"); fflush(stdout)
            print("  first 5: \(result[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            print("  linear2.weight[0,:5]: \(linear2.weight[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            if let b = linear2.bias { print("  linear2.bias[:5]: \(b[0..<5].asArray(Float.self))"); fflush(stdout) }
        }

        return result
    }

    private func sinusoidalEmbedding(_ t: MLXArray, dim: Int, scale: Float = 1.0) -> MLXArray {
        // Sinusoidal position embedding
        // Python decoder uses this for time conditioning
        let halfDim = dim / 2
        let embScale = log(Float(10000)) / Float(halfDim - 1)
        let emb = exp(MLXArray(0..<halfDim) * (-embScale))
        let tExpanded = t.reshaped([-1, 1])
        let embExpanded = emb.reshaped([1, -1])
        let angles = scale * tExpanded * embExpanded
        let features = concatenated([sin(angles), cos(angles)], axis: -1)
        if dim % 2 == 1 {
             return concatenated([features, MLXArray.zeros([features.shape[0], 1])], axis: -1)
        }
        return features
    }
}

public class FlowMLP: Module {
    // Use array for weight loading compatibility
    public let layers: [FixedLinear]

    public static var debugEnabled: Bool = false

    public init(dim: Int, mult: Int = 4) {
        self.layers = [
            FixedLinear(dim, dim * mult, name: "FlowMLP.layers.0"),
            FixedLinear(dim * mult, dim, name: "FlowMLP.layers.1")
        ]
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if FlowMLP.debugEnabled { eval(h); print("  FF input: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
        h = layers[0](h)
        if FlowMLP.debugEnabled { eval(h); print("  FF after layers[0]: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
        h = gelu(h)
        if FlowMLP.debugEnabled { eval(h); print("  FF after gelu: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
        h = layers[1](h)
        if FlowMLP.debugEnabled { eval(h); print("  FF after layers[1]: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
        return h
    }
}

public class MultiHeadAttention: Module {
    public let queryProj: FixedLinear
    public let keyProj: FixedLinear
    public let valueProj: FixedLinear
    public let outProj: FixedLinear
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dims: Int, numHeads: Int, headDim: Int, qkvBias: Bool = false, outBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let innerDim = numHeads * headDim
        // Python DiffusersAttention: qkv_bias=False, out_bias=True
        self.queryProj = FixedLinear(dims, innerDim, bias: qkvBias, name: "MultiHeadAttention.queryProj")
        self.keyProj = FixedLinear(dims, innerDim, bias: qkvBias, name: "MultiHeadAttention.keyProj")
        self.valueProj = FixedLinear(dims, innerDim, bias: qkvBias, name: "MultiHeadAttention.valueProj")
        self.outProj = FixedLinear(innerDim, dims, bias: outBias, name: "MultiHeadAttention.outProj")
        super.init()
    }
    
    public static var debugEnabled: Bool = false
    public static var debugId: String = ""

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        let debug = MultiHeadAttention.debugEnabled

        var q = queryProj(x)
        var k = keyProj(x)
        var v = valueProj(x)

        if debug {
            eval(q); eval(k); eval(v)
            print("  ATTN[\(MultiHeadAttention.debugId)] Q: [\(q.min().item(Float.self)), \(q.max().item(Float.self))]")
            print("  ATTN[\(MultiHeadAttention.debugId)] K: [\(k.min().item(Float.self)), \(k.max().item(Float.self))]")
            print("  ATTN[\(MultiHeadAttention.debugId)] V: [\(v.min().item(Float.self)), \(v.max().item(Float.self))]")
        }

        // Reshape to (B, numHeads, L, headDim) for scaled dot product attention
        q = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        v = v.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Manual attention computation to match Python's DiffusersAttention when mask is provided
        // Python uses manual attention (not fast path) when attention_mask is not None
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale

        if debug {
            eval(scores)
            print("  ATTN[\(MultiHeadAttention.debugId)] scores (before mask): [\(scores.min().item(Float.self)), \(scores.max().item(Float.self))]")
        }

        if let m = mask {
            // Additive bias mask: broadcast m to [B, numHeads, L, L]
            // m is expected to be [B, 1, L, L] or [L, L] or similar
            scores = scores + m
        }
        let probs = softmax(scores, axis: -1)

        if debug {
            eval(probs)
            print("  ATTN[\(MultiHeadAttention.debugId)] probs: [\(probs.min().item(Float.self)), \(probs.max().item(Float.self))]")
        }

        var output = matmul(probs, v)

        if debug {
            eval(output)
            print("  ATTN[\(MultiHeadAttention.debugId)] output (before reshape): [\(output.min().item(Float.self)), \(output.max().item(Float.self))]")
        }

        output = output.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        let result = outProj(output)

        if debug {
            eval(result)
            print("  ATTN[\(MultiHeadAttention.debugId)] result (after outProj): [\(result.min().item(Float.self)), \(result.max().item(Float.self))]")
        }

        return result
    }
}

// MARK: - Upsample Encoder Components

public class Upsample1D: Module {
    public let conv: Conv1d  // Made public for testing
    let stride: Int
    
    public init(channels: Int, outChannels: Int, stride: Int = 2) {
        self.stride = stride
        // Matches Python: kernel=stride*2+1, stride=1, padding=0
        // BUT we need padding manually for MLX Conv1d?
        // Python Upsample1D pads: (0,0, stride*2, 0).
        self.conv = Conv1d(inputChannels: channels, outputChannels: outChannels, kernelSize: stride * 2 + 1, stride: 1, padding: 0, dilation: 1)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C, T]
        // Repeat
        // Note: MLX doesn't have `repeat` along axis easily like pytorch, use tile?
        // x shape [B, C, T]. We want [B, C, T*stride]
        // Expand, tile, reshape
        // [B, C, T, 1] -> [B, C, T, stride] -> [B, C, T*stride]
        let xExp = x.expandedDimensions(axis: 3)
        let xTiled = tiled(xExp, repetitions: [1, 1, 1, stride])
        var out = xTiled.reshaped([x.shape[0], x.shape[1], -1]) // [B, C, T*S]
        
        // Pad left
        let padLen = stride * 2
        out = padded(out, widths: [[0,0], [0,0], [padLen, 0]])
        
        // Conv expects [B, Time, Channels]
        out = out.transposed(0, 2, 1)
        out = conv(out)
        out = out.transposed(0, 2, 1)
        return out
    }
}

public class PreLookahead: Module {
    public let conv1: Conv1d
    public let conv2: Conv1d
    public let preLookaheadLen: Int
    
    public init(channels: Int, preLookaheadLen: Int = 3) {
        self.preLookaheadLen = preLookaheadLen
        self.conv1 = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: preLookaheadLen + 1, stride: 1, padding: 0)
        self.conv2 = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 1, padding: 0)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C] or [B, C, T]? 
        // Python calls it with (B, T, C). But Conformer usually works (B, T, C).
        // Let's assume input is [B, T, C].
        
        var out = x
        // Pad for lookahead
        // Python: pad(0, pre_lookahead_len) on time axis (1)
        out = padded(out, widths: [[0,0], [0, preLookaheadLen], [0,0]])
        
        // Conv1 needs [B, T, C] in MLX.
        // Python uses negative_slope=0.01 (MLX default)
        out = leakyRelu(conv1(out), negativeSlope: 0.01)
        
        // Pad for conv2
        out = padded(out, widths: [[0,0], [2, 0], [0,0]])
        out = conv2(out)
        
        return out + x
    }
}

// Reusing FlowTransformerBlock logic but renamed for clarity/mapping
public class S3ConformerFeedForward: Module {
    public let w1: FixedLinear  // Made public for debugging
    public let w2: FixedLinear  // Made public for debugging

    public init(dim: Int, mult: Int = 4, dropout: Float = 0.1) {
        // Python PositionwiseFeedForward: w_1 -> act -> dropout -> w_2
        // Typically hidden = dim * mult
        let hidden = dim * mult
        self.w1 = FixedLinear(dim, hidden, name: "S3ConformerFeedForward.w1")
        self.w2 = FixedLinear(hidden, dim, name: "S3ConformerFeedForward.w2")
        super.init()
    }
    
    public static var debugEnabled: Bool = false

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = w1(x)
        if S3ConformerFeedForward.debugEnabled {
            eval(h)
            print("    FF after w1: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        h = silu(h) // Swish/SiLU for Conformer
        if S3ConformerFeedForward.debugEnabled {
            eval(h)
            print("    FF after silu: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        h = w2(h)
        if S3ConformerFeedForward.debugEnabled {
            eval(h)
            print("    FF after w2: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        return h
    }

    /// Load weights for feed forward
    public func load(weights: [String: MLXArray], prefix: String) {
        // NOTE: w_1 and w_2 weights are loaded via ChatterboxEngine.update()
        // DO NOT load them here to avoid double-transpose bug
        if let w = weights["\(prefix).w_1.weight"] {
            print("  Found \(prefix).w_1.weight: \(w.shape) - will be loaded via ChatterboxEngine.update()")
        }
        if let w = weights["\(prefix).w_2.weight"] {
            print("  Found \(prefix).w_2.weight: \(w.shape) - will be loaded via ChatterboxEngine.update()")
        }
    }
}

public class ConformerBlock: Module {
    let normMHA: LayerNorm
    public let attention: RelPositionMultiHeadAttention  // Made public for debugging
    let normFF: LayerNorm
    public let feedForward: S3ConformerFeedForward  // Made public for debugging
    
    public init(embedDim: Int, numHeads: Int = 8, headDim: Int = 64, dropout: Float = 0.1) {
        // Python uses eps=1e-12 for LayerNorm
        self.normMHA = LayerNorm(dimensions: embedDim, eps: 1e-12)
        self.attention = RelPositionMultiHeadAttention(dModel: embedDim, numHeads: numHeads, dropout: dropout)
        self.normFF = LayerNorm(dimensions: embedDim, eps: 1e-12)
        // Conformer usually uses mult=4. 
        self.feedForward = S3ConformerFeedForward(dim: embedDim, mult: 4, dropout: dropout)
        super.init()
    }
    
    public static var debugEnabled: Bool = false

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, posEmb: MLXArray) -> MLXArray {
        // x: [B, T, C]
        var h = x
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  ConformerBlock input: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        let res1 = h
        h = normMHA(h)
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  After normMHA: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        h = attention(h, mask: mask, posEmb: posEmb)
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  After attention: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        h = h + res1
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  After res1: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        let res2 = h
        h = normFF(h)
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  After normFF: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        if ConformerBlock.debugEnabled {
            S3ConformerFeedForward.debugEnabled = true
        }
        h = feedForward(h)
        if ConformerBlock.debugEnabled {
            S3ConformerFeedForward.debugEnabled = false
            eval(h)
            print("  After feedForward: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }

        h = h + res2
        if ConformerBlock.debugEnabled {
            eval(h)
            print("  ConformerBlock output: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        return h
    }

    /// Load weights for this ConformerBlock
    public func load(weights: [String: MLXArray], prefix: String) {
        // Load norm_mha weights
        if let w = weights["\(prefix).norm_mha.weight"] {
            normMHA.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).norm_mha.bias"] {
            normMHA.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load attention weights
        attention.load(weights: weights, prefix: "\(prefix).self_attn")

        // Load norm_ff weights
        if let w = weights["\(prefix).norm_ff.weight"] {
            normFF.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).norm_ff.bias"] {
            normFF.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load feed_forward weights
        feedForward.load(weights: weights, prefix: "\(prefix).feed_forward")
    }
}

public class UpsampleEncoder: Module {
    // Embed
    public let embedLinear: FixedLinear
    public let embedNorm: LayerNorm // Added missing Norm
    public let posEnc: EspnetRelPositionalEncoding
    public let preLookaheadLayer: Module  // Can be PreLookahead or PreLookaheadLayer
    public let encoders: [ConformerBlock]
    public let upLayer: Upsample1D
    public let upEmbedLinear: FixedLinear
    public let upEmbedNorm: LayerNorm // Added missing Norm
    public let upPosEnc: EspnetRelPositionalEncoding
    public let upEncoders: [ConformerBlock]
    public let afterNorm: LayerNorm

    public init(inputDim: Int = 512, outputDim: Int = 512, weights: [String: MLXArray]? = nil) {
        print("🔧 UpsampleEncoder.init START: inputDim=\(inputDim), outputDim=\(outputDim)"); fflush(stdout)

        print("🔧  [1/10] Creating embedLinear(\(inputDim), \(outputDim))..."); fflush(stdout)
        self.embedLinear = FixedLinear(inputDim, outputDim, name: "UpsampleEncoder.embedLinear")
        print("🔧  [1/10] ✓ embedLinear created"); fflush(stdout)

        print("🔧  [2/10] Creating embedNorm(dim=\(outputDim)) with identity init..."); fflush(stdout)
        self.embedNorm = LayerNorm(dimensions: outputDim, eps: 1e-5)
        // CRITICAL FIX: Python's Conv1dSubsampling2 does NOT have a norm layer
        // Initialize as identity transform (gamma=1, beta=0) to avoid signal suppression
        self.embedNorm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([outputDim]),
            "bias": MLXArray.zeros([outputDim])
        ]))
        print("🔧  [2/10] ✓ embedNorm created with identity init (gamma=1, beta=0)"); fflush(stdout)

        print("🔧  [3/10] Creating posEnc(dModel=\(outputDim))..."); fflush(stdout)
        self.posEnc = EspnetRelPositionalEncoding(dModel: outputDim)
        print("🔧  [3/10] ✓ posEnc created"); fflush(stdout)

        // FIX: Use PreLookaheadLayer with weight loading instead of random weights
        print("🔧  [4/10] Creating preLookaheadLayer..."); fflush(stdout)
        if let w = weights {
            // Use snake_case to match HuggingFace weight keys
            self.preLookaheadLayer = PreLookaheadLayer(dim: outputDim, weights: w, prefix: "encoder.pre_lookahead_layer")
        } else {
            self.preLookaheadLayer = PreLookahead(channels: outputDim)
        }
        print("🔧  [4/10] ✓ preLookaheadLayer created"); fflush(stdout)

        print("🔧  [5/10] Creating 6 ConformerBlock encoders(embedDim=\(outputDim))..."); fflush(stdout)
        var encs: [ConformerBlock] = []
        for i in 0..<6 {
            print("🔧   - Creating encoder[\(i)]..."); fflush(stdout)
            encs.append(ConformerBlock(embedDim: outputDim))
            print("🔧   - encoder[\(i)] created"); fflush(stdout)
        }
        self.encoders = encs
        print("🔧  [5/10] ✓ All encoders created"); fflush(stdout)
        
        // Correctly use Upsample1D matching Python (Repeat + Conv)
        print("🔧  [6/10] Creating upLayer(channels=\(outputDim), stride=2)..."); fflush(stdout)
        self.upLayer = Upsample1D(channels: outputDim, outChannels: outputDim, stride: 2)
        print("🔧  [6/10] ✓ upLayer created"); fflush(stdout)

        print("🔧  [7/10] Creating upEmbedLinear(\(outputDim), \(outputDim))..."); fflush(stdout)
        self.upEmbedLinear = FixedLinear(outputDim, outputDim, name: "UpsampleEncoder.upEmbedLinear")
        print("🔧  [7/10] ✓ upEmbedLinear created"); fflush(stdout)

        print("🔧  [8/10] Creating upEmbedNorm(dim=\(outputDim)) with identity init..."); fflush(stdout)
        self.upEmbedNorm = LayerNorm(dimensions: outputDim, eps: 1e-5)
        // CRITICAL FIX: Python's Conv1dSubsampling2 does NOT have a norm layer
        // Initialize as identity transform (gamma=1, beta=0) to avoid signal suppression
        self.upEmbedNorm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([outputDim]),
            "bias": MLXArray.zeros([outputDim])
        ]))
        print("🔧  [8/10] ✓ upEmbedNorm created with identity init (gamma=1, beta=0)"); fflush(stdout)

        print("🔧  [9/10] Creating upPosEnc(dModel=\(outputDim))..."); fflush(stdout)
        self.upPosEnc = EspnetRelPositionalEncoding(dModel: outputDim)
        print("🔧  [9/10] ✓ upPosEnc created"); fflush(stdout)

        print("🔧  [10/10] Creating 4 ConformerBlock upEncoders(embedDim=\(outputDim))..."); fflush(stdout)
        var upEncs: [ConformerBlock] = []
        for i in 0..<4 {
            print("🔧    - Creating upEncoder[\(i)]..."); fflush(stdout)
            upEncs.append(ConformerBlock(embedDim: outputDim))
            print("🔧    - upEncoder[\(i)] created"); fflush(stdout)
        }
        self.upEncoders = upEncs
        print("🔧  [10/10] ✓ All upEncoders created"); fflush(stdout)

        print("🔧  Creating afterNorm..."); fflush(stdout)
        self.afterNorm = LayerNorm(dimensions: outputDim)
        print("🔧  ✓ afterNorm created"); fflush(stdout)

        print("🔧  Calling super.init()..."); fflush(stdout)
        super.init()
        print("🔧 UpsampleEncoder.init COMPLETE"); fflush(stdout)
    }

    public func callAsFunction(_ x: MLXArray, seqLen: MLXArray? = nil) -> MLXArray {
        print("DEBUG Encoder: callAsFunction called, x.shape = \(x.shape)"); fflush(stdout)
        // x: [B, T, D]
        // seqLen: [B] - actual sequence lengths for attention masking

        // Create attention mask if sequence length is provided
        // Mask shape: [B, 1, T] where mask[b, 0, t] = 1 if t < seqLen[b] else 0
        print("DEBUG Encoder: Creating mask..."); fflush(stdout)
        var mask: MLXArray? = nil
        if let len = seqLen {
            let B = x.shape[0]
            let T = x.shape[1]
            // Create positions: [0, 1, 2, ..., T-1] -> [1, T]
            let positions = MLXArray(0..<T).reshaped([1, T]).asType(.float32)
            // Expand seqLen to [B, 1]
            let lenExpanded = len.asType(.float32).reshaped([-1, 1])
            // Compare: positions < seqLen -> [B, T]
            let maskBT = less(positions, lenExpanded).asType(.float32)
            // Add dimension: [B, 1, T]
            mask = maskBT.expandedDimensions(axis: 1)
        }
        print("DEBUG Encoder: Mask created"); fflush(stdout)

        // 1. Embed + RelPos
        print("DEBUG Encoder: 1. Calling embedLinear..."); fflush(stdout)
        var h = embedLinear(x)
        eval(h)
        print("DEBUG Encoder: ⚠️  After embedLinear - Range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]"); fflush(stdout)
        print("DEBUG Encoder: embedLinear done, calling embedNorm..."); fflush(stdout)

        // CHECK: LayerNorm weights
        print("DEBUG Encoder: ⚠️  Checking embedNorm weights..."); fflush(stdout)
        if let gamma = embedNorm.weight {
            eval(gamma)
            print("DEBUG Encoder: embedNorm gamma: shape=\(gamma.shape), range=[\(gamma.min().item(Float.self)), \(gamma.max().item(Float.self))], mean=\(gamma.mean().item(Float.self))"); fflush(stdout)
        } else {
            print("DEBUG Encoder: ❌ embedNorm has NO gamma (weight is nil)"); fflush(stdout)
        }
        if let beta = embedNorm.bias {
            eval(beta)
            print("DEBUG Encoder: embedNorm beta: shape=\(beta.shape), range=[\(beta.min().item(Float.self)), \(beta.max().item(Float.self))]"); fflush(stdout)
        } else {
            print("DEBUG Encoder: ❌ embedNorm has NO beta (bias is nil)"); fflush(stdout)
        }

        h = embedNorm(h)
        print("DEBUG Encoder: embedNorm done"); fflush(stdout)

        // CHECK: embedNorm output FULL statistics for Python comparison
        eval(h)
        let embedNormMin = h.min().item(Float.self)
        let embedNormMax = h.max().item(Float.self)
        let embedNormMean = h.mean().item(Float.self)
        // Calculate std: sqrt(variance)
        let embedNormVariance = ((h - h.mean()) * (h - h.mean())).mean()
        eval(embedNormVariance)
        let embedNormStd = sqrt(embedNormVariance.item(Float.self))
        print("DEBUG Encoder: ⚠️  embedNorm output FULL STATS:"); fflush(stdout)
        print("  Range: [\(embedNormMin), \(embedNormMax)]"); fflush(stdout)
        print("  Mean: \(embedNormMean)"); fflush(stdout)
        print("  Std: \(embedNormStd)"); fflush(stdout)
        print("  Python ref: mean=0.001553, std=0.150711, range=[-0.630, 0.768]"); fflush(stdout)
        if abs(embedNormStd - 0.150711) < 0.01 {
            print("  ✅ Std matches Python!"); fflush(stdout)
        } else {
            print("  ❌ Std DIFFERS from Python by \((embedNormStd - 0.150711) / 0.150711 * 100)%"); fflush(stdout)
        }
        if abs(embedNormMax - 0.768) < 0.05 {
            print("  ✅ Max matches Python!"); fflush(stdout)
        } else {
            print("  ❌ Max is \(embedNormMax / 0.768 * 100)% of Python"); fflush(stdout)
        }

        print("DEBUG Encoder: Calling posEnc (xscale=\(posEnc.xscale))..."); fflush(stdout)
        print("DEBUG Encoder: posEnc.pe shape = \(posEnc.pe.shape) (should be [1, 9999, 512] if loaded)"); fflush(stdout)
        print("DEBUG Encoder: posEnc.pe first 3 = \(posEnc.pe[0, 0, 0..<3])"); fflush(stdout)
        // DEBUG: Check h BEFORE posEnc
        eval(h)
        let hBeforeMin = h.min().item(Float.self)
        let hBeforeMax = h.max().item(Float.self)
        print("DEBUG Encoder: h BEFORE posEnc - Range: [\(hBeforeMin), \(hBeforeMax)]"); fflush(stdout)
        let (hScaled, posEmb) = posEnc(h)
        h = hScaled
        // DEBUG: Check h AFTER posEnc (should be h * xscale)
        eval(h)
        let hAfterMin = h.min().item(Float.self)
        let hAfterMax = h.max().item(Float.self)
        let actualRatio = hAfterMax / hBeforeMax
        print("DEBUG Encoder: h AFTER posEnc - Range: [\(hAfterMin), \(hAfterMax)]"); fflush(stdout)
        print("DEBUG Encoder: Actual scaling ratio: \(actualRatio)x (expected \(posEnc.xscale)x)"); fflush(stdout)
        print("DEBUG Encoder: posEmb shape = \(posEmb.shape)"); fflush(stdout)

        // CHECK: After posEnc scaling
        eval(h)
        print("DEBUG Encoder: ⚠️  After posEnc - Range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]"); fflush(stdout)
        let scalingRatio = h.max().item(Float.self) / 0.768  // Compare to Python's after_embed max
        print("DEBUG Encoder: Scaling ratio from embedNorm output: \(scalingRatio)x"); fflush(stdout)

        // 2. PreLookahead
        print("DEBUG Encoder: 2. Calling preLookaheadLayer..."); fflush(stdout)
        if let layer = preLookaheadLayer as? PreLookahead {
            h = layer(h)
        } else if let layer = preLookaheadLayer as? PreLookaheadLayer {
            h = layer(h)
        }
        eval(h)
        print("DEBUG Encoder: After preLookaheadLayer - Range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]"); fflush(stdout)
        print("DEBUG Encoder: After preLookahead[0,0,:5]: \(h[0, 0, 0..<5].asArray(Float.self))"); fflush(stdout)

        // 3. Encoders (pass posEmb and mask)
        print("DEBUG Encoder: 3. Running \(encoders.count) encoder layers..."); fflush(stdout)
        for (i, layer) in encoders.enumerated() {
            // Track input for amplification calculation
            let hInput = h
            eval(hInput)
            let inputMax = max(abs(hInput.min().item(Float.self)), abs(hInput.max().item(Float.self)))

            // Enable debug for first layer only
            if i == 0 {
                ConformerBlock.debugEnabled = true
            }
            h = layer(h, mask: mask, posEmb: posEmb)
            if i == 0 {
                ConformerBlock.debugEnabled = false
            }

            // Track output and calculate amplification
            eval(h)
            let outputMin = h.min().item(Float.self)
            let outputMax = h.max().item(Float.self)
            let outputMaxAbs = max(abs(outputMin), abs(outputMax))
            let amplification = inputMax > 0 ? outputMaxAbs / inputMax : 0.0

            print("DEBUG Encoder: Encoder[\(i)]: input_max_abs=\(inputMax), output_max_abs=\(outputMaxAbs), amplification=\(String(format: "%.4f", amplification))x"); fflush(stdout)
        }
        print("DEBUG Encoder: All encoder layers done"); fflush(stdout)
        eval(h)
        print("DEBUG Encoder: After 6 conformers - Range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]"); fflush(stdout)
        print("DEBUG Encoder: After 6 conformers[0,0,:5]: \(h[0, 0, 0..<5].asArray(Float.self))"); fflush(stdout)

        // MAGNITUDE CHECK: Before Upsample
        print("DEBUG Encoder: ⚠️  MAGNITUDE CHECK - Before upLayer"); fflush(stdout)
        eval(h)
        let preUpMin = h.min().item(Float.self)
        let preUpMax = h.max().item(Float.self)
        print("DEBUG Encoder: Before upLayer - Range: [\(preUpMin), \(preUpMax)], shape: \(h.shape)"); fflush(stdout)

        // 4. Upsample
        print("DEBUG Encoder: 4. Transposing for upsample..."); fflush(stdout)
        h = h.transposed(0, 2, 1) // [B, C, T]

        // CHECK: Upsample1D conv weights
        print("DEBUG Encoder: ⚠️  Checking upLayer.conv weights..."); fflush(stdout)
        let convWeight = upLayer.conv.weight
        eval(convWeight)
        print("DEBUG Encoder: upLayer.conv.weight: shape=\(convWeight.shape), range=[\(convWeight.min().item(Float.self)), \(convWeight.max().item(Float.self))], mean=\(convWeight.mean().item(Float.self))"); fflush(stdout)

        print("DEBUG Encoder: Calling upLayer..."); fflush(stdout)
        h = upLayer(h)
        print("DEBUG Encoder: upLayer done, transposing back..."); fflush(stdout)
        h = h.transposed(0, 2, 1) // [B, T, C]
        print("DEBUG Encoder: Transpose back done, h.shape = \(h.shape)"); fflush(stdout)

        // MAGNITUDE CHECK: After Upsample
        print("DEBUG Encoder: ⚠️  MAGNITUDE CHECK - After upLayer"); fflush(stdout)
        eval(h)
        let postUpMin = h.min().item(Float.self)
        let postUpMax = h.max().item(Float.self)
        print("DEBUG Encoder: After upLayer - Range: [\(postUpMin), \(postUpMax)], shape: \(h.shape)"); fflush(stdout)
        print("DEBUG Encoder: Magnitude change: \(preUpMax) -> \(postUpMax) (ratio: \(postUpMax/preUpMax))"); fflush(stdout)
        print("DEBUG Encoder: CHECKPOINT A - About to check boundary trace"); fflush(stdout)

        // Debug: Print boundary values after upsampling
        if h.shape[1] >= 548 {
            print("DEBUG Encoder: Boundary trace condition met, h.shape[1] = \(h.shape[1])"); fflush(stdout)
            print("DEBUG Encoder: About to eval(h) for boundary trace..."); fflush(stdout)
            eval(h)
            print("DEBUG Encoder: eval(h) for boundary trace done"); fflush(stdout)
            print("\n====== ENCODER BOUNDARY TRACE ======"); fflush(stdout)
            print("After upsampling: \(h.shape)"); fflush(stdout)
            print("DEBUG Encoder: About to get h.min() for range..."); fflush(stdout)
            let hMin = h.min()
            eval(hMin)
            let hMax = h.max()
            eval(hMax)
            print("Range: [\(hMin.item(Float.self)), \(hMax.item(Float.self))]"); fflush(stdout)
            print("\n⚠️  Boundary positions:"); fflush(stdout)
            // Only check positions that exist in this sequence (h.shape[1] = \(h.shape[1]))
            let maxPos = h.shape[1] - 1
            for t in [498, 499, 500] {
                if t <= maxPos {
                    print("DEBUG Encoder: Getting position \(t)..."); fflush(stdout)
                    let v0 = h[0, t, 0].item(Float.self)
                    let v1 = h[0, t, 1].item(Float.self)
                    let v2 = h[0, t, 2].item(Float.self)
                    print(String(format: "⚠️ Pos %3d: [%10.6f, %10.6f, %10.6f, ...]", t, v0, v1, v2)); fflush(stdout)
                }
            }
            print("\nPython comparison:"); fflush(stdout)
            print("  Python range: [-595.097, 851.417]"); fflush(stdout)
            print("  ⚠️  Pos 498: [-16.364107, -14.741450, -57.778362]"); fflush(stdout)
            print("  ⚠️  Pos 499: [ 57.294426,  -1.231192, -85.779099]"); fflush(stdout)
            print("  ⚠️  Pos 500: [ 76.392265,  19.593866, -79.950188]"); fflush(stdout)
            if hMax.item(Float.self) > 50.0 {
                print("\n✅ Swift HAS huge boundary spikes (like Python)"); fflush(stdout)
            } else {
                print("\n❌ Swift MISSING huge boundary spikes - ROOT CAUSE!"); fflush(stdout)
            }
            print("====================================\n"); fflush(stdout)
        }
        print("DEBUG Encoder: CHECKPOINT B - After boundary trace check"); fflush(stdout)

        // After upsampling, sequence length doubles
        print("DEBUG Encoder: CHECKPOINT C - Before mask doubling, mask is nil: \(mask == nil)"); fflush(stdout)
        if var m = mask {
            // Repeat each element twice along time dimension
            // [B, 1, T] -> [B, 1, 2*T]
            // Simply concatenate mask with itself to double the sequence
            print("DEBUG Encoder: CHECKPOINT D - Doubling mask"); fflush(stdout)
            m = concatenated([m, m], axis: 2)
            mask = m
            print("DEBUG Encoder: CHECKPOINT E - Mask doubled"); fflush(stdout)
        }
        print("DEBUG Encoder: CHECKPOINT F - After mask doubling"); fflush(stdout)

        // 5. UpEmbed
        eval(h)  // Force evaluation before accessing shape
        print("DEBUG Encoder: 5. Starting upEmbed, h.shape = \(h.shape)"); fflush(stdout)
        print("DEBUG Encoder: upEmbedLinear.weight shape = \(upEmbedLinear.weight.shape)"); fflush(stdout)
        eval(upEmbedLinear.weight)
        h = upEmbedLinear(h)
        print("DEBUG Encoder: After upEmbedLinear"); fflush(stdout)
        eval(h)  // Force eval before norm
        print("DEBUG Encoder: Before upEmbedNorm, h.shape = \(h.shape)"); fflush(stdout)
        h = upEmbedNorm(h)
        print("DEBUG Encoder: After upEmbedNorm"); fflush(stdout)
        eval(h)
        print("DEBUG Encoder: After upEmbedNorm eval, h.shape = \(h.shape)"); fflush(stdout)
        eval(h)  // Force eval before upPosEnc
        print("DEBUG Encoder: upPosEnc.pe shape = \(upPosEnc.pe.shape) (should be [1, 9999, 512])")
        let (hUp, posEmbUp) = upPosEnc(h)
        eval(posEmbUp)  // Force eval of posEmbUp
        print("DEBUG Encoder: After upPosEnc, posEmbUp.shape = \(posEmbUp.shape)")
        h = hUp
        eval(h)  // Force eval before accessing shape
        print("DEBUG Encoder: Set h = hUp, h.shape = \(h.shape)")

        // 6. Up Encoders (with upsampled mask)
        print("DEBUG Encoder: About to run \(upEncoders.count) upEncoders, h.shape = \(h.shape), mask = \(mask?.shape ?? [0])")
        for (i, layer) in upEncoders.enumerated() {
            print("DEBUG Encoder: Running upEncoder[\(i)]...")
            h = layer(h, mask: mask, posEmb: posEmbUp)
            print("DEBUG Encoder: upEncoder[\(i)] done, h.shape = \(h.shape)")
        }

        // 7. Final Norm
        h = afterNorm(h)
        print("DEBUG Encoder: About to return h.shape = \(h.shape)")

        return h
    }

    /// Load trained weights from Python model
    public func load(weights: [String: MLXArray], prefix: String = "encoder") {
        print("UpsampleEncoder: Loading weights with prefix '\(prefix)'")

        // NOTE: embedLinear weights are loaded via ChatterboxEngine.update()
        // DO NOT load them here to avoid double-transpose bug
        if let w = weights["\(prefix).embedLinear.weight"] {
            print("  Found \(prefix).embedLinear.weight: \(w.shape) - will be loaded via ChatterboxEngine.update()")
        }

        // 2. Load embedNorm weights
        // Load LayerNorm weights if they exist (Python model has these)
        if let w = weights["\(prefix).embedNorm.weight"] {
            eval(w)
            let avgWeight = abs(w).mean().item(Float.self)
            print("  📊 embedNorm.weight FROM FILE: shape=\(w.shape), range=[\(w.min().item(Float.self)), \(w.max().item(Float.self))], mean=\(avgWeight)")
            embedNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  ✅ Loaded embedNorm.weight")
        }
        if let b = weights["\(prefix).embedNorm.bias"] {
            eval(b)
            let avgBias = abs(b).mean().item(Float.self)
            print("  📊 embedNorm.bias FROM FILE: shape=\(b.shape), range=[\(b.min().item(Float.self)), \(b.max().item(Float.self))], mean=\(avgBias)")
            embedNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
            print("  ✅ Loaded embedNorm.bias")
        }

        // 3. Load pos_enc.pe
        if let pe = weights["\(prefix).embed.pos_enc.pe"] {
            posEnc.pe = pe
            print("  ✅ Loaded pos_enc.pe shape=\(pe.shape)")
        }

        // 4. Load pre_lookahead_layer weights (if using PreLookaheadLayer)
        if let layer = preLookaheadLayer as? PreLookaheadLayer {
            // PreLookaheadLayer loads weights in its init, so just verify
            print("  ✅ PreLookaheadLayer weights loaded in init")
        }

        // 5. Load main encoder blocks (0-5)
        for i in 0..<encoders.count {
            let blockPrefix = "\(prefix).encoders.\(i)"
            encoders[i].load(weights: weights, prefix: blockPrefix)
            print("  ✅ Loaded encoder block \(i)")

            // Debug: Check first encoder block's feed-forward weights
            if i == 0 {
                if let w1Weight = weights["\(blockPrefix).feed_forward.w_1.weight"] {
                    eval(w1Weight)
                    print("    DEBUG: encoder[0].ff.w_1.weight: shape=\(w1Weight.shape), range=[\(w1Weight.min().item(Float.self)), \(w1Weight.max().item(Float.self))], mean=\(w1Weight.mean().item(Float.self))")
                }
                if let w2Weight = weights["\(blockPrefix).feed_forward.w_2.weight"] {
                    eval(w2Weight)
                    print("    DEBUG: encoder[0].ff.w_2.weight: shape=\(w2Weight.shape), range=[\(w2Weight.min().item(Float.self)), \(w2Weight.max().item(Float.self))], mean=\(w2Weight.mean().item(Float.self))")
                }
            }
        }

        // NOTE: up_layer.conv weights are loaded later via ChatterboxEngine.update()
        // DO NOT transpose weights here to avoid double-transpose bug
        // Conv1d weights will be transposed once in remapS3Keys() and applied via update()
        if let w = weights["\(prefix).up_layer.conv.weight"] {
            print("  Found up_layer.conv.weight: \(w.shape) - will be loaded via ChatterboxEngine.update()")
        }

        // NOTE: upEmbedLinear weights are loaded via ChatterboxEngine.update()
        // DO NOT load them here to avoid double-transpose bug
        if let w = weights["\(prefix).upEmbedLinear.weight"] {
            print("  Found \(prefix).upEmbedLinear.weight: \(w.shape) - will be loaded via ChatterboxEngine.update()")
        }
        // Load upEmbedNorm weights if they exist (Python model has these)
        if let w = weights["\(prefix).upEmbedNorm.weight"] {
            eval(w)
            let avgWeight = abs(w).mean().item(Float.self)
            print("  📊 upEmbedNorm.weight FROM FILE: shape=\(w.shape), range=[\(w.min().item(Float.self)), \(w.max().item(Float.self))], mean=\(avgWeight)")
            upEmbedNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  ✅ Loaded upEmbedNorm.weight")
        }
        if let b = weights["\(prefix).upEmbedNorm.bias"] {
            eval(b)
            let avgBias = abs(b).mean().item(Float.self)
            print("  📊 upEmbedNorm.bias FROM FILE: shape=\(b.shape), range=[\(b.min().item(Float.self)), \(b.max().item(Float.self))], mean=\(avgBias)")
            upEmbedNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
            print("  ✅ Loaded upEmbedNorm.bias")
        }
        if let pe = weights["\(prefix).up_embed.pos_enc.pe"] {
            upPosEnc.pe = pe
            print("  ✅ Loaded up_embed.pos_enc.pe shape=\(pe.shape)")
        }

        // 8. Load up encoder blocks (0-3)
        for i in 0..<upEncoders.count {
            let blockPrefix = "\(prefix).up_encoders.\(i)"
            upEncoders[i].load(weights: weights, prefix: blockPrefix)
            print("  ✅ Loaded up_encoder block \(i)")
        }

        // 9. Load after_norm weights
        if let w = weights["\(prefix).after_norm.weight"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
            print("  ✅ Loaded after_norm.weight")
        }
        if let b = weights["\(prefix).after_norm.bias"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
            print("  ✅ Loaded after_norm.bias")
        }

        print("UpsampleEncoder: Weight loading complete")
    }
}

// MARK: - Flow Blocks (Decoder)

// Helper: Apply mask smartly - skip batch elements with all-zero masks (CFG unconditional)
func applyMaskSmartGlobal(_ h: MLXArray, _ mask: MLXArray?, debug: Bool = false) -> MLXArray {
    guard let m = mask else { return h }

    let B = h.shape[0]
    if debug {
        print("applyMaskSmart: h.shape=\(h.shape), mask.shape=\(m.shape), B=\(B)")
    }
    if B == 1 {
        // Single batch element - apply mask normally
        return h * m
    } else {
        // Multiple batch elements - check each one
        var result = h
        for b in 0..<B {
            let maskSum = m[b].sum().item(Float.self)
            if debug {
                eval(h[b])
                print("applyMaskSmart: batch[\(b)] maskSum=\(maskSum), h range=[\(h[b].min().item(Float.self)), \(h[b].max().item(Float.self))]")
            }
            if maskSum > 0 {
                // This batch element has a non-zero mask - apply it
                result[b] = h[b] * m[b]
                if debug {
                    eval(result[b])
                    print("applyMaskSmart: batch[\(b)] MASKED, result range=[\(result[b].min().item(Float.self)), \(result[b].max().item(Float.self))]")
                }
            } else {
                if debug {
                    print("applyMaskSmart: batch[\(b)] SKIPPED masking (unconditional pass)")
                }
            }
            // If maskSum == 0, skip masking (CFG unconditional pass)
        }
        if debug {
            eval(result)
            print("applyMaskSmart: final result range=[\(result.min().item(Float.self)), \(result.max().item(Float.self))]")
        }
        return result
    }
}

public class CausalBlock1D: Module {
    public let conv: CausalConv1d
    public let norm: LayerNorm
    
    public init(dim: Int, dimOut: Int) {
        self.conv = CausalConv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 3)
        self.norm = LayerNorm(dimensions: dimOut)
        super.init()
    }
    
    public static var debugCalls = false

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Python CausalBlock1D: output = block(x * mask); return output * mask
        var h = x

        // Multiply input by mask (Python line 61)
        if let mask = mask {
            if CausalBlock1D.debugCalls {
                eval(h)
                eval(mask)
                print("CAUSALBLOCK1D: x.shape=\(h.shape), mask.shape=\(mask.shape)")
                print("CAUSALBLOCK1D: mask sum=\(mask.sum().item(Float.self)), mask.min=\(mask.min().item(Float.self)), mask.max=\(mask.max().item(Float.self))")
                print("CAUSALBLOCK1D: h BEFORE mask: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
                fflush(stdout)
            }
            h = applyMaskSmartGlobal(h, mask)
            if CausalBlock1D.debugCalls {
                eval(h)
                print("CAUSALBLOCK1D: h AFTER mask: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
                fflush(stdout)
            }
        }

        // [B, C, L]
        h = conv(h)
        // Transpose for LayerNorm [B, L, C]
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = h.transposed(0, 2, 1)
        h = mish(h)

        // Multiply output by mask (Python line 62)
        if let mask = mask {
            h = applyMaskSmartGlobal(h, mask)
        }

        return h
    }
}

public class CausalResNetBlock: Module {
    public let block1: CausalBlock1D
    public let block2: CausalBlock1D
    public let mlpLinear: FixedLinear
    public let resConv: Conv1d  // Use regular Conv1d like Python (not CausalConv1d)

    public init(dim: Int, dimOut: Int, timeEmbDim: Int) {
        self.block1 = CausalBlock1D(dim: dim, dimOut: dimOut)
        self.block2 = CausalBlock1D(dim: dimOut, dimOut: dimOut)
        self.mlpLinear = FixedLinear(timeEmbDim, dimOut, name: "CausalResNetBlock.mlpLinear")
        // Python: self.res_conv = nn.Conv1d(dim, dim_out, 1) - regular Conv1d
        self.resConv = Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 1)
        super.init()
    }

    public static var debugEnabled: Bool = false
    private static var callCount: Int = 0

    public static func resetCallCount() {
        callCount = 0
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, timeEmb: MLXArray) -> MLXArray {
        CausalResNetBlock.callCount += 1
        let isFirstCall = CausalResNetBlock.callCount == 1

        if FlowMatchingDecoder.debugStep > 0 {
            print("🔧 ResNet call #\(CausalResNetBlock.callCount), isFirstCall=\(isFirstCall), debugStep=\(FlowMatchingDecoder.debugStep)")
        }

        var h = block1(x, mask: mask)
        if CausalResNetBlock.debugEnabled { eval(h); print("RESNET block1: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }

        // 🔍 DEBUG: After block1
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(h, name: "DEBUG_after_block1")
        }

        // Python: mlp = Sequential(Mish(), Linear()) - mish BEFORE linear!
        var tEmb = mlpLinear(mish(timeEmb))
        if CausalResNetBlock.debugEnabled { eval(tEmb); print("RESNET mlp_linear(mish(tEmb)): [\(tEmb.min().item(Float.self)), \(tEmb.max().item(Float.self))]") }
        tEmb = tEmb.expandedDimensions(axis: 2)

        // 🔍 DEBUG: Time embedding
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(tEmb, name: "DEBUG_tEmb")
        }

        h = h + tEmb
        if CausalResNetBlock.debugEnabled { eval(h); print("RESNET h + tEmb: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }

        // 🔍 MICRO-BISECTION CHECKPOINT: After Time Embedding Injection
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(h, name: "CHECKPOINT_B1_Norm1_Temb")
        }

        h = block2(h, mask: mask)  // block2 already handles masking (input AND output)
        if CausalResNetBlock.debugEnabled { eval(h); print("RESNET block2: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }

        // 🔍 MICRO-BISECTION CHECKPOINT: After Main Branch Output
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(h, name: "CHECKPOINT_B2_Branch_Output")
        }

        // Python: res_conv with transpose, same as CausalBlock1D transpose pattern
        // Python does: self.res_conv(x * mask) - simple element-wise multiplication!
        // x is [B, C, T], Conv1d expects [B, T, C]
        var xForRes = x
        if let mask = mask {
            xForRes = x * mask  // Simple multiplication, like Python!
        }

        // 🔍 DEBUG: Before resConv (x * mask)
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(xForRes, name: "DEBUG_xForRes_before_resConv")
        }

        var res = xForRes.transposed(0, 2, 1)  // [B, T, C]
        res = resConv(res)                      // [B, T, Cout]
        res = res.transposed(0, 2, 1)           // [B, Cout, T]
        if CausalResNetBlock.debugEnabled { eval(res); print("RESNET resConv: [\(res.min().item(Float.self)), \(res.max().item(Float.self))]") }

        // 🔍 MICRO-BISECTION CHECKPOINT: Skip Connection
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(res, name: "CHECKPOINT_B3_Skip")
        }

        let output = h + res

        // 🔍 MICRO-BISECTION CHECKPOINT: Final Sum
        if isFirstCall && FlowMatchingDecoder.debugStep > 0 {
            debugStats(output, name: "CHECKPOINT_B4_Final_Sum")
        }

        return output
    }
}

public class FlowTransformerBlock: Module {
    public let norm1: LayerNorm
    public let attention: MultiHeadAttention  // Weight loading maps .attn. -> .attention.
    public let norm2: LayerNorm  // Weight loading maps .norm3. -> .norm2.
    public let ff: FlowMLP

    public static var debugEnabled: Bool = false
    public static var debugBlockId: Int = -1
    public static var debugTfmrId: Int = -1

    public init(dim: Int, numHeads: Int, headDim: Int) {
        self.norm1 = LayerNorm(dimensions: dim)
        // Python DiffusersAttention: qkv_bias=False, out_bias=True
        // Note: out_proj.bias is NOT in safetensors, loaded separately from Python export
        self.attention = MultiHeadAttention(dims: dim, numHeads: numHeads, headDim: headDim, qkvBias: false, outBias: true)
        self.norm2 = LayerNorm(dimensions: dim)
        self.ff = FlowMLP(dim: dim)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        let res1 = h
        h = norm1(h)
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after norm1: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        h = attention(h, mask: mask)
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after attn: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        h = h + res1
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after res1: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        let res2 = h
        if FlowTransformerBlock.debugEnabled && FlowTransformerBlock.debugBlockId == 0 && FlowTransformerBlock.debugTfmrId == 0 {
            if let w = norm2.weight { print("TFMR[0,0] norm2 weights: \(w[0..<3].asArray(Float.self))") }
            if let b = norm2.bias { print("TFMR[0,0] norm2 bias: \(b[0..<3].asArray(Float.self))") }
            print("TFMR[0,0] res2 (norm2 input): [\(res2.min().item(Float.self)), \(res2.max().item(Float.self))]")
        }
        h = norm2(h)
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after norm2: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            FlowMLP.debugEnabled = true
        }
        h = ff(h)
        FlowMLP.debugEnabled = false
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after ff: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        h = h + res2
        if FlowTransformerBlock.debugEnabled {
            eval(h); print("TFMR[\(FlowTransformerBlock.debugBlockId),\(FlowTransformerBlock.debugTfmrId)] after res2: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
        }
        return h
    }
}

// Decoder Upsample layer using ConvTranspose1d
// Python: ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
public class DecoderUpsample1D: Module {
    let conv: ConvTransposed1d

    public init(channels: Int) {
        // Python: nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.conv = ConvTransposed1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x is [B, C, T] in Python format
        // ConvTransposed1d expects [B, T, C] in MLX
        let xTransposed = x.transposed(0, 2, 1)  // [B, T, C]
        let out = conv(xTransposed)  // [B, T*2, C]
        return out.transposed(0, 2, 1)  // [B, C, T*2]
    }
}

public class UNetBlock: Module {
    public let resnet: CausalResNetBlock
    public let transformers: [FlowTransformerBlock]
    public let downLayer: CausalConv1d?
    public let upLayer: CausalConv1d?  // Changed from DecoderUpsample1D to CausalConv1d

    public init(inChannels: Int, outChannels: Int, timeEmbDim: Int, numTransformers: Int, numHeads: Int = 4, headDim: Int = 64, isDown: Bool = false, isUp: Bool = false) {
        self.resnet = CausalResNetBlock(dim: inChannels, dimOut: outChannels, timeEmbDim: timeEmbDim)
        var tfmrs: [FlowTransformerBlock] = []
        for _ in 0..<numTransformers {
            tfmrs.append(FlowTransformerBlock(dim: outChannels, numHeads: numHeads, headDim: headDim))
        }
        self.transformers = tfmrs

        // Python decoder with channels=[256] has only 1 down block and 1 up block
        // When is_last=True, Python uses CausalConv1d(3) instead of Downsample/Upsample
        // Since we have only 1 down block, is_last=True, so NO downsampling!
        // Similarly for up block: is_last=True means CausalConv1d(3) not Upsample
        if isDown {
            // Python line 160: if is_last, use CausalConv1d(3) not Downsample1D
            // With channels=[256], there's only 1 down block, so is_last=True
            self.downLayer = CausalConv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1)
            self.upLayer = nil
        } else if isUp {
            self.downLayer = nil
            // Python line 214: if is_last, use CausalConv1d(3) not Upsample1D
            // With channels=[256], there's only 1 up block, so is_last=True
            self.upLayer = CausalConv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1)
        } else {
            self.downLayer = nil
            self.upLayer = nil
        }
        super.init()
    }
}

public class FlowMatchingDecoder: Module {
    let config: S3GenConfig
    public let timeMLP: TimeMLP
    public let downBlocks: [UNetBlock]
    public let midBlocks: [UNetBlock]
    public let upBlocks: [UNetBlock]

    public let finalBlock: CausalBlock1D
    public let finalProj: Conv1d
    public let convIn: Conv1d // Added missing initial convolution

    // Whether to use causal (streaming) or full (non-streaming) attention
    public var useFullAttention: Bool = true  // Default: full attention like Python non-streaming

    public init(config: S3GenConfig) {
        self.config = config
        self.timeMLP = TimeMLP(inputDim: config.hiddenDim + 64, embDim: config.timeEmbDim)

        let inChannels = 320  // x(80) + mu(80) + spk(80) + cond(80)
        let hiddenDim = config.hiddenDim
        let timeEmbDim = config.timeEmbDim

        // Placeholder convIn - not used in forward pass but needed for Module structure
        // The real input projection is done by downBlocks[0].resnet.block1 (320 -> 256)
        self.convIn = Conv1d(inputChannels: inChannels, outputChannels: hiddenDim, kernelSize: 1)

        // downBlocks[0] takes the raw 320-channel concatenated input
        self.downBlocks = [
            UNetBlock(inChannels: inChannels, outChannels: hiddenDim, timeEmbDim: timeEmbDim, numTransformers: 4, numHeads: config.numHeads, headDim: config.headDim, isDown: true)
        ]
        var mids: [UNetBlock] = []
        for _ in 0..<config.numMidBlocks {
            mids.append(UNetBlock(inChannels: hiddenDim, outChannels: hiddenDim, timeEmbDim: timeEmbDim, numTransformers: 4, numHeads: config.numHeads, headDim: config.headDim))
        }
        self.midBlocks = mids
        self.upBlocks = [
            UNetBlock(inChannels: hiddenDim * 2, outChannels: hiddenDim, timeEmbDim: timeEmbDim, numTransformers: 4, numHeads: config.numHeads, headDim: config.headDim, isUp: true)
        ]
        self.finalBlock = CausalBlock1D(dim: hiddenDim, dimOut: hiddenDim)
        self.finalProj = Conv1d(inputChannels: hiddenDim, outputChannels: 80, kernelSize: 1)
        super.init()
    }
    
    private func makeAttentionMask(_ L: Int, paddingMask: MLXArray? = nil) -> MLXArray {
        // Start with causal/non-causal pattern
        var biasMask: MLXArray
        if useFullAttention {
            // Full bidirectional attention - all positions attend to all
            biasMask = MLXArray.zeros([1, 1, L, L])
        } else {
            // Causal mask for streaming mode
            let indices = MLXArray(Array(0..<L).map { Int32($0) })
            let row = indices.expandedDimensions(axis: 1)
            let col = indices.expandedDimensions(axis: 0)
            // mask where col > row (future steps)
            let mask = (col .> row)
            // bias: -1e9 for future, 0 for past/present
            biasMask = MLX.where(mask, MLXArray(-1e9), MLXArray.zeros([L, L]))
            // [1, 1, L, L] for broadcasting
            biasMask = biasMask.expandedDimensions(axis: 0).expandedDimensions(axis: 1)
        }

        // Add padding mask if provided
        // paddingMask is [B, 1, T] where 0=invalid/padded, 1=valid
        if let pm = paddingMask {
            // Expand to [B, 1, 1, T] for broadcasting
            let pmExpanded = pm.expandedDimensions(axis: 2)  // [B, 1, 1, T]
            // Where padding mask is 0, set bias to -1e9
            let paddingBias = MLX.where(pmExpanded .== 0, MLXArray(-1e9), MLXArray.zeros(pmExpanded.shape))
            // Broadcast and add to existing bias
            biasMask = biasMask + paddingBias
        }

        return biasMask
    }

    public static var debugStep: Int = 0  // Set to 1 during first ODE step for tracing

    public func callAsFunction(x: MLXArray, mu: MLXArray, t: MLXArray, speakerEmb: MLXArray, cond: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let L = x.shape[2]
        let debug = FlowMatchingDecoder.debugStep == 1

        // Reset ResNet call counter for debugging
        CausalResNetBlock.resetCallCount()

        // Helper function to check spatial variation (prompt vs generated regions)
        func checkSpatial(_ h: MLXArray, label: String) {
            let T = h.shape[2]
            // The sequence length varies: 696 or 748 depending on the test
            // Find the prompt length by checking cond for zeros
            // For now, we'll handle both common cases
            if T >= 500 {
                // Assume prompt is first 500 frames (common pattern)
                let L_pm = 500
                let promptRegion = h[0..., 0..., 0..<L_pm]
                let generatedRegion = h[0..., 0..., L_pm...]
                eval(promptRegion); eval(generatedRegion)
                let promptMean = promptRegion.mean().item(Float.self)
                let generatedMean = generatedRegion.mean().item(Float.self)
                let bias = generatedMean - promptMean
                print("   \(label) SPATIAL: prompt=\(String(format: "%.4f", promptMean)), generated=\(String(format: "%.4f", generatedMean)), bias=\(String(format: "%.4f", bias))")
            }
        }

        if debug {
            print("DEC: Entered callAsFunction, x.shape = \(x.shape), mask?.shape = \(mask?.shape ?? [0])")
            eval(x); eval(mu); eval(speakerEmb); eval(cond)

            // Show each batch element separately
            let B = x.shape[0]
            for b in 0..<B {
                print("DEC: Batch[\(b)] inputs:")
                let xb = x[b]
                let mub = mu[b]
                let spkb = speakerEmb[b]
                let condb = cond[b]
                eval(xb); eval(mub); eval(spkb); eval(condb)
                print("  x:   [\(xb.min().item(Float.self)), \(xb.max().item(Float.self))], mean=\(xb.mean().item(Float.self))")
                print("  mu:  [\(mub.min().item(Float.self)), \(mub.max().item(Float.self))], mean=\(mub.mean().item(Float.self))")
                print("  spk: [\(spkb.min().item(Float.self)), \(spkb.max().item(Float.self))], mean=\(spkb.mean().item(Float.self))")
                print("  cond:[\(condb.min().item(Float.self)), \(condb.max().item(Float.self))], mean=\(condb.mean().item(Float.self))")
            }
            fflush(stdout)
            print("DEC: About to call timeMLP, t.shape = \(t.shape)")
            fflush(stdout)
        }

        TimeMLP.debugEnabled = debug
        let tEmb = timeMLP(t)
        eval(tEmb)
        if debug {
            print("DEC: timeMLP output:")
            print("  shape: \(tEmb.shape)")
            print("  tEmb[0,:5]: \(tEmb[0, 0..<5].asArray(Float.self))")
            print("  mean: \(tEmb.mean().item(Float.self))")
            fflush(stdout)
        }

        // spkEmb is 80-dim already (projected by caller)
        let spkExpanded = tiled(speakerEmb.expandedDimensions(axis: 2), repetitions: [1, 1, L])

        if debug {
            print("DEC: spkExpanded created")
            fflush(stdout)
        }

        // Matches Python order: x, mu, spks, cond
        // x(80) + mu(80) + spk(80) + cond(80) = 320
        if debug {
            print("DEC: Concatenation components (spatial bias):")
            checkSpatial(x, label: "  x")
            checkSpatial(mu, label: "  mu")
            checkSpatial(spkExpanded, label: "  spk_expanded")
            checkSpatial(cond, label: "  cond")
            fflush(stdout)
        }
        var h = concatenated([x, mu, spkExpanded, cond], axis: 1) // [B, 320, T]
        if debug {
            eval(h)
            print("DEC: h concatenated, h.shape = \(h.shape), range=[\(h.min().item(Float.self)), \(h.max().item(Float.self))], mean=\(h.mean().item(Float.self))")
            checkSpatial(h, label: "01_concat")
            // ===== MICRO-BISECTION CHECKPOINT A: After input concatenation =====
            debugStats(h, name: "CHECKPOINT_A_input_concat")
            fflush(stdout)
        }

        // Attention Mask: combines causal/non-causal pattern with padding mask
        if debug {
            print("DEC: About to call makeAttentionMask(L=\(L))")
            print("DEC: useFullAttention = \(useFullAttention)")
            fflush(stdout)
        }
        var attnMask = makeAttentionMask(L, paddingMask: mask)
        if debug {
            print("DEC: makeAttentionMask returned, shape=\(attnMask.shape)")
            eval(attnMask)
            print("DEC: attnMask[0,0,0,:5] = \(attnMask[0,0,0,0..<5].asArray(Float.self))")
            print("DEC: attnMask[0,0,5,:5] = \(attnMask[0,0,5,0..<5].asArray(Float.self))")
            fflush(stdout)
        }

        // WORKAROUND: When mask=nil, pass nil to ALL operations to avoid MLX operator cache bug
        let useMask = (mask != nil)

        if debug {
            print("DEC: useMask=\(useMask), mask?.shape = \(mask?.shape ?? [0])")
            if let m = mask {
                eval(m)
                let B = m.shape[0]
                for b in 0..<B {
                    let maskSum = m[b].sum().item(Float.self)
                    print("DEC: mask[\(b)] sum = \(maskSum)")
                }
            }
            fflush(stdout)
        }

        // Helper: Apply mask smartly - skip batch elements with all-zero masks (CFG unconditional)
        func applyMaskSmart(_ h: MLXArray, _ mask: MLXArray?) -> MLXArray {
            guard let m = mask else { return h }

            let B = h.shape[0]
            if B == 1 {
                // Single batch element - apply mask normally
                return h * m
            } else {
                // Multiple batch elements - check each one
                var result = h
                for b in 0..<B {
                    let maskSum = m[b].sum().item(Float.self)
                    if maskSum > 0 {
                        // This batch element has a non-zero mask - apply it
                        result[b] = h[b] * m[b]
                    }
                    // If maskSum == 0, skip masking (CFG unconditional pass)
                }
                return result
            }
        }

        // Only create mask tracking if mask is provided
        var masks: [MLXArray] = []
        if useMask, let providedMask = mask {
            masks = [providedMask]
            if debug {
                print("DEC: Using provided mask, shape=\(providedMask.shape)")
                fflush(stdout)
            }
        }

        let down = downBlocks[0]
        let maskDown = useMask ? masks.last : nil

        if debug {
            if let md = maskDown {
                print("DEC: maskDown.shape=\(md.shape), h.shape=\(h.shape)")
            } else {
                print("DEC: maskDown=nil, h.shape=\(h.shape)")
            }
            print("DEC: Calling down.resnet with mask=\(maskDown == nil ? "nil" : "provided")")
            fflush(stdout)
            CausalResNetBlock.debugEnabled = true
            CausalBlock1D.debugCalls = true  // Enable mask debugging
        }
        h = down.resnet(h, mask: maskDown, timeEmb: tEmb)
        if debug {
            CausalResNetBlock.debugEnabled = false
            CausalBlock1D.debugCalls = false
            eval(h)
            print("DEC: down.resnet complete, h range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "02_down_resnet")
            // ===== MICRO-BISECTION CHECKPOINT B: After first ResNet =====
            debugStats(h, name: "CHECKPOINT_B_after_first_resnet")
        }
        h = h.transposed(0, 2, 1)
        if debug {
            eval(h)
            print("DEC: Transpose to [B,T,C] complete, h.shape=\(h.shape)")
        }
        if debug {
            print("DEC transformers input h.shape: \(h.shape)")
            checkSpatial(h.transposed(0, 2, 1), label: "tfmr_input (B,C,T)")
        }

        for (ti, tfmr) in down.transformers.enumerated() {
            if debug && ti == 0 {
                print("DEC FIRST TRANSFORMER - layer by layer:")
                let h_in = h
                checkSpatial(h_in.transposed(0, 2, 1), label: "  tfmr input (B,T,C)")

                let h_normed = tfmr.norm1(h); eval(h_normed)
                checkSpatial(h_normed.transposed(0, 2, 1), label: "  after norm1")
                print("DEC tfmr.norm1: [\(h_normed.min().item(Float.self)), \(h_normed.max().item(Float.self))]")

                // Enable detailed attention debug for first transformer
                MultiHeadAttention.debugEnabled = true
                MultiHeadAttention.debugId = "down.tfmr[0]"
                let attn_out = tfmr.attention(h_normed, mask: attnMask); eval(attn_out)
                checkSpatial(attn_out.transposed(0, 2, 1), label: "  after attn")
                MultiHeadAttention.debugEnabled = false
                print("DEC tfmr.attn: [\(attn_out.min().item(Float.self)), \(attn_out.max().item(Float.self))]")

                let h_res1 = attn_out + h_in; eval(h_res1)
                checkSpatial(h_res1.transposed(0, 2, 1), label: "  after residual1")

                let h_norm2 = tfmr.norm2(h_res1); eval(h_norm2)
                checkSpatial(h_norm2.transposed(0, 2, 1), label: "  after norm2")

                let ff_out = tfmr.ff(h_norm2); eval(ff_out)
                checkSpatial(ff_out.transposed(0, 2, 1), label: "  after ff")

                let h_final = ff_out + h_res1; eval(h_final)
                checkSpatial(h_final.transposed(0, 2, 1), label: "  after residual2 (final)")
            }
            h = tfmr(h, mask: attnMask)
            if debug {
                eval(h)
                print("DEC down.tfmr[\(ti)]: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
                checkSpatial(h.transposed(0, 2, 1), label: "After tfmr[\(ti)] (B,C,T)")
            }
        }
        h = h.transposed(0, 2, 1)
        if debug {
            eval(h)
            print("DEC after down.tfmrs (transposed): [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "03_down_tfmrs")
            // ===== MICRO-BISECTION CHECKPOINT C: After first transformers =====
            debugStats(h, name: "CHECKPOINT_C_after_first_transformers")
        }
        let skip = h
        // Python line 295: x = downsample(x * mask_down)
        // NOTE: With channels=[256], Python uses CausalConv1d(3) not Downsample1D,
        // so NO actual downsampling occurs - just a causal conv with stride=1.
        // Therefore we should NOT downsample the mask either!
        if let dl = down.downLayer {
            if useMask, let md = maskDown {
                h = dl(h * md)  // Multiply by mask before processing
                // Since downLayer is CausalConv1d(stride=1), no downsampling occurs
                // So mask stays at full resolution
                masks.append(md)  // Keep mask at full resolution
            } else {
                h = dl(h)
            }
            // Since no downsampling, L stays the same - no need to recreate attnMask
            // let newL = h.shape[2]
            // attnMask = makeAttentionMask(newL)
        }

        // ===== BISECTION CHECKPOINT 1: After down_blocks =====
        if debug {
            debugStats(h, name: "CHECKPOINT_1_after_down_blocks")
        }

        // Mid blocks use downsampled mask (or nil)
        let maskMid = useMask ? masks.last : nil
        for (i, mid) in midBlocks.enumerated() {
             if debug && i == 11 { CausalResNetBlock.debugEnabled = true }
             h = mid.resnet(h, mask: maskMid, timeEmb: tEmb)
             if debug && i == 11 { CausalResNetBlock.debugEnabled = false; eval(h); print("MID[11] after resnet: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
             h = h.transposed(0, 2, 1) // [B, T, C]
             for (ti, tfmr) in mid.transformers.enumerated() {
                 // Enable detailed debug for mid block 11, transformers 2 and 3
                 if debug && i == 11 && ti >= 2 {
                     FlowTransformerBlock.debugEnabled = true
                     FlowTransformerBlock.debugBlockId = i
                     FlowTransformerBlock.debugTfmrId = ti
                 }
                 h = tfmr(h, mask: attnMask)
                 FlowTransformerBlock.debugEnabled = false
                 if debug && i == 11 { eval(h); print("MID[11] tfmr[\(ti)]: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
             }
             h = h.transposed(0, 2, 1)
             if debug {
                 eval(h)
                 print("DEC after mid[\(i)]: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
                 checkSpatial(h, label: String(format: "04_mid_%02d", i))
             }
        }

        // ===== BISECTION CHECKPOINT 2: After mid_blocks =====
        if debug {
            debugStats(h, name: "CHECKPOINT_2_after_mid_blocks")
        }

        // Python truncates h to match skip length: x[:, :, :skip.shape[-1]]
        let skipLen = skip.shape[2]
        let hTrunc = h[.ellipsis, 0..<skipLen]
        if debug {
            eval(h); eval(skip); eval(hTrunc)
            print("DEC before concat: h=[\(h.min().item(Float.self)), \(h.max().item(Float.self))], skip=[\(skip.min().item(Float.self)), \(skip.max().item(Float.self))]")
            print("DEC up concat: h=\(h.shape), skip=\(skip.shape), hTrunc=\(hTrunc.shape)")
            // Check spatial bias of h and skip separately
            checkSpatial(hTrunc, label: "05_h_before_concat")
            checkSpatial(skip, label: "05_skip_before_concat")
        }
        h = concatenated([hTrunc, skip], axis: 1)
        if debug {
            eval(h)
            print("DEC after concat: shape=\(h.shape), range=[\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "05_skip_concat")
        }

        // Up blocks: after concat with skip, h is back to full resolution - use full mask (or nil)
        let maskFull: MLXArray? = useMask ? masks.first : nil

        if debug && useMask {
            if let mf = maskFull {
                eval(mf)
                print("DEC up: maskFull.shape = \(mf.shape)")
            }
        }

        let up = upBlocks[0]
        if debug {
            CausalResNetBlock.debugEnabled = true
            CausalBlock1D.debugCalls = true  // Enable debug for up block too
        }
        h = up.resnet(h, mask: maskFull, timeEmb: tEmb)  // Use FULL mask (h is at full res after concat)
        if debug {
            CausalResNetBlock.debugEnabled = false
            CausalBlock1D.debugCalls = false
        }
        if debug {
            eval(h)
            print("DEC after up.resnet: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "06_up_resnet")
        }
        h = h.transposed(0, 2, 1)
        // Recreate attention mask for full resolution (after concat)
        let fullL = h.shape[1]
        attnMask = makeAttentionMask(fullL, paddingMask: maskFull)
        for tfmr in up.transformers { h = tfmr(h, mask: attnMask) }
        h = h.transposed(0, 2, 1)
        if debug {
            eval(h)
            print("DEC after up.transformers: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "07_up_tfmrs")
            // Check per-channel bias after up.tfmrs
            print("Channel-wise spatial bias after up.tfmrs:")
            for c in [0, 64, 128, 192, 255] {
                let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                let genMean = h[0, c, 500...].mean().item(Float.self)
                print(String(format: "  ch%3d: prompt=%.4f, gen=%.4f, bias=%.4f", c, promptMean, genMean, genMean - promptMean))
            }
        }
        // Python applies mask before upsample (only if using mask)
        if let ul = up.upLayer {
            if debug {
                // Check upLayer weights
                let ulConv = ul.conv
                let ulWeight = ulConv.weight
                eval(ulWeight)
                print("upLayer.conv.weight shape=\(ulWeight.shape), range=[\(ulWeight.min().item(Float.self)), \(ulWeight.max().item(Float.self))], mean=\(ulWeight.mean().item(Float.self))")
                if let ulBias = ulConv.bias {
                    eval(ulBias)
                    print("upLayer.conv.bias shape=\(ulBias.shape), range=[\(ulBias.min().item(Float.self)), \(ulBias.max().item(Float.self))], mean=\(ulBias.mean().item(Float.self))")
                }
            }
            if useMask, let mf = maskFull {
                if debug { print("DEC: About to call applyMaskSmart before upLayer") }
                let hMasked = applyMaskSmartGlobal(h, mf, debug: debug)
                h = ul(hMasked)
            } else {
                h = ul(h)
            }
            if debug {
                eval(h)
                print("DEC after up.upLayer: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
                checkSpatial(h, label: "07b_after_upLayer")
                // Check per-channel bias after upLayer
                print("Channel-wise spatial bias after upLayer:")
                for c in [0, 64, 128, 192, 255] {
                    let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                    let genMean = h[0, c, 500...].mean().item(Float.self)
                    print(String(format: "  ch%3d: prompt=%.4f, gen=%.4f, bias=%.4f", c, promptMean, genMean, genMean - promptMean))
                }
            }
        }

        // ===== BISECTION CHECKPOINT 3: After up_blocks =====
        if debug {
            debugStats(h, name: "CHECKPOINT_3_after_up_blocks")
        }

        // After upsampling, h is back to full resolution - use full mask (or nil)
        h = finalBlock(h, mask: maskFull) // [B, C, T] output
        if debug {
            eval(h)
            print("DEC after finalBlock: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            checkSpatial(h, label: "08_finalBlock")
            // Check per-channel bias after finalBlock
            print("Channel-wise spatial bias after finalBlock:")
            for c in [0, 64, 128, 192, 255] {
                let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                let genMean = h[0, c, 500...].mean().item(Float.self)
                print(String(format: "  ch%3d: prompt=%.4f, gen=%.4f, bias=%.4f", c, promptMean, genMean, genMean - promptMean))
            }
        }

        // Python multiplies by mask before final_proj: output = self.final_proj(x * mask_up)
        if useMask, let mf = maskFull {
            if debug {
                eval(mf)
                let maskPrompt = mf[0..., 0..., 0..<500].mean().item(Float.self)
                let maskGen = mf[0..., 0..., 500...].mean().item(Float.self)
                print("DEC mask before finalProj: prompt_mean=\(maskPrompt), gen_mean=\(maskGen)")
            }
            h = applyMaskSmart(h, mf)
            if debug { eval(h); print("DEC after mask (before finalProj): [\(h.min().item(Float.self)), \(h.max().item(Float.self))]") }
        }

        // Final Proj: MLX Conv1d expects input as [B, T, C] (channels LAST!)
        if debug {
            print("\nDEBUG finalProj:")
            print("  Input h before transpose: shape=\(h.shape) (expecting [B, C, T])")
        }
        h = h.transposed(0, 2, 1) // [B, C, T] → [B, T, C]
        if debug {
            print("  After transpose h.shape=\(h.shape) (expecting [B, T, C])")
            eval(h)
            let hBCT = h.transposed(0, 2, 1)  // Back to [B, C, T] for spatial check
            eval(hBCT)
            checkSpatial(hBCT, label: "08b_before_finalProj")
            // Check spatial for first few timesteps
            let t0_prompt_slice = h[0, 0..<10, 0...]
            let t0_gen_slice = h[0, 500..<510, 0...]
            eval(t0_prompt_slice); eval(t0_gen_slice)
            let t0_prompt = t0_prompt_slice.mean().item(Float.self)
            let t0_gen = t0_gen_slice.mean().item(Float.self)
            print(String(format: "  Time-wise check: t[0:10]=%.4f, t[500:510]=%.4f, diff=%.4f", t0_prompt, t0_gen, t0_gen - t0_prompt))
        }

        // FORENSIC: Save ODE output before finalProj
        if debug {
            let forensicDir = "../test_audio/forensic"
            let odeOutputPath = forensicDir + "/swift_ode_output.safetensors"
            eval(h)
            try? MLX.save(arrays: ["ode_output": h], url: URL(fileURLWithPath: odeOutputPath))
            print("  [FORENSIC] Saved ODE output (before finalProj): \(odeOutputPath)")
            print("  [FORENSIC] ODE output shape: \(h.shape), range: [\(h.min().item(Float.self)), \(h.max().item(Float.self))], mean: \(h.mean().item(Float.self))")
        }

        h = finalProj(h)          // [B, T, 80]

        // TODO: ROOT CAUSE INVESTIGATION NEEDED
        // Current status: Swift mel is ~1.5 dB darker than Python
        // Need forensic comparison of actual mel values (not inferred from audio)
        // See: E2E/FORENSIC_INVESTIGATION_PLAN.md

        // For now: Output RAW for forensic comparison
        if debug {
            eval(h)
            print("  [FORENSIC] After finalProj (RAW) h.shape=\(h.shape)")
            print("  [FORENSIC] After finalProj (RAW) range=[\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            print("  [FORENSIC] After finalProj (RAW) mean=\(h.mean().item(Float.self))")
            print("  [FORENSIC] Expected Python: mean≈-5.81, max=0.00")

            // TEMPORARY: Save mel for forensic comparison
            let forensicDir = "../test_audio/forensic"
            try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)
            let forensicPath = forensicDir + "/swift_mel_raw.safetensors"

            // h is [B, T, 80], transpose to [B, 80, T] for saving
            let hTransposed = h.transposed(0, 2, 1)
            eval(hTransposed)
            try? MLX.save(arrays: ["mel_gen": hTransposed], url: URL(fileURLWithPath: forensicPath))
            print("  [FORENSIC] Saved mel to: \(forensicPath)")
        }
        h = h.transposed(0, 2, 1) // [B, T, 80] → [B, 80, T]
        if debug {
            eval(h)
            print("DEC after finalProj+correction: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            print("DEC mean after correction: \(h.mean().item(Float.self))")
            checkSpatial(h, label: "09_finalProj_corrected")
        }

        // Python multiplies by mask after final_proj: return output * mask
        if useMask, let mf = maskFull {
            h = applyMaskSmart(h, mf)
        }

        if debug {
            eval(h)
            print("DEC output: [\(h.min().item(Float.self)), \(h.max().item(Float.self))]")
            // Show each batch element output
            let B = h.shape[0]
            for b in 0..<B {
                let hb = h[b]
                eval(hb)
                print("DEC Batch[\(b)] output: [\(hb.min().item(Float.self)), \(hb.max().item(Float.self))], mean=\(hb.mean().item(Float.self))")
            }

            // Check spatial variation in vCond (Batch[0]) output
            let T = h.shape[2]
            if T == 748 {
                let L_pm = 500
                let vCondPrompt = h[0, 0..., 0..<L_pm]
                let vCondGenerated = h[0, 0..., L_pm...]
                eval(vCondPrompt); eval(vCondGenerated)
                print("DEC vCond (Batch[0]) SPATIAL - prompt[0..<\(L_pm)]: [\(vCondPrompt.min().item(Float.self)), \(vCondPrompt.max().item(Float.self))], mean=\(vCondPrompt.mean().item(Float.self))")
                print("DEC vCond (Batch[0]) SPATIAL - generated[\(L_pm)...]: [\(vCondGenerated.min().item(Float.self)), \(vCondGenerated.max().item(Float.self))], mean=\(vCondGenerated.mean().item(Float.self))")
            }
        }

        return h
    }
}

// MARK: - Vocoder

public class ResBlock: Module {
    public let convs1: [Conv1d]
    public let convs2: [Conv1d]
    public let acts1: [Snake]
    public let acts2: [Snake]
    
    public init(channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        var c1: [Conv1d] = []; var c2: [Conv1d] = []
        var a1: [Snake] = []; var a2: [Snake] = []
        for d in dilations {
            let p1 = (kernelSize * d - d) / 2
            c1.append(Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize, padding: p1, dilation: d))
            let p2 = (kernelSize - 1) / 2
            c2.append(Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize, padding: p2, dilation: 1))
            a1.append(Snake(channels: channels))
            a2.append(Snake(channels: channels))
        }
        self.convs1 = c1; self.convs2 = c2
        self.acts1 = a1; self.acts2 = a2
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0..<convs1.count {
            var xt = acts1[i](h)
            xt = convs1[i](xt)
            xt = acts2[i](xt)
            xt = convs2[i](xt)
            h = h + xt
        }
        return h
    }
}

public class ConvRNNF0Predictor: Module {
    let convs: [Conv1d]
    let classifier: FixedLinear
    
    public init(inChannels: Int = 80, condChannels: Int = 512) {
        var c: [Conv1d] = []
        // condnet: 5 layers. indices 0, 2, 4, 6, 8.
        // We just store them sequentially in 'convs'.
        c.append(Conv1d(inputChannels: inChannels, outputChannels: condChannels, kernelSize: 3, padding: 1))
        for _ in 0..<4 {
            c.append(Conv1d(inputChannels: condChannels, outputChannels: condChannels, kernelSize: 3, padding: 1))
        }
        self.convs = c
        self.classifier = FixedLinear(condChannels, 1, name: "ConvRNNF0Predictor.classifier")
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, C] (channels-last, already in MLX Conv1d format)
        var h = x

        for conv in convs {
            h = elu(conv(h))
        }

        // Classifier -> [B, T, 1]
        h = classifier(h)
        // Squeeze last dim -> [B, T]
        return abs(h.squeezed(axis: -1))
    }
}

public class SineGen: Module {
    let harmonicNum: Int
    let sineAmp: Float
    let noiseStd: Float
    let samplingRate: Float
    let voicedThreshold: Float
    
    public init(samplingRate: Float, harmonicNum: Int = 8, sineAmp: Float = 0.1, noiseStd: Float = 0.003, voicedThreshold: Float = 0) {
        self.samplingRate = samplingRate
        self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.voicedThreshold = voicedThreshold
        super.init()
    }
    
    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // f0: [B, T, 1]
        let B = f0.shape[0]
        // let T = f0.shape[1] // infer from f0
        
        // Harmonic multipliers [1, 2, ..., H+1] -> Reshape to [1, 1, H]
        let multi = MLXArray(Array(1...(harmonicNum+1)).map { Float($0) }).reshaped([1, 1, -1])
        
        // F_mat: [B, T, H] via broadcast
        // f0 is [B, T, 1]
        let fMat = f0 * multi / samplingRate
        
        // Theta: 2 * pi * cumsum(fMat) % 1
        let cumSum = cumsum(fMat, axis: 1)
        let thetaMat = 2 * Float.pi * (cumSum - floor(cumSum))
        
        // Phase Vec: DETERMINISTIC (all zeros) for cross-validation
        // CRITICAL FIX: PyTorch and MLX have different RNGs - seeding doesn't match!
        // Use zero phases for deterministic behavior matching Python (when Python also uses zeros)
        let phaseVec = MLXArray.zeros([B, 1, harmonicNum + 1]) // [B, 1, H+1] -> broadcast to T
        
        // Sine Waves: [B, T, H]
        // Note: Python output uses sin(theta + phase). 
        // H = harmonicNum+1? 
        // Python: harmonic_num=8. multipliers=1..9. Output is 9 harmonics?
        // Yes, H+1 harmonics (fundamental + 8 overtones).
        
        let sineWaves = sineAmp * sin(thetaMat + phaseVec)
        
        // UV Logic
        let uv = (f0 .> voicedThreshold).asType(Float.self)
        
        // Noise - DETERMINISTIC (zeros) for cross-validation
        // CRITICAL FIX: PyTorch randn and MLX normal are different - use zeros
        let noiseAmp = uv * noiseStd + (1 - uv) * sineAmp / 3.0
        let noise = MLXArray.zeros(sineWaves.shape) // Zero noise for determinism
        
        let outWaves = sineWaves * uv + noise
        return (outWaves, uv, noise)
    }
}

public class SourceModuleHnNSF: Module {
    public let sineGen: SineGen
    public let linear: FixedLinear
    
    public init(samplingRate: Int, harmonicNum: Int = 8, sineAmp: Float = 0.1, noiseStd: Float = 0.003) {
        self.sineGen = SineGen(samplingRate: Float(samplingRate), harmonicNum: harmonicNum, sineAmp: sineAmp, noiseStd: noiseStd)
        self.linear = FixedLinear(harmonicNum + 1, 1, name: "SourceModuleHnNSF.linear") // H+1 inputs -> 1 output
        super.init()
    }
    
    public func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        // f0: [B, T, 1]
        let (sineWaves, uv, _) = sineGen(f0) // [B, T, H]
        
        // Merge harmonics -> [B, T, 1]
        // sineWaves is [B, T, H]. Linear expects [B, T, H] -> [B, T, 1].
        let sineMerge = tanh(linear(sineWaves))
        
        let noise = MLXRandom.normal(uv.shape) * (sineGen.sineAmp / 3.0)
        return (sineMerge, noise, uv)
    }
}

public class Mel2Wav: Module {
    public let convPre: Conv1d
    public let ups: [ConvTransposed1d]
    public let resblocks: [ResBlock]
    public let convPost: Conv1d

    // Source components (made public for debugging)
    public let f0Predictor: ConvRNNF0Predictor
    public let mSource: SourceModuleHnNSF
    public let sourceDowns: [Conv1d]
    public let sourceResBlocks: [ResBlock]
    
    // Constants
    public static let nFFT = 16
    public static let hopLength = 4
    public static let winSize = 16

    public let stftWindow: MLXArray

    public override init() {
        self.convPre = Conv1d(inputChannels: 80, outputChannels: 512, kernelSize: 7, stride: 1, padding: 3)
        
        let upRates = [8, 5, 3]
        let upKernels = [16, 11, 7]
        let inputCh = 512
        var currentCh = inputCh
        
        var u: [ConvTransposed1d] = []
        var upChannels: [Int] = []
        
        for i in 0..<upRates.count {
            let rate = upRates[i]
            let k = upKernels[i]
            let outCh = currentCh / 2
            // Padding: (k - stride) // 2
            let p = (k - rate) / 2
            u.append(ConvTransposed1d(inputChannels: currentCh, outputChannels: outCh, kernelSize: k, stride: rate, padding: p))
            currentCh = outCh
            upChannels.append(currentCh)
        }
        self.ups = u
        
        var r: [ResBlock] = []
        let resKernels = [3, 7, 11]
        let resDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        for ch in upChannels {
            for j in 0..<3 {
                r.append(ResBlock(channels: ch, kernelSize: resKernels[j], dilations: resDilations[j]))
            }
        }
        self.resblocks = r
        
        // F0 & Source initialization
        self.f0Predictor = ConvRNNF0Predictor() 
        self.mSource = SourceModuleHnNSF(samplingRate: 24000)
        
        // Source Downs & ResBlocks logic
        var sDowns: [Conv1d] = []
        var sRes: [ResBlock] = []
        
        // Note: sRates derivation from [8, 5, 3] -> [15, 3, 1]
        let sRates = [15, 3, 1]
        let sKernelsList = [7, 7, 11]
        let sDilationsList = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        
        for i in 0..<3 {
            let u = sRates[i]
            let ch = upChannels[i]
            let inFreq = 18 // nFFT(16) + 2
            
            if u == 1 {
                sDowns.append(Conv1d(inputChannels: inFreq, outputChannels: ch, kernelSize: 1, stride: 1))
            } else {
                sDowns.append(Conv1d(inputChannels: inFreq, outputChannels: ch, kernelSize: u * 2, stride: u, padding: u / 2))
            }
            
            sRes.append(ResBlock(channels: ch, kernelSize: sKernelsList[i], dilations: sDilationsList[i]))
        }
        self.sourceDowns = sDowns
        self.sourceResBlocks = sRes

        self.convPost = Conv1d(inputChannels: 64, outputChannels: 18, kernelSize: 7, padding: 3)
        self.stftWindow = Mel2Wav.hannWindow(Mel2Wav.nFFT)
        super.init()
    }
    
    static func hannWindow(_ size: Int) -> MLXArray {
        let n = MLXArray(Array(0..<size).map { Float($0) })
        return 0.5 * (1 - cos(2 * Float.pi * n / Float(size))) // Periodic for analysis
    }

    // Linear upsampling for 1D signals (matches PyTorch's Upsample mode='linear')
    // Smoothly interpolates between values instead of repeating them (tiled)
    private func upsampleLinear1D(_ input: MLXArray, scaleFactor: Int) -> MLXArray {
        print("🚀 Starting upsampleLinear1D")
        print("   - Input Shape: \(input.shape)")
        print("   - Scale: \(scaleFactor)")

        // input: [B, T] where T is the number of time steps
        let B = input.shape[0]
        let inputLen = input.shape[1]
        let outputLen = inputLen * scaleFactor

        print("   - Output length will be: \(outputLen)")

        // Create output indices in the range [0, outputLen)
        let outIndices = MLXArray(0..<outputLen).asType(.float32)
        print("   - Created output indices")

        // Map output indices to input coordinate space
        // For linear interpolation: src_idx = dst_idx * (src_len / dst_len)
        let srcIndices = outIndices * Float(inputLen) / Float(outputLen)
        print("   - Computed source indices")

        // Get integer and fractional parts
        let srcIndicesInt = floor(srcIndices).asType(.int32)
        let alpha = srcIndices - floor(srcIndices)  // Fractional part for interpolation
        print("   - Computed integer and fractional parts")

        // Clamp the next index to avoid going out of bounds
        let srcIndicesInt1 = minimum(srcIndicesInt + 1, MLXArray(inputLen - 1))
        print("   - Clamped indices")

        // Process each batch element
        var results: [MLXArray] = []
        print("   - Processing \(B) batch elements...")
        for b in 0..<B {
            if b == 0 {
                print("     - Processing batch 0...")
            }
            let inputSlice = input[b]  // [T]

            // Gather values at integer positions
            let val0 = inputSlice[srcIndicesInt]
            let val1 = inputSlice[srcIndicesInt1]

            // Linear interpolation: val0 + (val1 - val0) * alpha
            let interpolated = val0 + (val1 - val0) * alpha

            results.append(interpolated)
            if b == 0 {
                print("     - Batch 0 complete")
            }
        }

        print("   - Stacking results...")
        // Stack back to [B, T_upsampled]
        let result = stacked(results, axis: 0)
        print("✅ Finished upsampleLinear1D. Output shape: \(result.shape)")
        return result
    }

    // Reflection padding for 1D signal (matches PyTorch's reflection_pad1d)
    // Mirrors the signal at boundaries: [1,2,3,4,5] with pad=2 -> [3,2,1,2,3,4,5,4,3]
    private func reflectionPad1D(_ x: MLXArray, padLen: Int) -> MLXArray {
        // x: [B, T]
        let T = x.shape[1]

        // Left padding: mirror positions 1 to padLen (excluding edge at 0)
        // For [1,2,3,4,5] with pad=2: mirror [2,3] -> [3,2]
        let leftRegion = x[0..., 1...(padLen)]  // [B, padLen] starting from index 1
        let leftIndices = MLXArray(Array((0..<padLen).reversed()))  // [padLen-1, ..., 1, 0]
        let leftPad = leftRegion[0..., leftIndices]  // Reverse it

        // Right padding: mirror positions (T-padLen-1) to (T-2) (excluding edge at T-1)
        // For [1,2,3,4,5] with pad=2: T=5, mirror [3,2] (indices 3,2) -> [3,2] reversed = [2,3] reversed = [3,2]
        let rightRegion = x[0..., (T-padLen-1)..<(T-1)]  // [B, padLen] ending before last element
        let rightIndices = MLXArray(Array((0..<padLen).reversed()))
        let rightPad = rightRegion[0..., rightIndices]  // Reverse it

        // Concatenate: [leftPad | x | rightPad]
        return concatenated([leftPad, x, rightPad], axis: 1)
    }

    // STFT Helper
    public func stft(x: MLXArray, nFFT: Int, hopLength: Int, window: MLXArray) -> (MLXArray, MLXArray) {
        // x: [B, T]
        let padLen = nFFT / 2

        // CRITICAL: Use reflection padding to match PyTorch's torch.stft (center=True)
        // PyTorch reflects the signal at the boundaries
        let xPad = reflectionPad1D(x, padLen: padLen)
        
        let numFrames = (xPad.shape[1] - nFFT) / hopLength + 1
        let B = x.shape[0]
        
        // Indices for gather
        let indices = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                      MLXArray(0..<nFFT).expandedDimensions(axis: 0) // [Frames, nFFT]
        
        var framesList: [MLXArray] = []
        for b in 0..<B {
            let xb = xPad[b] // [T]
            let f = xb[indices] // [Frames, nFFT]
            framesList.append(f)
        }
        var frames = stacked(framesList, axis: 0) // [B, Frames, nFFT]
        
        frames = frames * window
        
        let spectrum = MLXFFT.rfft(frames, axis: 2) // [B, Frames, F/2+1]
        
        let real = spectrum.realPart()
        let imag = spectrum.imaginaryPart()
        
        return (real.transposed(0, 2, 1), imag.transposed(0, 2, 1))
    }

    public static var debugEnabled: Bool = false

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        if Mel2Wav.debugEnabled {
            eval(mel)
            print("🔍 VOC INPUT: mel.shape = \(mel.shape)")
            print("   Expected from decoder: [B, 80, T] (channels first)")
        }

        // mel: [B, 80, L] -> [B, L, 80]
        var x = mel.transposed(0, 2, 1)

        if Mel2Wav.debugEnabled {
            eval(x)
            print("🔍 VOC AFTER TRANSPOSE: x.shape = \(x.shape)")
            print("   Should be: [B, T, 80] (channels last for MLX Conv1d)")
        }

        // F0 Prediction
        let f0 = f0Predictor(x) // [B, T]
        if Mel2Wav.debugEnabled {
            eval(f0)
            print("VOC F0: shape=\(f0.shape), range=[\(f0.min().item(Float.self)), \(f0.max().item(Float.self))]")
            print("VOC F0 first 20: \(Array(f0[0, 0..<20].asArray(Float.self)))")
        }

        // Upsample F0
        // Scale = product of uprates * hop(4) = 8*5*3 * 4 = 120 * 4 = 480
        // CRITICAL FIX: Use LINEAR interpolation (not tiled/nearest) to match PyTorch's Upsample
        // Tiled creates stair-steps which cause phase discontinuities in the sine generator
        let f0Upsampled = upsampleLinear1D(f0, scaleFactor: 480) // [B, T * 480]
        let f0Flat = f0Upsampled.expandedDimensions(axis: 2) // [B, T_high, 1]
        if Mel2Wav.debugEnabled {
            eval(f0Flat)
            print("VOC f0Flat: shape=\(f0Flat.shape), range=[\(f0Flat.min().item(Float.self)), \(f0Flat.max().item(Float.self))]")
        }

        // Generate Source
        if Mel2Wav.debugEnabled {
            let linearW = mSource.linear.weight
            eval(linearW)
            print("VOC mSource.linear.weight: shape=\(linearW.shape), values=\(linearW.flattened().asArray(Float.self))")
            if let linearB = mSource.linear.bias {
                eval(linearB)
                print("VOC mSource.linear.bias: \(linearB.asArray(Float.self))")
            } else {
                print("VOC mSource.linear has NO bias")
            }
        }
        let (s, _, _) = mSource(f0Flat) // [B, T_high, 1]
        if Mel2Wav.debugEnabled {
            eval(s)
            print("VOC source (s): shape=\(s.shape), range=[\(s.min().item(Float.self)), \(s.max().item(Float.self))]")
            print("VOC source first 20: \(Array(s[0, 0..<20, 0].asArray(Float.self)))")
        }
        
        // Source STFT
        let (sReal, sImag) = stft(x: s.squeezed(axis: 2), nFFT: 16, hopLength: 4, window: stftWindow)
        let sSTFT = concatenated([sReal, sImag], axis: 1) // [B, F(18), Frames]
        // var si = sSTFT.transposed(0, 2, 1) // REMOVED unused variable

        if Mel2Wav.debugEnabled {
            eval(sSTFT)
            print("\n--- Step 0c: Source STFT ---")
            print("   s_stft shape: \(sSTFT.shape)")
            print("   s_stft range: [\(sSTFT.min().item(Float.self)), \(sSTFT.max().item(Float.self))]")
        }

        // CRITICAL DEBUG: Check Conv1d weight shape to detect transposition issue
        if Mel2Wav.debugEnabled {
            eval(convPre.weight)
            print("\n🔍 VOCODER convPre.weight shape: \(convPre.weight.shape)")
            print("   Expected MLX format: [out_channels(512), kernel_size(7), in_channels(80)]")
            print("   PyTorch format would be: [out_channels(512), in_channels(80), kernel_size(7)]")
        }

        // Main Path
        x = convPre(x)  // [B, L, 512]

        if Mel2Wav.debugEnabled {
            eval(x)
            print("\n--- Step 1: conv_pre ---")
            print("   Shape: \(x.shape)")
            print("   Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
            print("   Mean: \(x.mean().item(Float.self)), Std: \(x.variance().sqrt().item(Float.self))")
        }

        for i in 0..<ups.count {
            if Mel2Wav.debugEnabled {
                print("\n--- Upsampling Stage \(i) ---")
            }

            x = leakyRelu(x, negativeSlope: 0.1)

            if Mel2Wav.debugEnabled {
                print("   Before ups[\(i)]: \(x.shape)")
            }

            x = ups[i](x)  // Upsample

            if Mel2Wav.debugEnabled {
                eval(x)
                print("   After ups[\(i)]: \(x.shape)")
                print("   Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
            }

            // Source Fusion
            // si is [B, T_current, F_current]
            // We need to downsample si to match x shape
            // Warning: si starts at high res?
            // No, si is computed from high res f0 source STFT.
            // sSTFT size matches FINAL output size approx.
            // Python logic:
            // "Source fusion: ...  si = self.source_downs[i](si)"
            // si starts as the FULL STFT.
            // source_downs[0] (stride 15) downsamples it to match block 0.
            // Wait, Python loop reuses `si`?
            // "si = mx.swapaxes(s_stft, ...)"
            // It resets `si` from `s_stft` every iteration!
            // It doesn't accumulate si downsampling. It downsamples from fresh each time.
            
            var siLocal = sSTFT.transposed(0, 2, 1) // [B, T, 18]

            if Mel2Wav.debugEnabled && i == 0 {
                eval(siLocal)
                print("   🔍 DEBUG source stage \(i): After transpose: shape=\(siLocal.shape), range=[\(siLocal.min().item(Float.self)), \(siLocal.max().item(Float.self))], mean=\(siLocal.mean().item(Float.self))")
            }

            siLocal = sourceDowns[i](siLocal)

            if Mel2Wav.debugEnabled && i == 0 {
                eval(siLocal)
                print("   🔍 DEBUG source stage \(i): After sourceDowns: shape=\(siLocal.shape), range=[\(siLocal.min().item(Float.self)), \(siLocal.max().item(Float.self))], mean=\(siLocal.mean().item(Float.self))")
            }

            siLocal = sourceResBlocks[i](siLocal)

            if Mel2Wav.debugEnabled {
                eval(siLocal)
                print("   Source fusion add (after sourceResBlocks[\(i)]): mean=\(siLocal.mean().item(Float.self)), range=[\(siLocal.min().item(Float.self)), \(siLocal.max().item(Float.self))]")
            }

            // Add to x. Check shapes.
            // x might be slightly different due to padding?
            if x.shape[1] != siLocal.shape[1] {
                // Crop to min
                let minLen = min(x.shape[1], siLocal.shape[1])
                x = x[0..., 0..<minLen, 0...]
                siLocal = siLocal[0..., 0..<minLen, 0...]
            }

            x = x + siLocal

            // Apply residual blocks
            var xs: MLXArray? = nil
            for j in 0..<3 {
                 let idx = i * 3 + j
                 let resX = resblocks[idx](x)
                 if let acc = xs { xs = acc + resX } else { xs = resX }
            }
            x = xs! / 3.0

            if Mel2Wav.debugEnabled {
                eval(x)
                print("   After ResBlocks: \(x.shape)")
                print("   Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
                print("   Mean: \(x.mean().item(Float.self)), Std: \(x.variance().sqrt().item(Float.self))")
            }
        }

        // NOTE: Python uses F.leaky_relu(x) here (no slope arg), which defaults to 0.01
        // The upsampling loop uses 0.1, but this final one uses PyTorch's default of 0.01
        x = leakyRelu(x, negativeSlope: 0.01)
        x = convPost(x)  // [B, L*480, 18]

        if Mel2Wav.debugEnabled {
            eval(x)
            print("🔍 VOC conv_post output (STFT params):")
            print("   Shape: \(x.shape)")
            print("   Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
            print("   Mean: \(x.mean().item(Float.self))")

            // Check magnitude/phase before ISTFT
            let nHalf = 9
            let mag = clip(exp(x[0..., 0..., 0..<nHalf]), max: 100.0)
            let phaseSin = sin(x[0..., 0..., nHalf...])
            eval(mag)
            eval(phaseSin)
            print("🔍 VOC Magnitude (before ISTFT):")
            print("   Shape: \(mag.shape), Range: [\(mag.min().item(Float.self)), \(mag.max().item(Float.self))]")
            print("   Mean: \(mag.mean().item(Float.self))")
            print("🔍 VOC Phase/sin (before ISTFT):")
            print("   Shape: \(phaseSin.shape), Range: [\(phaseSin.min().item(Float.self)), \(phaseSin.max().item(Float.self))]")
            print("   Mean: \(phaseSin.mean().item(Float.self))")
        }

        // ISTFT and clip audio to [-0.99, 0.99] (matching Python line 794)
        let audio = istft_hifigan(x)
        return clip(audio, min: -0.99, max: 0.99)
    }

    private func istft_hifigan(_ x: MLXArray) -> MLXArray {
        // x: [B, T, 18]
        let nFFT = 16
        let nHalf = nFFT / 2 + 1 // 9

        // Clip magnitude to max 100 (matching Python istft line 478)
        let mag = clip(exp(x[0..., 0..., 0..<nHalf]), max: 100.0)
        let phaseSin = sin(x[0..., 0..., nHalf...])

        // Reconstruct Real/Imag
        // Python uses sin(output) as the phase angle
        let phase = phaseSin
        let real = mag * cos(phase)
        let imag = mag * sin(phase)

        // Conjugate symmetry
        let realMirror = real[0..., 0..., 1..<(nHalf - 1)][0..., 0..., .stride(by: -1)]
        let imagMirror = -imag[0..., 0..., 1..<(nHalf - 1)][0..., 0..., .stride(by: -1)]

        let realFull = concatenated([real, realMirror], axis: 2)
        let imagFull = concatenated([imag, imagMirror], axis: 2)

        // IFFT
        let spectrum = realFull + imagFull.asImaginary()
        let frames = MLXFFT.ifft(spectrum, axis: 2).realPart()

        // Window
        let window = stftWindow // [16]
        let framesWindowed = frames * window

        // Overlap Add using scatter/accumulate approach
        // MLX indexed assignment doesn't work like Python, so we use a different strategy:
        // Build padded frames and sum them
        let B = frames.shape[0]
        let L = frames.shape[1]
        let hop = 4
        let audioLen = (L - 1) * hop + nFFT

        // Create output by summing shifted frames
        // Each frame[i] contributes to positions [i*hop, i*hop + nFFT)
        // We can do this by padding each frame appropriately and summing

        var audioAccum: MLXArray? = nil
        for i in 0..<L {
            let start = i * hop
            let endPad = audioLen - start - nFFT
            let frame = framesWindowed[0..., i, 0...] // [B, nFFT]

            // Pad frame to full audio length: [start zeros, frame, endPad zeros]
            let paddedFrame = padded(frame, widths: [[0, 0], [start, endPad]])

            if let acc = audioAccum {
                audioAccum = acc + paddedFrame
            } else {
                audioAccum = paddedFrame
            }
        }
        var audio = audioAccum!

        // Window sum normalization
        // Pre-compute window sum pattern (it's the same for all audio)
        let winSq = window * window // [nFFT]
        var windowSumAccum: MLXArray? = nil
        for i in 0..<L {
            let start = i * hop
            let endPad = audioLen - start - nFFT

            let paddedWin = padded(winSq.expandedDimensions(axis: 0), widths: [[0, 0], [start, endPad]])

            if let acc = windowSumAccum {
                windowSumAccum = acc + paddedWin
            } else {
                windowSumAccum = paddedWin
            }
        }
        let windowSum = windowSumAccum!.squeezed(axis: 0) // [audioLen]

        audio = audio / (windowSum + 1e-8)

        // Trim padding
        let pad = nFFT / 2
        return audio[0..., pad..<(audioLen - pad)]
    }
}

// MARK: - Main S3Gen Wrapper

public class S3Gen: Module {
    public let inputEmbedding: Embedding
    public let encoder: UpsampleEncoder  // Using UpsampleEncoder (matches Python's UpsampleConformerEncoder)
    public let encoderProj: FixedLinear
    public let spkEmbedAffine: FixedLinear
    public let decoder: FlowMatchingDecoder
    public let vocoder: Mel2Wav

    // Pre-generated fixed noise for deterministic generation (matches Python)
    // Python: mx.random.seed(0); self.rand_noise = mx.random.normal((1, 80, 50 * 300))
    public var fixedNoise: MLXArray

    // Helper function to remap Python's UNetBlock structure to Swift's
    // Python: {down|mid|up}Blocks.X.0.* -> Swift: {down|mid|up}Blocks.X.resnet.*
    // Python: {down|mid|up}Blocks.X.1.Y.* -> Swift: {down|mid|up}Blocks.X.transformers.Y.*
    // Python: {down|mid|up}Blocks.X.2.* -> Swift: {down|mid|up}Blocks.X.{down|up}Layer.*
    private static func remapUNetBlockKeys(_ key: String) -> String {
        var result = key

        // Match pattern: (downBlocks|midBlocks|upBlocks).(\d+).0.
        // Replace with: $1.$2.resnet.
        let blockTypes = ["downBlocks", "midBlocks", "upBlocks"]
        for blockType in blockTypes {
            // Pattern: blockType.X.0. -> blockType.X.resnet.
            result = result.replacingOccurrences(
                of: "\(blockType).",
                with: "%%%\(blockType).",
                options: []
            )
            // Now replace .0. with .resnet. (only for UNet blocks)
            if result.contains("%%%\(blockType).") {
                // Find instances like %%%blockType.0.0. and replace with %%%blockType.0.resnet.
                let pattern = "%%%\(blockType)\\.(\\d+)\\.0\\."
                if let regex = try? NSRegularExpression(pattern: pattern) {
                    let range = NSRange(result.startIndex..., in: result)
                    result = regex.stringByReplacingMatches(
                        in: result,
                        range: range,
                        withTemplate: "%%%\(blockType).$1.resnet."
                    )
                }
                // Find instances like %%%blockType.0.1.Y. and replace with %%%blockType.0.transformers.Y.
                let tfmrPattern = "%%%\(blockType)\\.(\\d+)\\.1\\.(\\d+)\\."
                if let regex = try? NSRegularExpression(pattern: tfmrPattern) {
                    let range = NSRange(result.startIndex..., in: result)
                    result = regex.stringByReplacingMatches(
                        in: result,
                        range: range,
                        withTemplate: "%%%\(blockType).$1.transformers.$2."
                    )
                }
                // Find instances like %%%blockType.0.2. and replace with appropriate layer
                let layerPattern = "%%%\(blockType)\\.(\\d+)\\.2\\."
                if let regex = try? NSRegularExpression(pattern: layerPattern) {
                    let range = NSRange(result.startIndex..., in: result)
                    let layerName = blockType == "downBlocks" ? "downLayer" : "upLayer"
                    result = regex.stringByReplacingMatches(
                        in: result,
                        range: range,
                        withTemplate: "%%%\(blockType).$1.\(layerName)."
                    )
                }
            }
        }

        // CRITICAL: Remap transformer attention keys
        // Python BasicTransformerBlock uses .attn1. but Swift uses .attention.
        result = result.replacingOccurrences(of: ".attn1.to_q.", with: ".attention.queryProj.")
        result = result.replacingOccurrences(of: ".attn1.to_k.", with: ".attention.keyProj.")
        result = result.replacingOccurrences(of: ".attn1.to_v.", with: ".attention.valueProj.")
        result = result.replacingOccurrences(of: ".attn1.to_out.0.", with: ".attention.outProj.")

        // CRITICAL: Remap transformer feedforward (FlowMLP) keys
        // Python: .ff.net.0.proj. (first linear with GELU activation layer wrapped)
        // Python: .ff.net.2. (second linear, index 2 because GELU is at index 1)
        // Swift: .ff.layers.0. and .ff.layers.1.
        result = result.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.layers.0.")
        result = result.replacingOccurrences(of: ".ff.net.2.", with: ".ff.layers.1.")

        // Python uses norm3 for feedforward normalization, Swift uses norm2
        result = result.replacingOccurrences(of: ".norm3.", with: ".norm2.")

        // Remove temporary markers
        result = result.replacingOccurrences(of: "%%%", with: "")
        return result
    }

    public init(flowWeights: [String: MLXArray], vocoderWeights: [String: MLXArray]?) {
        let config = S3GenConfig()
        self.inputEmbedding = Embedding(embeddingCount: config.vocabSize, dimensions: config.inputDim)

        // DEBUG: Check what keys are in flowWeights
        print("DEBUG S3Gen.init: flowWeights has \(flowWeights.count) total keys"); fflush(stdout)

        // Print first 50 keys to see what we have
        let allKeys = Array(flowWeights.keys.sorted().prefix(50))
        print("DEBUG S3Gen.init: First 50 keys:"); fflush(stdout)
        for key in allKeys {
            print("  \(key)"); fflush(stdout)
        }

        let encoderRelatedKeys = flowWeights.keys.filter { $0.contains("encoder") }.prefix(30).sorted()
        print("DEBUG S3Gen.init: Encoder-related keys (\(encoderRelatedKeys.count)):"); fflush(stdout)
        for key in encoderRelatedKeys {
            print("  \(key)"); fflush(stdout)
        }

        // Load input_embedding weights
        for (key, value) in flowWeights {
            if key.hasPrefix("s3gen.flow.input_embedding.") {
                let remappedKey = key.replacingOccurrences(of: "s3gen.flow.input_embedding.", with: "")
                if remappedKey == "weight" {
                    inputEmbedding.update(parameters: ModuleParameters.unflattened(["weight": value]))
                    print("DEBUG S3Gen: Loaded inputEmbedding.weight")
                }
            } else if key.hasPrefix("flow.input_embedding.") {
                let remappedKey = key.replacingOccurrences(of: "flow.input_embedding.", with: "")
                if remappedKey == "weight" {
                    inputEmbedding.update(parameters: ModuleParameters.unflattened(["weight": value]))
                    print("DEBUG S3Gen: Loaded inputEmbedding.weight")
                }
            }
        }

        // Remap encoder weights: s3gen.flow.encoder.* → encoder.*
        // UpsampleEncoder expects "encoder.*" prefix
        // ALSO: Python uses .embed.out.0. for linear and .embed.out.1. for norm
        //       Swift expects .embed.linear. and .embed.norm.
        var encoderWeights: [String: MLXArray] = [:]
        for (key, value) in flowWeights {
            var remappedKey = key
            if key.hasPrefix("s3gen.flow.encoder.") {
                remappedKey = key.replacingOccurrences(of: "s3gen.flow.encoder.", with: "encoder.")
            } else if key.hasPrefix("flow.encoder.") {
                remappedKey = key.replacingOccurrences(of: "flow.encoder.", with: "encoder.")
            } else {
                continue  // Not an encoder key
            }

            // CRITICAL: Remap .embed.out.0. → .embedLinear. and .embed.out.1. → .embedNorm.
            // Python's Sequential([Linear, LayerNorm]) uses indices 0 and 1
            // MUST use camelCase to match Swift property names!
            remappedKey = remappedKey.replacingOccurrences(of: ".embed.out.0.", with: ".embedLinear.")
            remappedKey = remappedKey.replacingOccurrences(of: ".embed.out.1.", with: ".embedNorm.")

            // Same for up_embed
            remappedKey = remappedKey.replacingOccurrences(of: ".up_embed.out.0.", with: ".upEmbedLinear.")
            remappedKey = remappedKey.replacingOccurrences(of: ".up_embed.out.1.", with: ".upEmbedNorm.")

            encoderWeights[remappedKey] = value
        }

        print("DEBUG S3Gen: Creating UpsampleEncoder with \(encoderWeights.count) weights"); fflush(stdout)

        // Check what embed-related keys exist after remapping
        let embedKeys = encoderWeights.keys.filter { $0.contains("embed") }.sorted()
        print("DEBUG S3Gen: Embed-related keys (\(embedKeys.count)):"); fflush(stdout)
        for key in embedKeys.prefix(20) {
            if let arr = encoderWeights[key] {
                print("  \(key): shape=\(arr.shape)"); fflush(stdout)
            }
        }

        // Check specifically for norm and pos_enc keys
        let normKeys = encoderWeights.keys.filter { $0.contains("norm") && $0.contains("embed") }.sorted()
        print("DEBUG S3Gen: Norm keys for embed (\(normKeys.count)): \(normKeys)"); fflush(stdout)

        let posEncKeys = encoderWeights.keys.filter { $0.contains("pos_enc.pe") }
        print("DEBUG S3Gen: Position encoding keys (\(posEncKeys.count)): \(posEncKeys.sorted())"); fflush(stdout)
        for key in posEncKeys {
            if let pe = encoderWeights[key] {
                print("  \(key): shape=\(pe.shape)"); fflush(stdout)
            }
        }
        // Create UpsampleEncoder with weights for PreLookaheadLayer
        self.encoder = UpsampleEncoder(inputDim: config.inputDim, outputDim: config.inputDim, weights: encoderWeights)
        print("DEBUG S3Gen: UpsampleEncoder created, about to call .load()"); fflush(stdout)
        // Load all other weights using the load() method
        self.encoder.load(weights: encoderWeights, prefix: "encoder")
        print("DEBUG S3Gen: encoder.load() completed"); fflush(stdout)

        print("DEBUG S3Gen: Creating encoderProj FixedLinear(\(config.inputDim), \(config.melChannels))..."); fflush(stdout)
        self.encoderProj = FixedLinear(config.inputDim, config.melChannels, name: "S3Gen.encoderProj")
        print("DEBUG S3Gen: encoderProj created"); fflush(stdout)
        // NOTE: encoderProj weights are loaded later in ChatterboxEngine via update()
        // DO NOT load weights here to avoid double-transpose bug
        // Weights will be transposed once in remapS3Keys() and applied via update()
        print("DEBUG S3Gen: encoderProj will be loaded via ChatterboxEngine.update()"); fflush(stdout)

        print("DEBUG S3Gen: Creating spkEmbedAffine FixedLinear(192, \(config.melChannels))..."); fflush(stdout)
        self.spkEmbedAffine = FixedLinear(192, config.melChannels, name: "S3Gen.spkEmbedAffine")
        print("DEBUG S3Gen: spkEmbedAffine created"); fflush(stdout)

        // NOTE: spkEmbedAffine weights are loaded later in ChatterboxEngine via update()
        // DO NOT load weights here to avoid double-transpose bug
        // Weights will be transposed once in remapS3Keys() and applied via update()
        print("DEBUG S3Gen: spkEmbedAffine will be loaded via ChatterboxEngine.update()"); fflush(stdout)

        print("DEBUG S3Gen: Creating FlowMatchingDecoder..."); fflush(stdout)
        self.decoder = FlowMatchingDecoder(config: config)
        print("DEBUG S3Gen: FlowMatchingDecoder created"); fflush(stdout)
        print("DEBUG S3Gen: Creating Mel2Wav vocoder..."); fflush(stdout)
        self.vocoder = Mel2Wav()
        print("DEBUG S3Gen: Mel2Wav vocoder created"); fflush(stdout)
        // Generate fixed noise once at init (matching Python's CausalConditionalCFM)
        print("DEBUG S3Gen: Generating fixed noise..."); fflush(stdout)
        MLXRandom.seed(0)
        self.fixedNoise = MLXRandom.normal([1, 80, 50 * 300])
        print("DEBUG S3Gen: Fixed noise generated"); fflush(stdout)
        print("DEBUG S3Gen: Calling super.init()..."); fflush(stdout)
        super.init()
        print("DEBUG S3Gen: super.init() complete"); fflush(stdout)

        // Load decoder weights with comprehensive key remapping
        print("DEBUG S3Gen: Loading decoder weights..."); fflush(stdout)
        var decoderWeights: [String: MLXArray] = [:]
        var transposedLinearCount = 0
        var transposedConv1dCount = 0
        for (key, value) in flowWeights {
            if key.contains("decoder.estimator") {
                // Remap: flow.decoder.estimator.* -> (no prefix, decoder is implied)
                var remappedKey = key
                    .replacingOccurrences(of: "flow.decoder.estimator.", with: "")
                    .replacingOccurrences(of: "s3gen.flow.decoder.estimator.", with: "")

                // Convert Python snake_case to Swift camelCase
                remappedKey = remappedKey
                    .replacingOccurrences(of: "time_mlp.linear_1", with: "timeMLP.linear1")
                    .replacingOccurrences(of: "time_mlp.linear_2", with: "timeMLP.linear2")
                    .replacingOccurrences(of: "down_blocks", with: "downBlocks")
                    .replacingOccurrences(of: "mid_blocks", with: "midBlocks")
                    .replacingOccurrences(of: "up_blocks", with: "upBlocks")
                    .replacingOccurrences(of: "final_block", with: "finalBlock")
                    .replacingOccurrences(of: "final_proj", with: "finalProj")

                // Remap UNetBlock structure (down_blocks.X.0 -> downBlocks.X.resnet, etc.)
                remappedKey = S3Gen.remapUNetBlockKeys(remappedKey)

                // CRITICAL FIX: Remap ResNet's internal MLP
                // Python: .resnet.mlp.1. (Sequential(Mish, Linear)[1] is the Linear)
                // Swift: .resnet.mlpLinear.
                remappedKey = remappedKey.replacingOccurrences(of: ".mlp.1.", with: ".mlpLinear.")

                // CRITICAL FIX: Remap ResNet's res_conv
                // Python: .resnet.res_conv.
                // Swift: .resnet.resConv.
                remappedKey = remappedKey.replacingOccurrences(of: ".res_conv.", with: ".resConv.")

                // Remap CausalBlock1D internal structure:
                // Python: .block.0. (Conv1d) -> Swift: .conv.conv.
                // Python: .block.2. (GroupNorm) -> Swift: .norm.
                remappedKey = remappedKey
                    .replacingOccurrences(of: ".block.0.", with: ".conv.conv.")
                    .replacingOccurrences(of: ".block.2.", with: ".norm.")

                // Remap CausalConv1d structure (downLayer/upLayer are CausalConv1d):
                // Python: downLayer.weight -> Swift: downLayer.conv.weight
                remappedKey = remappedKey
                    .replacingOccurrences(of: "downLayer.weight", with: "downLayer.conv.weight")
                    .replacingOccurrences(of: "downLayer.bias", with: "downLayer.conv.bias")
                    .replacingOccurrences(of: "upLayer.weight", with: "upLayer.conv.weight")
                    .replacingOccurrences(of: "upLayer.bias", with: "upLayer.conv.bias")

                var finalW = value

                // Transpose Linear weights from PyTorch [out, in] to MLX [in, out]
                let isLinear = remappedKey.hasSuffix(".weight") && finalW.ndim == 2 &&
                              !remappedKey.contains(".conv.") && !remappedKey.contains("norm.")
                if isLinear {
                    finalW = finalW.transposed()
                    transposedLinearCount += 1
                }

                // Transpose Conv1d weights from PyTorch [out, in, kernel] to MLX [out, kernel, in]
                let isConv1d = remappedKey.hasSuffix(".weight") && finalW.ndim == 3 &&
                              !remappedKey.contains("norm") && !remappedKey.contains("embedding")
                if isConv1d {
                    finalW = finalW.transposed(0, 2, 1)
                    transposedConv1dCount += 1
                }

                decoderWeights[remappedKey] = finalW
            }
        }
        print("DEBUG S3Gen: Found \(decoderWeights.count) decoder weight keys"); fflush(stdout)
        print("DEBUG S3Gen: Transposed \(transposedLinearCount) Linear weights, \(transposedConv1dCount) Conv1d weights"); fflush(stdout)
        if decoderWeights.count > 0 {
            let sampleKeys = Array(decoderWeights.keys.sorted().prefix(10))
            print("DEBUG S3Gen: Sample decoder keys (first 10):"); fflush(stdout)
            for k in sampleKeys {
                print("  \(k)"); fflush(stdout)
            }

            // Check if timeMLP keys are present
            let timeMlpKeys = decoderWeights.keys.filter { $0.contains("timeMLP") }.sorted()
            print("DEBUG S3Gen: timeMLP keys (\(timeMlpKeys.count)): \(timeMlpKeys)"); fflush(stdout)

            // Also print what keys the decoder Module expects
            let decoderParams = self.decoder.parameters().flattened()
            let allDecoderParamKeys = decoderParams.map { $0.0 }.sorted()
            print("DEBUG S3Gen: Decoder module has \(allDecoderParamKeys.count) parameter keys"); fflush(stdout)
            let decoderParamKeys = Array(allDecoderParamKeys.prefix(10))
            print("DEBUG S3Gen: Decoder module keys (first 10):"); fflush(stdout)
            for k in decoderParamKeys {
                print("  \(k)"); fflush(stdout)
            }

            // Check if timeMLP keys exist in module
            let moduleTimeMlpKeys = allDecoderParamKeys.filter { $0.contains("timeMLP") }
            print("DEBUG S3Gen: Module timeMLP keys (\(moduleTimeMlpKeys.count)): \(moduleTimeMlpKeys)"); fflush(stdout)

            // Check if transformer keys exist in decoderWeights (what we're loading)
            let tfmrKeysInWeights = decoderWeights.keys.filter { $0.contains("transformers") }.sorted()
            print("DEBUG S3Gen: Transformer keys in decoderWeights (\(tfmrKeysInWeights.count)):"); fflush(stdout)
            for k in Array(tfmrKeysInWeights.prefix(10)) {
                print("  \(k)"); fflush(stdout)
            }

            // Check if transformer keys exist in module
            let moduleTfmrKeys = allDecoderParamKeys.filter { $0.contains("transformers") }
            print("DEBUG S3Gen: Module transformer keys (\(moduleTfmrKeys.count)):"); fflush(stdout)
            for k in Array(moduleTfmrKeys.sorted().prefix(10)) {
                print("  \(k)"); fflush(stdout)
            }

            // Check a weight value before update
            eval(self.decoder.timeMLP.linear1.weight)
            let timeMlpWeightBefore = self.decoder.timeMLP.linear1.weight[0, 0..<5].asArray(Float.self)
            print("DEBUG S3Gen: timeMLP.linear1.weight[0,:5] BEFORE update: \(timeMlpWeightBefore)"); fflush(stdout)

            self.decoder.update(parameters: ModuleParameters.unflattened(decoderWeights))

            // Check the same weight after update
            eval(self.decoder.timeMLP.linear1.weight)
            let timeMlpWeightAfter = self.decoder.timeMLP.linear1.weight[0, 0..<5].asArray(Float.self)
            print("DEBUG S3Gen: timeMLP.linear1.weight[0,:5] AFTER update: \(timeMlpWeightAfter)"); fflush(stdout)

            // Check if they're different
            let changed = timeMlpWeightBefore != timeMlpWeightAfter

            // CRITICAL: Check transformer attention weights to verify loading
            let tfmr0 = self.decoder.downBlocks[0].transformers[0]
            eval(tfmr0.attention.queryProj.weight)
            let qWeight = tfmr0.attention.queryProj.weight
            print("DEBUG S3Gen: downBlocks[0].transformers[0].attention.queryProj.weight shape: \(qWeight.shape)"); fflush(stdout)
            print("DEBUG S3Gen: downBlocks[0].transformers[0].attention.queryProj.weight[0,:5]: \(qWeight[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            print("DEBUG S3Gen: downBlocks[0].transformers[0].attention.queryProj.weight range: [\(qWeight.min().item(Float.self)), \(qWeight.max().item(Float.self))]"); fflush(stdout)
            print("DEBUG S3Gen: Weights changed: \(changed)"); fflush(stdout)

            // Check ResNet MLP weights (THE SMOKING GUN)
            print("DEBUG S3Gen: About to check ResNet MLP weights..."); fflush(stdout)
            let resnet0 = self.decoder.downBlocks[0].resnet
            print("DEBUG S3Gen: Got resnet0"); fflush(stdout)
            eval(resnet0.mlpLinear.weight)
            print("DEBUG S3Gen: Evaluated mlpLinear.weight"); fflush(stdout)
            let mlpW = resnet0.mlpLinear.weight
            print("DEBUG S3Gen: 🔍 ResNet[0] MLP Weight Shape: \(mlpW.shape) (should be [1024, 256])"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 ResNet[0] MLP weight[0,:5]: \(mlpW[0, 0..<5].asArray(Float.self))"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 ResNet[0] MLP weight[:5,0]: \(mlpW[0..<5, 0].asArray(Float.self))"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 Expected [0,:5] = [0.0115, 0.0138, -0.0350, 0.0076, -0.0255]"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 Expected [:5,0] = [0.0115, 0.0098, -0.0121, -0.0090, 0.0022]"); fflush(stdout)

            // Check ResNet resConv weights
            print("DEBUG S3Gen: Checking ResNet resConv weights..."); fflush(stdout)
            eval(resnet0.resConv.weight)
            let resConvW = resnet0.resConv.weight
            print("DEBUG S3Gen: 🔍 ResNet[0] resConv Weight Shape: \(resConvW.shape)"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 ResNet[0] resConv weight[0,0,:]: \(resConvW[0, 0, 0..<min(5, resConvW.shape[2])].asArray(Float.self))"); fflush(stdout)

            print("DEBUG S3Gen: Decoder weights loaded via decoder.update()"); fflush(stdout)
        } else {
            print("⚠️ WARNING: No decoder weights found in flowWeights!"); fflush(stdout)
        }

        // =======================================================================================
        // LOAD VOCODER WEIGHTS
        // =======================================================================================
        if let vocoderWeights = vocoderWeights {
            print("DEBUG S3Gen: Loading vocoder weights..."); fflush(stdout)

            var processedVocoderWeights: [String: MLXArray] = [:]
            var transposedConv1dCount = 0
            var transposedConvTCount = 0
            var transposedLinearCount = 0
            var weightNormCount = 0

            // Group weight_norm parametrizations by base key
            var weightNormGroups: [String: (original0: MLXArray?, original1: MLXArray?)] = [:]
            var weightNormProcessedKeys: Set<String> = []  // Track which keys came from weight_norm

            for (key, value) in vocoderWeights {
                guard key.contains("mel2wav") else { continue }

                // Handle weight_norm parametrizations
                if key.contains(".parametrizations.weight.original") {
                    // Extract base key: mel2wav.conv_pre.parametrizations.weight.original0 -> mel2wav.conv_pre
                    let baseKey = key.replacingOccurrences(of: ".parametrizations.weight.original0", with: "")
                                      .replacingOccurrences(of: ".parametrizations.weight.original1", with: "")

                    if weightNormGroups[baseKey] == nil {
                        weightNormGroups[baseKey] = (nil, nil)
                    }

                    if key.contains(".original0") {
                        weightNormGroups[baseKey]!.original0 = value
                    } else if key.contains(".original1") {
                        weightNormGroups[baseKey]!.original1 = value
                    }
                    continue
                }

                // Remap non-parametrized keys
                var remappedKey = key
                    .replacingOccurrences(of: "s3gen.mel2wav.", with: "")
                    .replacingOccurrences(of: "mel2wav.", with: "")
                    .replacingOccurrences(of: "conv_pre", with: "convPre")
                    .replacingOccurrences(of: "conv_post", with: "convPost")
                    .replacingOccurrences(of: "f0_predictor", with: "f0Predictor")
                    .replacingOccurrences(of: "m_source.l_linear", with: "mSource.linear")
                    .replacingOccurrences(of: "source_downs", with: "sourceDowns")
                    .replacingOccurrences(of: "source_resblocks", with: "sourceResBlocks")
                    .replacingOccurrences(of: "activations1", with: "acts1")
                    .replacingOccurrences(of: "activations2", with: "acts2")

                // Map f0_predictor.condnet indices: Python uses 0,2,4,6,8 -> Swift uses 0,1,2,3,4
                if remappedKey.contains("f0Predictor.condnet.") {
                    remappedKey = remappedKey
                        .replacingOccurrences(of: "condnet.0.", with: "convs.0.")
                        .replacingOccurrences(of: "condnet.2.", with: "convs.1.")
                        .replacingOccurrences(of: "condnet.4.", with: "convs.2.")
                        .replacingOccurrences(of: "condnet.6.", with: "convs.3.")
                        .replacingOccurrences(of: "condnet.8.", with: "convs.4.")
                }

                processedVocoderWeights[remappedKey] = value
            }

            // Process weight_norm groups: combine original0 and original1
            for (baseKey, pair) in weightNormGroups {
                guard let original0 = pair.original0, let original1 = pair.original1 else {
                    print("⚠️ WARNING: Incomplete weight_norm pair for \(baseKey)"); fflush(stdout)
                    continue
                }

                // Combine: weight = original0 * (original1 / ||original1||)
                // original0: [Out, 1, 1] (magnitude scale)
                // original1: [Out, In, Kernel] (direction)
                let norm = sqrt((original1 * original1).sum(axes: [1, 2], keepDims: true))
                let normalized = original1 / (norm + 1e-8)
                let combined = original0 * normalized

                // Remap key
                var remappedKey = baseKey
                    .replacingOccurrences(of: "s3gen.mel2wav.", with: "")
                    .replacingOccurrences(of: "mel2wav.", with: "")
                    .replacingOccurrences(of: "conv_pre", with: "convPre")
                    .replacingOccurrences(of: "conv_post", with: "convPost")
                    .replacingOccurrences(of: "f0_predictor", with: "f0Predictor")
                    .replacingOccurrences(of: "source_downs", with: "sourceDowns")
                    .replacingOccurrences(of: "source_resblocks", with: "sourceResBlocks")
                    .replacingOccurrences(of: "activations1", with: "acts1")
                    .replacingOccurrences(of: "activations2", with: "acts2")

                // Map f0_predictor.condnet indices
                if remappedKey.contains("f0Predictor.condnet.") {
                    remappedKey = remappedKey
                        .replacingOccurrences(of: "condnet.0", with: "convs.0")
                        .replacingOccurrences(of: "condnet.2", with: "convs.1")
                        .replacingOccurrences(of: "condnet.4", with: "convs.2")
                        .replacingOccurrences(of: "condnet.6", with: "convs.3")
                        .replacingOccurrences(of: "condnet.8", with: "convs.4")
                }

                var finalWeight = combined

                // Transpose Conv1d: PyTorch [Out, In, Kernel] -> MLX [Out, Kernel, In]
                if finalWeight.ndim == 3 && !remappedKey.contains("ups.") {
                    let before = finalWeight.shape
                    finalWeight = finalWeight.transposed(0, 2, 1)
                    let after = finalWeight.shape
                    if remappedKey.contains("convPre") || remappedKey.contains("f0Predictor") {
                        print("DEBUG: Transposed weight_norm Conv1d \(remappedKey): \(before) -> \(after)"); fflush(stdout)
                    }
                    transposedConv1dCount += 1
                }

                // Transpose ConvTranspose1d: PyTorch [In, Out, Kernel] -> MLX [Out, Kernel, In]
                if remappedKey.contains("ups.") && finalWeight.ndim == 3 {
                    finalWeight = finalWeight.transposed(1, 2, 0)
                    transposedConvTCount += 1
                }

                let finalKey = remappedKey + ".weight"
                processedVocoderWeights[finalKey] = finalWeight
                weightNormProcessedKeys.insert(finalKey)  // Track that this came from weight_norm
                weightNormCount += 1
            }

            // Transpose non-weight_norm weights
            for (key, value) in processedVocoderWeights {
                guard key.hasSuffix(".weight") else { continue }

                // Skip if already processed via weight_norm
                if weightNormProcessedKeys.contains(key) {
                    continue
                }

                var finalW = value

                // Transpose Linear weights: PyTorch [Out, In] -> MLX [In, Out]
                let isLinear = finalW.ndim == 2 && !key.contains("conv")
                if isLinear {
                    finalW = finalW.transposed()
                    transposedLinearCount += 1
                    processedVocoderWeights[key] = finalW
                }

                // Transpose Conv1d weights
                let isConv1d = finalW.ndim == 3 && !key.contains("ups.")
                if isConv1d {
                    finalW = finalW.transposed(0, 2, 1)
                    transposedConv1dCount += 1
                    processedVocoderWeights[key] = finalW
                }

                // Transpose ConvTranspose1d weights
                let isConvT = key.contains("ups.") && finalW.ndim == 3
                if isConvT {
                    finalW = finalW.transposed(1, 2, 0)
                    transposedConvTCount += 1
                    processedVocoderWeights[key] = finalW
                }
            }

            print("DEBUG S3Gen: Vocoder weights processed:"); fflush(stdout)
            print("  Total keys: \(processedVocoderWeights.count)"); fflush(stdout)
            print("  Weight norm combined: \(weightNormCount)"); fflush(stdout)
            print("  Conv1d transposed: \(transposedConv1dCount)"); fflush(stdout)
            print("  ConvTranspose1d transposed: \(transposedConvTCount)"); fflush(stdout)
            print("  Linear transposed: \(transposedLinearCount)"); fflush(stdout)

            // Apply weights to vocoder
            self.vocoder.update(parameters: ModuleParameters.unflattened(processedVocoderWeights))
            print("DEBUG S3Gen: Vocoder weights loaded via vocoder.update()"); fflush(stdout)

            // Verify weights actually loaded by checking a known weight
            print("DEBUG S3Gen: Verifying vocoder weights..."); fflush(stdout)
            let convPreW = self.vocoder.convPre.weight
            eval(convPreW)
            print("DEBUG S3Gen: 🔍 Vocoder convPre.weight shape: \(convPreW.shape)"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 Vocoder convPre.weight[0,0,:5]: \(convPreW[0, 0, 0..<5].asArray(Float.self))"); fflush(stdout)

            let mSourceW = self.vocoder.mSource.linear.weight
            eval(mSourceW)
            print("DEBUG S3Gen: 🔍 Vocoder mSource.linear.weight shape: \(mSourceW.shape)"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 Vocoder mSource.linear.weight values: \(mSourceW.flattened().asArray(Float.self))"); fflush(stdout)
            print("DEBUG S3Gen:    Expected: [-0.0012, -0.0003, -0.0004, 0.0011, 0.0014, 0.0018, -0.0004, -0.0010, -0.0019]"); fflush(stdout)

            // Verify sourceResBlocks weights
            let srcRB0C1W = self.vocoder.sourceResBlocks[0].convs1[0].weight
            eval(srcRB0C1W)
            print("DEBUG S3Gen: 🔍 sourceResBlocks[0].convs1[0].weight shape: \(srcRB0C1W.shape)"); fflush(stdout)
            print("DEBUG S3Gen: 🔍 sourceResBlocks[0].convs1[0].weight[0,0,:5]: \(srcRB0C1W[0, 0, 0..<5].asArray(Float.self))"); fflush(stdout)
        } else {
            print("⚠️ WARNING: No vocoder weights provided!"); fflush(stdout)
        }

        print("DEBUG S3Gen: S3Gen.init COMPLETE"); fflush(stdout)
    }

    // Legacy support - DEPRECATED: Use init(flowWeights:vocoderWeights:) instead
    // This init doesn't load weights properly and will produce garbage output
    /* public override init() {
        let config = S3GenConfig()
        self.inputEmbedding = Embedding(embeddingCount: config.vocabSize, dimensions: config.inputDim)
        self.encoder = FlowEncoder(hiddenDim: config.inputDim, melDim: config.melChannels, weights: [:])
        self.encoderProj = Linear(config.inputDim, config.melChannels)
        self.spkEmbedAffine = Linear(192, config.melChannels)
        self.decoder = FlowMatchingDecoder(config: config)
        self.vocoder = Mel2Wav()
        // Generate fixed noise once at init (matching Python's CausalConditionalCFM)
        MLXRandom.seed(0)
        self.fixedNoise = MLXRandom.normal([1, 80, 50 * 300])
        super.init()
    } */

    /// Replace fixed noise with externally loaded noise (e.g., from Python)
    /// This is needed because Swift MLX and Python MLX have different RNG implementations
    public func setFixedNoise(_ noise: MLXArray) {
        self.fixedNoise = noise
    }

    /// Load fixed noise from a safetensors file (generated by Python)
    public func loadFixedNoise(from url: URL) throws {
        let arrays = try MLX.loadArrays(url: url)
        guard let noise = arrays["fixed_noise"] else {
            throw NSError(domain: "S3Gen", code: 1, userInfo: [NSLocalizedDescriptionKey: "fixed_noise not found in safetensors"])
        }
        self.fixedNoise = noise
        print("Loaded Python fixed noise: shape=\(noise.shape), range=[\(noise.min().item(Float.self)), \(noise.max().item(Float.self))]")
    }
    
    public func generate(tokens: MLXArray, speakerEmb: MLXArray, speechEmbMatrix: MLXArray, promptToken: MLXArray, promptFeat: MLXArray) -> MLXArray {
        print("DEBUG S3Gen: generate() called"); fflush(stdout)
        // tokens: [1, T] generated speech tokens
        // promptToken: [1, P] history tokens
        // promptFeat: [1, T, 80] history mels (channels-last)

        // 1. Prepare Embedding (use speechEmbMatrix, not speakerEmb!)
        // speechEmbMatrix is [1, 192] and gets projected to [1, 80] via spkEmbedAffine
        // speakerEmb [1, 256] is used separately as "finalize" parameter for the decoder
        print("DEBUG S3Gen: Preparing speech embedding matrix..."); fflush(stdout)
        var speechEmb = speechEmbMatrix  // [1, 192]
        let norm = sqrt(sum(speechEmb * speechEmb, axis: 1, keepDims: true)) + 1e-8
        speechEmb = speechEmb / norm
        print("DEBUG S3Gen: Calling spkEmbedAffine on speechEmbMatrix [1, 192]..."); fflush(stdout)
        // Project speech embedding [1, 192] -> [1, 80]
        let spkCond = matmul(speechEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias! // [1, 80]
        eval(spkCond)
        let spkFlat = spkCond.flattened(); eval(spkFlat)
        let spkFirst10 = Array(spkFlat.asArray(Float.self).prefix(10))
        print("📊 Swift speech embedding (after projection to spkCond):")
        print("   Shape: \(spkCond.shape)")
        print("   Range: [\(spkCond.min().item(Float.self)), \(spkCond.max().item(Float.self))]")
        print("   Mean: \(spkCond.mean().item(Float.self))")
        print("   First 10: \(spkFirst10)")
        print("DEBUG S3Gen: spkCond created"); fflush(stdout)

        // 2. Concat prompt tokens + new tokens
        print("DEBUG S3Gen: Concatenating tokens..."); fflush(stdout)
        let inputs = concatenated([promptToken, tokens], axis: 1)
        print("DEBUG S3Gen: inputs shape: \(inputs.shape)"); fflush(stdout)

        // 3. Embed (clip tokens to valid range like Python)
        print("DEBUG S3Gen: Embedding tokens..."); fflush(stdout)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)
        print("DEBUG S3Gen: Calling inputEmbedding..."); fflush(stdout)
        let x = inputEmbedding(clippedInputs) // [1, T_total, 512]
        print("DEBUG S3Gen: inputEmbedding returned, x.shape = \(x.shape)"); fflush(stdout)
        print("DEBUG S3Gen: About to eval(x)..."); fflush(stdout)
        eval(x)
        print("DEBUG S3Gen: eval(x) completed"); fflush(stdout)
        print("DEBUG S3Gen: About to get x.min()..."); fflush(stdout)
        let xMin = x.min()
        print("DEBUG S3Gen: About to eval xMin..."); fflush(stdout)
        eval(xMin)
        print("DEBUG S3Gen: About to call xMin.item()..."); fflush(stdout)
        let xMinVal = xMin.item(Float.self)
        print("DEBUG S3Gen: xMin = \(xMinVal)"); fflush(stdout)
        print("TRACE 1: token_emb shape=\(x.shape), dtype=\(x.dtype), range=[\(xMinVal), \(x.max().item(Float.self))], mean=\(x.mean().item(Float.self))"); fflush(stdout)

        // 4. Encode (with sequence length for proper masking)
        print("DEBUG S3Gen: Creating seqLen array..."); fflush(stdout)
        print("DEBUG S3Gen: About to call encoder(x)..."); fflush(stdout)
        // UpsampleEncoder does encode + 2x upsample, then we project
        let h = encoder(x) // [1, 2*T_total, 512]
        print("DEBUG S3Gen: encoder returned, h.shape = \(h.shape)"); fflush(stdout)
        eval(h)
        print("DEBUG S3Gen: Calling encoderProj to project 512 → 80..."); fflush(stdout)
        let mu = encoderProj(h) // [1, 2*T_total, 80]
        print("DEBUG S3Gen: encoderProj returned, mu.shape = \(mu.shape)"); fflush(stdout)
        eval(mu)
        print("TRACE 2: encoder_out (mu) shape=\(mu.shape), range=[\(mu.min().item(Float.self)), \(mu.max().item(Float.self))], mean=\(mu.mean().item(Float.self))"); fflush(stdout)

        // FORENSIC: Save encoder output (mu after projection) - ALWAYS SAVE
        do {
            let forensicDir = "../test_audio/forensic"
            try? FileManager.default.createDirectory(atPath: forensicDir, withIntermediateDirectories: true)
            let encoderOutputPath = forensicDir + "/swift_encoder_output.safetensors"
            try? MLX.save(arrays: ["encoder_output": mu], url: URL(fileURLWithPath: encoderOutputPath))
            print("  [FORENSIC] Saved encoder output (mu): \(encoderOutputPath)")
            print("  [FORENSIC] Encoder shape: \(mu.shape), range: [\(mu.min().item(Float.self)), \(mu.max().item(Float.self))], mean: \(mu.mean().item(Float.self))")
        }

        // DEBUG: Check mu for prompt vs generated regions
        let muPromptRegion = mu[0, 0..<500, 0...]
        let muGeneratedRegion = mu[0, 500..., 0...]
        eval(muPromptRegion); eval(muGeneratedRegion)
        print("DEBUG: mu prompt region [0, 0..<500] range=[\(muPromptRegion.min().item(Float.self)), \(muPromptRegion.max().item(Float.self))], mean=\(muPromptRegion.mean().item(Float.self))")
        print("DEBUG: mu generated region [0, 500...] range=[\(muGeneratedRegion.min().item(Float.self)), \(muGeneratedRegion.max().item(Float.self))], mean=\(muGeneratedRegion.mean().item(Float.self))")

        // 6. Cut prompt from mu to get "generated" target length
        let promptMel = promptFeat // Already [1, T, 80]
        let L_pm = promptMel.shape[1]
        eval(promptMel)
        print("🔍 PROMPT CHECK: promptMel shape=\(promptMel.shape), range=[\(promptMel.min().item(Float.self)), \(promptMel.max().item(Float.self))], mean=\(promptMel.mean().item(Float.self))")

        // conds is a hybrid: Ground Truth prompt mels + zeros for new mels
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        let conds = concatenated([promptMel, muZeros], axis: 1) // [1, L_total, 80]
        eval(conds)
        print("TRACE 4: L_pm=\(L_pm), L_new=\(L_new), conds shape=\(conds.shape), range=[\(conds.min().item(Float.self)), \(conds.max().item(Float.self))]")

        // 7. Decode (Flow Matching) with CFG
        let L_total = conds.shape[1]

        // Use pre-generated fixed noise (matches Python's CausalConditionalCFM)
        var xt = fixedNoise[0..., 0..., 0..<L_total]
        eval(xt)
        print("TRACE 5: noise shape=\(xt.shape), range=[\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")

        // WEIGHT VERIFICATION
        let timeMlpW = decoder.timeMLP.linear1.weight
        eval(timeMlpW)
        let tmMin = timeMlpW.min().item(Float.self)
        let tmMax = timeMlpW.max().item(Float.self)
        print("WEIGHT: timeMLP.linear1.weight \(timeMlpW.shape) range=[\(tmMin), \(tmMax)]")

        // Check attention weights
        let attn = decoder.downBlocks[0].transformers[0].attention
        let qW = attn.queryProj.weight; eval(qW)
        print("WEIGHT: attn.queryProj.weight \(qW.shape) range=[\(qW.min().item(Float.self)), \(qW.max().item(Float.self))], sum=\(qW.sum().item(Float.self))")
        // Check if bias exists
        if let qB = attn.queryProj.bias { eval(qB); print("WEIGHT: queryProj has bias: \(qB.shape)") }
        else { print("WEIGHT: queryProj has NO bias") }
        if let oB = attn.outProj.bias { eval(oB); print("WEIGHT: outProj has bias: \(oB.shape)") }
        else { print("WEIGHT: outProj has NO bias") }

        // Check block1.conv and res_conv weights
        let resnet0 = decoder.downBlocks[0].resnet
        let block1ConvW = resnet0.block1.conv.conv.weight; eval(block1ConvW)
        print("WEIGHT: block1.conv.weight shape=\(block1ConvW.shape), range=[\(block1ConvW.min().item(Float.self)), \(block1ConvW.max().item(Float.self))], sum=\(block1ConvW.sum().item(Float.self))")
        let block1Flat = block1ConvW.flattened(); eval(block1Flat)
        print("WEIGHT: First 10 block1.conv.weight: \(Array(block1Flat.asArray(Float.self).prefix(10)))")

        let resConvW = resnet0.resConv.weight; eval(resConvW)
        print("WEIGHT: res_conv.weight shape=\(resConvW.shape), range=[\(resConvW.min().item(Float.self)), \(resConvW.max().item(Float.self))], sum=\(resConvW.sum().item(Float.self))")
        let resConvFlat = resConvW.flattened(); eval(resConvFlat)
        print("WEIGHT: First 10 res_conv.weight: \(Array(resConvFlat.asArray(Float.self).prefix(10)))")

        // Check mid block 11 transformer 3 weights
        if decoder.midBlocks.count > 11 {
            let mb11 = decoder.midBlocks[11]
            if mb11.transformers.count > 3 {
                let tfmr3 = mb11.transformers[3]
                let qW11 = tfmr3.attention.queryProj.weight; eval(qW11)
                let ff0W = tfmr3.ff.layers[0].weight; eval(ff0W)
                let ff1W = tfmr3.ff.layers[1].weight; eval(ff1W)
                print("WEIGHT: midBlocks[11].tfmr[3].attn.queryProj.weight \(qW11.shape) range=[\(qW11.min().item(Float.self)), \(qW11.max().item(Float.self))], sum=\(qW11.sum().item(Float.self))")
                print("WEIGHT: midBlocks[11].tfmr[3].ff.layers[0].weight \(ff0W.shape) range=[\(ff0W.min().item(Float.self)), \(ff0W.max().item(Float.self))], sum=\(ff0W.sum().item(Float.self))")
                print("WEIGHT: midBlocks[11].tfmr[3].ff.layers[1].weight \(ff1W.shape) range=[\(ff1W.min().item(Float.self)), \(ff1W.max().item(Float.self))], sum=\(ff1W.sum().item(Float.self))")
            }
        }

        // Check finalProj weights
        let fpW = decoder.finalProj.weight; eval(fpW)
        print("WEIGHT: finalProj.weight \(fpW.shape) range=[\(fpW.min().item(Float.self)), \(fpW.max().item(Float.self))], mean=\(fpW.mean().item(Float.self))")
        if let fpB = decoder.finalProj.bias {
            eval(fpB)
            print("WEIGHT: finalProj.bias \(fpB.shape) range=[\(fpB.min().item(Float.self)), \(fpB.max().item(Float.self))], mean=\(fpB.mean().item(Float.self))")
        }
        print("WEIGHT: Expected finalProj.weight: shape=[80,1,256], range≈[-0.76,0.87], mean≈0.001")
        print("WEIGHT: Expected finalProj.bias: shape=[80], range≈[-0.10,-0.04], mean≈-0.074")

        // Transpose conds and mu for decoder [B, C, T]
        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)

        // Create mask matching Python: [B, 1, T]
        // Python: mask = (~make_pad_mask(h_lengths)).unsqueeze(1)
        // This creates mask=1 for ALL valid positions, mask=0 only for padding
        // Since we have no padding, mask should be all 1s
        // The generation conditioning is done through `cond` parameter, not mask!
        let mask = MLXArray.ones([1, 1, L_total], dtype: muT.dtype)
        eval(mask)
        print("   Mask: shape=\(mask.shape), all_ones (no padding), sum=\(mask.sum().item(Float.self))")

        // ODE Parameters (Python uses n_timesteps=10 by default in flow.py)
        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG default (was 0.7)

        // Cosine time scheduling
        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        // Euler ODE solver
        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        print("\n📊 Swift ODE Configuration:")
        print("   n_timesteps: \(nTimesteps)")
        print("   CFG rate: \(cfgRate)")
        print("   t_scheduler: cosine")
        print("   Timesteps: \(tSpan)")
        print("   Initial noise: [\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")
        print("\n🔄 Starting Swift ODE loop...")

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])
            FlowMatchingDecoder.debugStep = (step == 1) ? 1 : 0

            // CFG: Conditional and unconditional passes
            // Python flow_matching.py line 128: mask_in[:B] = mask_in[B:] = mask
            // BOTH use SAME mask! Unconditional behavior comes from zero mu/spks/cond only.

            // Prepare batch for CFG: [Cond, Uncond]
            let xIn = concatenated([xt, xt], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)  // Both use same mask (match Python!)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)  // Uncond mu=0
            let spkIn = concatenated([spkCond, MLXArray.zeros(like: spkCond)], axis: 0)  // Uncond spk=0
            let condIn = concatenated([condsT, MLXArray.zeros(like: condsT)], axis: 0)  // Uncond cond=0
            let tIn = concatenated([t, t], axis: 0)

            // Forward pass (Batch=2)
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskIn)
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)

            // CFG formula: v = (1 + cfg) * vCond - cfg * vUncond
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Debug CFG components for first step
            if step == 1 || step == 5 || step == 10 {
                eval(vCond); eval(vUncond); eval(v)
                print("   CFG DEBUG (step \(step)):")
                print("     vCond:   [\(vCond.min().item(Float.self)), \(vCond.max().item(Float.self))], mean=\(vCond.mean().item(Float.self))")
                print("     vUncond: [\(vUncond.min().item(Float.self)), \(vUncond.max().item(Float.self))], mean=\(vUncond.mean().item(Float.self))")
                print("     CFG formula: (1.7) * vCond - (0.7) * vUncond")

                 // Check spatial variation in velocity field to trace accumulation
                // v shape is [1, 80, T] where T is time dimension at index 2
                let L_total = v.shape[2]
                if L_total == 748 {
                    let L_pm = 500
                    let vPromptRegion = v[0..., 0..., 0..<L_pm]
                    let vGeneratedRegion = v[0..., 0..., L_pm...]
                    eval(vPromptRegion); eval(vGeneratedRegion)
                    print("     v SPATIAL (step \(step)) - prompt[0..<\(L_pm)]: [\(vPromptRegion.min().item(Float.self)), \(vPromptRegion.max().item(Float.self))], mean=\(vPromptRegion.mean().item(Float.self))")
                    print("     v SPATIAL (step \(step)) - generated[\(L_pm)...]: [\(vGeneratedRegion.min().item(Float.self)), \(vGeneratedRegion.max().item(Float.self))], mean=\(vGeneratedRegion.mean().item(Float.self))")
                }
            }

            // Detailed tracing for all steps
            eval(v); eval(xt)
            print("\n--- Swift ODE Step \(step)/\(nTimesteps) ---")
            print("   t = \(String(format: "%.6f", currentT)), dt = \(String(format: "%.6f", dt))")
            print("   x before: [\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")
            print("   dphi_dt: [\(v.min().item(Float.self)), \(v.max().item(Float.self))]")

            // Euler step
            xt = xt + v * dt
            eval(xt)
            print("   x after: [\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")

            // Check spatial variation in xt at key steps
            if step == 1 || step == 5 || step == 10 {
                let L_total = xt.shape[2]
                if L_total == 748 {
                    let L_pm = 500
                    let xtPromptRegion = xt[0..., 0..., 0..<L_pm]
                    let xtGeneratedRegion = xt[0..., 0..., L_pm...]
                    eval(xtPromptRegion); eval(xtGeneratedRegion)
                    print("   xt SPATIAL (after step \(step)) - prompt[0..<\(L_pm)]: [\(xtPromptRegion.min().item(Float.self)), \(xtPromptRegion.max().item(Float.self))], mean=\(xtPromptRegion.mean().item(Float.self))")
                    print("   xt SPATIAL (after step \(step)) - generated[\(L_pm)...]: [\(xtGeneratedRegion.min().item(Float.self)), \(xtGeneratedRegion.max().item(Float.self))], mean=\(xtGeneratedRegion.mean().item(Float.self))")
                }
            }

            // Update time for next step
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        // 8. Vocode
        eval(xt)
        print("TRACE 6: ODE output xt shape=\(xt.shape), dtype=\(xt.dtype), range=[\(xt.min().item(Float.self)), \(xt.max().item(Float.self))], mean=\(xt.mean().item(Float.self))")

        // DEBUG: Check prompt region vs generated region
        let promptRegion = xt[0..., 0..., 0..<L_pm]
        eval(promptRegion)
        print("DEBUG: prompt region [0..., 0..., 0..<\(L_pm)] range=[\(promptRegion.min().item(Float.self)), \(promptRegion.max().item(Float.self))], mean=\(promptRegion.mean().item(Float.self))")

        let generatedMel = xt[0..., 0..., L_pm...]
        eval(generatedMel)
        print("TRACE 7: generatedMel [0..., 0..., \(L_pm)...] shape=\(generatedMel.shape), range=[\(generatedMel.min().item(Float.self)), \(generatedMel.max().item(Float.self))], mean=\(generatedMel.mean().item(Float.self))")

        // Debug: Check raw mel from flow decoder
        print("MEL CHANNEL ENERGIES (decoder output):")
        for i in [0, 10, 20, 30, 40, 50, 60, 70, 79] {
            let channelEnergy = generatedMel[0, i, 0...].mean().item(Float.self)
            print(String(format: "  Channel %2d: %.4f", i, channelEnergy))
        }

        // CRITICAL: Pass decoder mel output DIRECTLY to vocoder (match Python)
        // Python does: output_mels, _ = self.flow.inference(...)
        //              output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, ...)
        // NO transformation is applied between decoder and vocoder!
        let wav = vocoder(generatedMel)
        eval(wav)
        print("TRACE 8: vocoder output shape=\(wav.shape), range=[\(wav.min().item(Float.self)), \(wav.max().item(Float.self))]")
        return wav
    }

    // Get raw mel from encoder + flow decoder (before vocoder transformation)
    public func getEncoderAndFlowOutput(
        tokens: MLXArray,
        speakerEmb: MLXArray,
        speechEmbMatrix: MLXArray,
        promptToken: MLXArray,
        promptFeat: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Same setup as generate() but returns raw mel before vocoder

        // 1. Speech embedding normalization (use speechEmbMatrix [1, 192], not speakerEmb [1, 256]!)
        var speechEmb = speechEmbMatrix
        let norm = sqrt(sum(speechEmb * speechEmb, axis: 1, keepDims: true)) + 1e-8
        speechEmb = speechEmb / norm
        let spkCond = matmul(speechEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias!

        // 2. Prepare inputs
        let inputs = concatenated([promptToken, tokens], axis: 1)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)

        // 3. Token embedding
        let x = inputEmbedding(clippedInputs)

        // 4. Encode
        let h = encoder(x)
        let mu = encoderProj(h)
        eval(mu)  // Force evaluation to avoid deferred computation overhead

        // 5. Prepare conditions for flow decoder
        let promptMel = promptFeat
        let L_pm = promptMel.shape[1]
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        let conds = concatenated([promptMel, muZeros], axis: 1)

        // 6. Flow decoder - prepare inputs
        let L_total = conds.shape[1]
        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)
        let mask = MLXArray.ones([1, 1, L_total])

        // ODE Parameters
        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG

        // Cosine time scheduling
        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        // Initialize noise and ODE solver
        var xt = fixedNoise[0..., 0..., 0..<L_total]
        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        // ODE solver loop
        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // CFG: Conditional and unconditional passes
            // Python flow_matching.py line 128: mask_in[:B] = mask_in[B:] = mask
            // BOTH use SAME mask! Unconditional behavior comes from zero mu/spks/cond only.

            // Prepare Batch for CFG: [Cond, Uncond]
            let xIn = concatenated([xt, xt], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)  // Both use same mask (match Python!)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)  // Uncond mu=0
            let spkIn = concatenated([spkCond, MLXArray.zeros(like: spkCond)], axis: 0)  // Uncond spk=0
            let condIn = concatenated([condsT, MLXArray.zeros(like: condsT)], axis: 0)  // Uncond cond=0
            let tIn = concatenated([t, t], axis: 0)

            // Forward pass (Batch=2)
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskIn)

            // Split and apply CFG
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Euler step
            xt = xt + v * dt

            // Update time
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        return (mu, xt)
    }

    // Get encoder output for comparison
    public func getEncoderOutput(tokens: MLXArray, speechEmbMatrix: MLXArray, promptToken: MLXArray) -> (MLXArray, MLXArray) {
        let inputs = concatenated([promptToken, tokens], axis: 1)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)
        let x = inputEmbedding(clippedInputs)
        // Encoder outputs [1, T, 512], need to project to [1, T, 80]
        let h = encoder(x)
        let mu = encoderProj(h)
        eval(x); eval(mu)
        return (mu, mu)  // Return mu twice for compatibility (h is no longer separate)
    }

    // Version that uses Python's encoder_proj directly (for debugging)
    public func generateWithPythonMu(mu: MLXArray, speakerEmb: MLXArray, promptFeat: MLXArray) -> MLXArray {
        // mu: [1, T, 80] from Python's encoder
        var spkEmb = speakerEmb
        let norm = sqrt(sum(spkEmb * spkEmb, axis: 1, keepDims: true)) + 1e-8
        spkEmb = spkEmb / norm
        // WORKAROUND: Manual matmul since Linear.update() doesn't persist transpose
        let spkCond = matmul(spkEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias!

        let promptMel = promptFeat
        let L_pm = promptMel.shape[1]
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        let conds = concatenated([promptMel, muZeros], axis: 1)
        let L_total = conds.shape[1]

        var xt = fixedNoise[0..., 0..., 0..<L_total]
        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)

        // Create mask (all 1s for valid positions, same for conditional and unconditional)
        let mask = MLXArray.ones([1, 1, L_total], dtype: muT.dtype)

        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG

        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            let xIn = concatenated([xt, xt], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
            let spkIn = concatenated([spkCond, MLXArray.zeros(like: spkCond)], axis: 0)
            let condIn = concatenated([condsT, MLXArray.zeros(like: condsT)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskIn)
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            xt = xt + v * dt
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        eval(xt)
        print("  🔍 getEncoderAndFlowOutput: xt.shape before return: \(xt.shape)")
        // Return FULL mel (including prompt) for testing
        // In production, you might want to slice: xt[0..., 0..., L_pm...] to get only new frames
        let generatedMel = xt  // Return full 696 frames
        eval(generatedMel)
        print("  🔍 getEncoderAndFlowOutput: returning mel.shape: \(generatedMel.shape)")
        return generatedMel
    }

    /// Run ODE solver with all inputs provided directly (for isolated testing)
    /// All inputs should be in [B, C, T] format (channels second, time last)
    public func runODEOnly(
        initialNoise: MLXArray,  // [1, 80, T]
        muT: MLXArray,           // [1, 80, T]
        condT: MLXArray,         // [1, 80, T]
        spkEmb: MLXArray         // [1, 80] - already projected
    ) -> MLXArray {
        var xt = initialNoise

        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG

        // Cosine time scheduling
        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        print("\n📊 ODE-Only Configuration:")
        print("   n_timesteps: \(nTimesteps)")
        print("   CFG rate: \(cfgRate)")
        print("   t_span: \(tSpan)")
        print("   Initial noise: [\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // Prepare Batch for CFG: [Cond, Uncond]
            let xIn = concatenated([xt, xt], axis: 0)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
            let spkIn = concatenated([spkEmb, MLXArray.zeros(like: spkEmb)], axis: 0)
            let condIn = concatenated([condT, MLXArray.zeros(like: condT)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            // Forward pass (Batch=2)
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: nil)

            // Split
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)

            // CFG Formula: v = (1 + cfg) * vCond - cfg * vUncond
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Euler step
            xt = xt + v * dt

            // Update time for next step
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }

            if step == 1 || step == nTimesteps {
                eval(xt)
                print("  Step \(step): t=\(String(format: "%.4f", currentT-dt)), x=[\(xt.min().item(Float.self)), \(xt.max().item(Float.self))]")
            }
        }

        eval(xt)
        return xt
    }

    // Version that generates with full ODE tracing (saves to swift_ode_trace.safetensors)
    public func generateWithTracing(tokens: MLXArray, speakerEmb: MLXArray, speechEmbMatrix: MLXArray, promptToken: MLXArray, promptFeat: MLXArray, tracePath: String = "swift_ode_trace.safetensors") throws -> (MLXArray, MLXArray) {
        var traceData: [String: MLXArray] = [:]

        var spkEmb = speakerEmb
        let norm = sqrt(sum(spkEmb * spkEmb, axis: 1, keepDims: true)) + 1e-8
        spkEmb = spkEmb / norm
        // WORKAROUND: Manual matmul since Linear.update() doesn't persist transpose
        let spkCond = matmul(spkEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias!

        // Save speaker conditioning
        traceData["spk_cond"] = spkCond

        let inputs = concatenated([promptToken, tokens], axis: 1)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)
        let x = inputEmbedding(clippedInputs)
        eval(x)
        print("\n=== Swift Encoder Tracing ===")
        print("1. Concatenated tokens: \(inputs.shape)")
        print("2. Token embedding: \(x.shape)")
        print("   Range: [\(x.min().item(Float.self)), \(x.max().item(Float.self))]")
        print("   Mean: \(x.mean().item(Float.self))")

        // Encoder outputs [1, T, 512], need to project to [1, T, 80]
        let h = encoder(x)
        let mu = encoderProj(h)
        eval(mu)
        print("3. Encoder output (mu): \(mu.shape)")
        print("   Range: [\(mu.min().item(Float.self)), \(mu.max().item(Float.self))]")
        print("   Mean: \(mu.mean().item(Float.self))")
        print("=== End Swift Encoder Tracing ===\n")

        let promptMel = promptFeat
        let L_pm = promptMel.shape[1]
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        let conds = concatenated([promptMel, muZeros], axis: 1)
        let L_total = conds.shape[1]

        var xt = fixedNoise[0..., 0..., 0..<L_total]
        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)

        // Save initial state
        traceData["mu_t"] = muT
        traceData["conds_t"] = condsT
        traceData["initial_noise"] = xt

        print("=== Swift ODE Trace ===\n")
        print("Saving ODE traces to \(tracePath)...")

        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG

        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // Save xt BEFORE this step
            eval(xt)
            traceData["xt_before_step\(step)"] = xt.asType(.float32)

            let xIn = concatenated([xt, xt], axis: 0)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
            let spkIn = concatenated([spkCond, MLXArray.zeros(like: spkCond)], axis: 0)
            let condIn = concatenated([condsT, MLXArray.zeros(like: condsT)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn)

            // Save raw v_batch (before CFG)
            eval(vBatch)
            traceData["v_batch_step\(step)"] = vBatch.asType(.float32)

            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Save v after CFG
            eval(v)
            traceData["v_cfg_step\(step)"] = v.asType(.float32)

            xt = xt + v * dt

            // Save xt AFTER this step
            eval(xt)
            traceData["xt_after_step\(step)"] = xt.asType(.float32)

            print("Step \(step): t=\(String(format: "%.4f", currentT)) dt=\(String(format: "%.4f", dt))")
            print("  v_batch: range=[\(vBatch.min().item(Float.self)), \(vBatch.max().item(Float.self))], mean=\(vBatch.mean().item(Float.self))")
            print("  v (CFG): range=[\(v.min().item(Float.self)), \(v.max().item(Float.self))], mean=\(v.mean().item(Float.self))")
            print("  xt_after: range=[\(xt.min().item(Float.self)), \(xt.max().item(Float.self))], mean=\(xt.mean().item(Float.self))")

            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        eval(xt)
        let generatedMel = xt[0..., 0..., L_pm...]
        eval(generatedMel)

        // Save final mel
        traceData["final_mel"] = generatedMel.asType(.float32)

        print("\nFinal mel: shape=\(generatedMel.shape), range=[\(generatedMel.min().item(Float.self)), \(generatedMel.max().item(Float.self))], mean=\(generatedMel.mean().item(Float.self))")

        // Save all traces to file
        print("\n--- Saving Trace ---")
        try MLX.save(arrays: traceData, url: URL(fileURLWithPath: tracePath))
        print("Saved \(traceData.count) tensors to \(tracePath)")
        print("\nSaved tensors:")
        for key in traceData.keys.sorted() {
            let arr = traceData[key]!
            eval(arr)
            print("  \(key): \(arr.shape)")
        }

        let wav = vocoder(generatedMel)
        eval(wav)
        return (wav, generatedMel)
    }

    // Version that also returns the mel for debugging
    public func generateWithMel(tokens: MLXArray, speakerEmb: MLXArray, speechEmbMatrix: MLXArray, promptToken: MLXArray, promptFeat: MLXArray) -> (MLXArray, MLXArray) {
        // Call the main generate logic but capture mel
        var spkEmb = speakerEmb
        let norm = sqrt(sum(spkEmb * spkEmb, axis: 1, keepDims: true)) + 1e-8
        spkEmb = spkEmb / norm
        // WORKAROUND: Manual matmul since Linear.update() doesn't persist transpose
        let spkCond = matmul(spkEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias!

        let inputs = concatenated([promptToken, tokens], axis: 1)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)
        let x = inputEmbedding(clippedInputs)
        // Encoder outputs [1, T, 512], need to project to [1, T, 80]
        var h = encoder(x)
        print("DEBUG generateWithMel: encoder returned h.shape = \(h.shape)")

        // CRITICAL: Force CPU roundtrip to break MLX computation graph
        // This severs the link between encoder's 317-length mask and decoder's 634-length operations
        print("DEBUG generateWithMel: Breaking computation graph with CPU roundtrip...")
        let hData = h.asArray(Float.self)  // Download to CPU
        let hShape = h.shape
        h = MLXArray(hData, hShape)  // Re-upload as fresh tensor with no graph history
        print("DEBUG generateWithMel: h graph severed, shape=\(h.shape)")

        var mu = encoderProj(h)
        print("DEBUG generateWithMel: encoderProj returned mu.shape = \(mu.shape)")

        // Force CPU roundtrip for mu as well
        let muData = mu.asArray(Float.self)  // Download to CPU
        let muShape = mu.shape
        mu = MLXArray(muData, muShape)  // Re-upload as fresh tensor
        print("DEBUG generateWithMel: mu graph severed, shape=\(mu.shape)")

        let promptMel = promptFeat
        let L_pm = promptMel.shape[1]
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        var conds = concatenated([promptMel, muZeros], axis: 1)
        let L_total = conds.shape[1]

        // Force CPU roundtrip for conds to break any remaining graph links
        let condsData = conds.asArray(Float.self)
        let condsShape = conds.shape
        conds = MLXArray(condsData, condsShape)
        print("DEBUG generateWithMel: conds graph severed, shape=\(conds.shape)")

        print("DEBUG generateWithMel: mu.shape = \(mu.shape), L_pm = \(L_pm), L_new = \(L_new), L_total = \(L_total)")

        var xt = fixedNoise[0..., 0..., 0..<L_total]

        // Break graph for xt as well
        let xtData = xt.asArray(Float.self)
        let xtShape = xt.shape
        xt = MLXArray(xtData, xtShape)

        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)

        // Create identity mask with explicit broadcast to break MLX graph caching bug
        // The encoder creates a 260-length mask that persists in MLX's computation graph
        // Explicit broadcast forces MLX to create a new tensor, breaking the cached reference
        let baseMask = MLXArray.ones([1, 1, L_total], dtype: .float32)
        eval(baseMask)  // Force evaluation of base mask

        // Explicit broadcast to batch size (will be 2 due to CFG in ODE loop)
        // This is safe and zero-cost - broadcasting is a view operation in MLX
        let mask = broadcast(baseMask, to: [1, 1, L_total])  // Keep as [1, 1, L] for now, will broadcast in loop
        eval(mask)
        print("DEBUG generateWithMel: Created mask with explicit broadcast, shape=\(mask.shape), L_total=\(L_total)")
        fflush(stdout)

        let nTimesteps = 10
        let cfgRate: Float = 0.7  // Match Python decoder CFG

        var tSpan: [Float] = []
        for i in 0...(nTimesteps) {
            let linearT = Float(i) / Float(nTimesteps)
            let cosineT = 1.0 - cos(linearT * 0.5 * Float.pi)
            tSpan.append(cosineT)
        }

        var currentT = tSpan[0]
        var dt = tSpan[1] - tSpan[0]

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // Enable debug for first step
            FlowMatchingDecoder.debugStep = (step == 1) ? 1 : 0

            let xIn = concatenated([xt, xt], axis: 0)
            let muIn = concatenated([muT, MLXArray.zeros(like: muT)], axis: 0)
            let spkIn = concatenated([spkCond, MLXArray.zeros(like: spkCond)], axis: 0)
            let condIn = concatenated([condsT, MLXArray.zeros(like: condsT)], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            // Create maskIn using explicit broadcast (not concatenation) to break graph links
            // Broadcast the [1,1,L] mask to [2,1,L] for batch size 2 (CFG)
            let maskIn = broadcast(mask, to: [2, 1, L_total])
            eval(maskIn)  // Force evaluation of the broadcasted mask

            if step == 1 {
                print("DEBUG generateWithMel: ODE step \(step)")
                print("  xIn.shape = \(xIn.shape)")
                print("  muIn.shape = \(muIn.shape)")
                print("  spkIn.shape = \(spkIn.shape)")
                print("  condIn.shape = \(condIn.shape)")
                print("  tIn.shape = \(tIn.shape)")
                print("  maskIn.shape = \(maskIn.shape) (explicitly broadcasted)")
                print("DEBUG generateWithMel: About to call decoder...")
                print("⚠️  WARNING: Mask disabled due to MLX operator cache bug")
                print("    Audio will be ~1.5 dB quieter than expected")
                fflush(stdout)
            }

            // WORKAROUND: Disable mask entirely due to MLX operator cache bug
            // Even with CPU roundtrips and explicit broadcasts, MLX's cached operators
            // from 260-length sequences contaminate 520-length operations
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: nil)

            if step == 1 {
                print("DEBUG generateWithMel: Decoder returned!")
                fflush(stdout)
            }
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            xt = xt + v * dt
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        eval(xt)
        let generatedMel = xt[0..., 0..., L_pm...]
        eval(generatedMel)
        let wav = vocoder(generatedMel)
        eval(wav)
        return (wav, generatedMel)
    }
}
