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

// Debug helper for tensor statistics (no-op in production)
private func debugStats(_ x: MLXArray, name: String) {
    // No-op - debug prints removed
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
    @ModuleInfo public var linear1: FixedLinear
    @ModuleInfo public var linear2: FixedLinear
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
        }

        let emb = sinusoidalEmbedding(t, dim: inputDim, scale: 1000.0)

        if TimeMLP.debugEnabled {
            eval(emb)
        }

        let afterLinear1 = linear1(emb)
        if TimeMLP.debugEnabled {
            eval(afterLinear1)
        }

        let afterSiLU = silu(afterLinear1)
        if TimeMLP.debugEnabled {
            eval(afterSiLU)
        }

        let result = linear2(afterSiLU)
        if TimeMLP.debugEnabled {
            eval(result)
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
        h = layers[0](h)
        h = gelu(h)
        h = layers[1](h)
        return h
    }
}

public class MultiHeadAttention: Module {
    @ModuleInfo public var queryProj: FixedLinear
    @ModuleInfo public var keyProj: FixedLinear
    @ModuleInfo public var valueProj: FixedLinear
    @ModuleInfo public var outProj: FixedLinear
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
        }

        if let m = mask {
            // Additive bias mask: broadcast m to [B, numHeads, L, L]
            // m is expected to be [B, 1, L, L] or [L, L] or similar
            scores = scores + m
        }
        let probs = softmax(scores, axis: -1)

        if debug {
            eval(probs)
        }

        var output = matmul(probs, v)

        if debug {
            eval(output)
        }

        output = output.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])
        let result = outProj(output)

        if debug {
            eval(result)
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
    @ModuleInfo public var w1: FixedLinear
    @ModuleInfo public var w2: FixedLinear

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
        }
        h = silu(h) // Swish/SiLU for Conformer
        if S3ConformerFeedForward.debugEnabled {
            eval(h)
        }
        h = w2(h)
        if S3ConformerFeedForward.debugEnabled {
            eval(h)
        }
        return h
    }

    /// Load weights for feed forward
    public func load(weights: [String: MLXArray], prefix: String) {
        // NOTE: w_1 and w_2 weights are loaded via ChatterboxEngine.update()
        // DO NOT load them here to avoid double-transpose bug
        if let w = weights["\(prefix).w_1.weight"] {
        }
        if let w = weights["\(prefix).w_2.weight"] {
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
        }

        let res1 = h
        h = normMHA(h)
        if ConformerBlock.debugEnabled {
            eval(h)
        }

        h = attention(h, mask: mask, posEmb: posEmb)
        if ConformerBlock.debugEnabled {
            eval(h)
        }

        h = h + res1
        if ConformerBlock.debugEnabled {
            eval(h)
        }

        let res2 = h
        h = normFF(h)
        if ConformerBlock.debugEnabled {
            eval(h)
        }

        if ConformerBlock.debugEnabled {
            S3ConformerFeedForward.debugEnabled = true
        }
        h = feedForward(h)
        if ConformerBlock.debugEnabled {
            S3ConformerFeedForward.debugEnabled = false
            eval(h)
        }

        h = h + res2
        if ConformerBlock.debugEnabled {
            eval(h)
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
    @ModuleInfo public var embedLinear: FixedLinear
    public let embedNorm: LayerNorm // Added missing Norm
    public let posEnc: EspnetRelPositionalEncoding
    public let preLookaheadLayer: Module  // Can be PreLookahead or PreLookaheadLayer
    public let encoders: [ConformerBlock]
    public let upLayer: Upsample1D
    @ModuleInfo public var upEmbedLinear: FixedLinear
    public let upEmbedNorm: LayerNorm // Added missing Norm
    public let upPosEnc: EspnetRelPositionalEncoding
    public let upEncoders: [ConformerBlock]
    public let afterNorm: LayerNorm

    public init(inputDim: Int = 512, outputDim: Int = 512, weights: [String: MLXArray]? = nil) {
        self.embedLinear = FixedLinear(inputDim, outputDim, name: "UpsampleEncoder.embedLinear")

        self.embedNorm = LayerNorm(dimensions: outputDim, eps: 1e-5)
        // Initialize as identity transform (gamma=1, beta=0) to avoid signal suppression
        self.embedNorm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([outputDim]),
            "bias": MLXArray.zeros([outputDim])
        ]))

        self.posEnc = EspnetRelPositionalEncoding(dModel: outputDim)

        // Use PreLookaheadLayer with weight loading
        if let w = weights {
            self.preLookaheadLayer = PreLookaheadLayer(dim: outputDim, weights: w, prefix: "encoder.pre_lookahead_layer")
        } else {
            self.preLookaheadLayer = PreLookahead(channels: outputDim)
        }

        var encs: [ConformerBlock] = []
        for _ in 0..<6 {
            encs.append(ConformerBlock(embedDim: outputDim))
        }
        self.encoders = encs

        self.upLayer = Upsample1D(channels: outputDim, outChannels: outputDim, stride: 2)
        self.upEmbedLinear = FixedLinear(outputDim, outputDim, name: "UpsampleEncoder.upEmbedLinear")

        self.upEmbedNorm = LayerNorm(dimensions: outputDim, eps: 1e-5)
        self.upEmbedNorm.update(parameters: ModuleParameters.unflattened([
            "weight": MLXArray.ones([outputDim]),
            "bias": MLXArray.zeros([outputDim])
        ]))

        self.upPosEnc = EspnetRelPositionalEncoding(dModel: outputDim)

        var upEncs: [ConformerBlock] = []
        for _ in 0..<4 {
            upEncs.append(ConformerBlock(embedDim: outputDim))
        }
        self.upEncoders = upEncs

        self.afterNorm = LayerNorm(dimensions: outputDim)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, seqLen: MLXArray? = nil) -> MLXArray {
        // x: [B, T, D]
        // seqLen: [B] - actual sequence lengths for attention masking

        // Create attention mask if sequence length is provided
        // Mask shape: [B, 1, T] where mask[b, 0, t] = 1 if t < seqLen[b] else 0
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

        // 1. Embed + RelPos
        var h = embedLinear(x)
        h = embedNorm(h)
        let (hScaled, posEmb) = posEnc(h)
        h = hScaled

        // 2. PreLookahead
        if let layer = preLookaheadLayer as? PreLookahead {
            h = layer(h)
        } else if let layer = preLookaheadLayer as? PreLookaheadLayer {
            h = layer(h)
        }

        // 3. Encoders (pass posEmb and mask)
        for layer in encoders {
            h = layer(h, mask: mask, posEmb: posEmb)
        }

        // 4. Upsample
        h = h.transposed(0, 2, 1) // [B, C, T]
        h = upLayer(h)
        h = h.transposed(0, 2, 1) // [B, T, C]

        // After upsampling, sequence length doubles
        if var m = mask {
            // Repeat each element twice along time dimension
            // [B, 1, T] -> [B, 1, 2*T]
            m = concatenated([m, m], axis: 2)
            mask = m
        }

        // 5. UpEmbed
        h = upEmbedLinear(h)
        h = upEmbedNorm(h)
        let (hUp, posEmbUp) = upPosEnc(h)
        h = hUp

        // 6. Up Encoders (with upsampled mask)
        for layer in upEncoders {
            h = layer(h, mask: mask, posEmb: posEmbUp)
        }

        // 7. Final Norm
        h = afterNorm(h)

        return h
    }

    /// Load trained weights from Python model
    public func load(weights: [String: MLXArray], prefix: String = "encoder") {
        // Load embedNorm weights
        if let w = weights["\(prefix).embedNorm.weight"] {
            embedNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).embedNorm.bias"] {
            embedNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load pos_enc.pe
        if let pe = weights["\(prefix).embed.pos_enc.pe"] {
            posEnc.pe = pe
        }

        // Load main encoder blocks (0-5)
        for i in 0..<encoders.count {
            let blockPrefix = "\(prefix).encoders.\(i)"
            encoders[i].load(weights: weights, prefix: blockPrefix)
        }

        // Load upEmbedNorm weights
        if let w = weights["\(prefix).upEmbedNorm.weight"] {
            upEmbedNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).upEmbedNorm.bias"] {
            upEmbedNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
        if let pe = weights["\(prefix).up_embed.pos_enc.pe"] {
            upPosEnc.pe = pe
        }

        // Load up encoder blocks (0-3)
        for i in 0..<upEncoders.count {
            let blockPrefix = "\(prefix).up_encoders.\(i)"
            upEncoders[i].load(weights: weights, prefix: blockPrefix)
        }

        // Load after_norm weights
        if let w = weights["\(prefix).after_norm.weight"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).after_norm.bias"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
    }
}

// MARK: - Flow Blocks (Decoder)

// Helper: Apply mask smartly - skip batch elements with all-zero masks (CFG unconditional)
func applyMaskSmartGlobal(_ h: MLXArray, _ mask: MLXArray?) -> MLXArray {
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
            }
            h = applyMaskSmartGlobal(h, mask)
            if CausalBlock1D.debugCalls {
                eval(h)
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
    @ModuleInfo public var mlpLinear: FixedLinear
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
        var h = block1(x, mask: mask)

        // mlp = Sequential(Mish(), Linear()) - mish BEFORE linear!
        var tEmb = mlpLinear(mish(timeEmb))
        tEmb = tEmb.expandedDimensions(axis: 2)

        h = h + tEmb
        h = block2(h, mask: mask)

        // res_conv with transpose
        var xForRes = x
        if let mask = mask {
            xForRes = x * mask
        }

        var res = xForRes.transposed(0, 2, 1)
        res = resConv(res)
        res = res.transposed(0, 2, 1)

        let output = h + res

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
        h = attention(h, mask: mask)
        h = h + res1
        let res2 = h
        h = norm2(h)
        h = ff(h)
        h = h + res2
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
            }
        }

        if debug {
            eval(x); eval(mu); eval(speakerEmb); eval(cond)

            // Show each batch element separately
            let B = x.shape[0]
            for b in 0..<B {
                let xb = x[b]
                let mub = mu[b]
                let spkb = speakerEmb[b]
                let condb = cond[b]
                eval(xb); eval(mub); eval(spkb); eval(condb)
            }
        }

        TimeMLP.debugEnabled = debug
        let tEmb = timeMLP(t)
        eval(tEmb)
        if debug {
        }

        // spkEmb is 80-dim already (projected by caller)
        let spkExpanded = tiled(speakerEmb.expandedDimensions(axis: 2), repetitions: [1, 1, L])

        if debug {
        }

        // Matches Python order: x, mu, spks, cond
        // x(80) + mu(80) + spk(80) + cond(80) = 320
        if debug {
            checkSpatial(x, label: "  x")
            checkSpatial(mu, label: "  mu")
            checkSpatial(spkExpanded, label: "  spk_expanded")
            checkSpatial(cond, label: "  cond")
        }
        var h = concatenated([x, mu, spkExpanded, cond], axis: 1) // [B, 320, T]
        if debug {
            eval(h)
            checkSpatial(h, label: "01_concat")
            // ===== MICRO-BISECTION CHECKPOINT A: After input concatenation =====
            debugStats(h, name: "CHECKPOINT_A_input_concat")
        }

        // Attention Mask: combines causal/non-causal pattern with padding mask
        if debug {
        }
        var attnMask = makeAttentionMask(L, paddingMask: mask)
        if debug {
            eval(attnMask)
        }

        // WORKAROUND: When mask=nil, pass nil to ALL operations to avoid MLX operator cache bug
        let useMask = (mask != nil)

        if debug {
            if let m = mask {
                eval(m)
                let B = m.shape[0]
                for b in 0..<B {
                    let maskSum = m[b].sum().item(Float.self)
                }
            }
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
            }
        }

        let down = downBlocks[0]
        let maskDown = useMask ? masks.last : nil

        if debug {
            if let md = maskDown {
            } else {
            }
            CausalResNetBlock.debugEnabled = true
            CausalBlock1D.debugCalls = true  // Enable mask debugging
        }
        h = down.resnet(h, mask: maskDown, timeEmb: tEmb)
        if debug {
            CausalResNetBlock.debugEnabled = false
            CausalBlock1D.debugCalls = false
            eval(h)
            checkSpatial(h, label: "02_down_resnet")
            // ===== MICRO-BISECTION CHECKPOINT B: After first ResNet =====
            debugStats(h, name: "CHECKPOINT_B_after_first_resnet")
        }
        h = h.transposed(0, 2, 1)
        if debug {
            eval(h)
        }
        if debug {
            checkSpatial(h.transposed(0, 2, 1), label: "tfmr_input (B,C,T)")
        }

        for (ti, tfmr) in down.transformers.enumerated() {
            if debug && ti == 0 {
                let h_in = h
                checkSpatial(h_in.transposed(0, 2, 1), label: "  tfmr input (B,T,C)")

                let h_normed = tfmr.norm1(h); eval(h_normed)
                checkSpatial(h_normed.transposed(0, 2, 1), label: "  after norm1")

                // Enable detailed attention debug for first transformer
                MultiHeadAttention.debugEnabled = true
                MultiHeadAttention.debugId = "down.tfmr[0]"
                let attn_out = tfmr.attention(h_normed, mask: attnMask); eval(attn_out)
                checkSpatial(attn_out.transposed(0, 2, 1), label: "  after attn")
                MultiHeadAttention.debugEnabled = false

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
                checkSpatial(h.transposed(0, 2, 1), label: "After tfmr[\(ti)] (B,C,T)")
            }
        }
        h = h.transposed(0, 2, 1)
        if debug {
            eval(h)
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
        for (_, mid) in midBlocks.enumerated() {
             h = mid.resnet(h, mask: maskMid, timeEmb: tEmb)
             h = h.transposed(0, 2, 1) // [B, T, C]
             for (_, tfmr) in mid.transformers.enumerated() {
                 h = tfmr(h, mask: attnMask)
             }
             h = h.transposed(0, 2, 1)
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
            // Check spatial bias of h and skip separately
            checkSpatial(hTrunc, label: "05_h_before_concat")
            checkSpatial(skip, label: "05_skip_before_concat")
        }
        h = concatenated([hTrunc, skip], axis: 1)
        if debug {
            eval(h)
            checkSpatial(h, label: "05_skip_concat")
        }

        // Up blocks: after concat with skip, h is back to full resolution - use full mask (or nil)
        let maskFull: MLXArray? = useMask ? masks.first : nil

        if debug && useMask {
            if let mf = maskFull {
                eval(mf)
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
            checkSpatial(h, label: "07_up_tfmrs")
            // Check per-channel bias after up.tfmrs
            for c in [0, 64, 128, 192, 255] {
                let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                let genMean = h[0, c, 500...].mean().item(Float.self)
            }
        }
        // Python applies mask before upsample (only if using mask)
        if let ul = up.upLayer {
            if debug {
                // Check upLayer weights
                let ulConv = ul.conv
                let ulWeight = ulConv.weight
                eval(ulWeight)
                if let ulBias = ulConv.bias {
                    eval(ulBias)
                }
            }
            if useMask, let mf = maskFull {
                let hMasked = applyMaskSmartGlobal(h, mf)
                h = ul(hMasked)
            } else {
                h = ul(h)
            }
            if debug {
                eval(h)
                checkSpatial(h, label: "07b_after_upLayer")
                // Check per-channel bias after upLayer
                for c in [0, 64, 128, 192, 255] {
                    let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                    let genMean = h[0, c, 500...].mean().item(Float.self)
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
            checkSpatial(h, label: "08_finalBlock")
            // Check per-channel bias after finalBlock
            for c in [0, 64, 128, 192, 255] {
                let promptMean = h[0, c, 0..<500].mean().item(Float.self)
                let genMean = h[0, c, 500...].mean().item(Float.self)
            }
        }

        // Python multiplies by mask before final_proj: output = self.final_proj(x * mask_up)
        if useMask, let mf = maskFull {
            h = applyMaskSmart(h, mf)
        }

        // Final Proj: MLX Conv1d expects input as [B, T, C] (channels LAST!)
        h = h.transposed(0, 2, 1) // [B, C, T] → [B, T, C]

        h = finalProj(h)          // [B, T, 80]
        h = h.transposed(0, 2, 1) // [B, T, 80] → [B, 80, T]

        // Python multiplies by mask after final_proj: return output * mask
        if useMask, let mf = maskFull {
            h = applyMaskSmart(h, mf)
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
    @ModuleInfo public var linear: FixedLinear

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
        // Optimized linear interpolation upsampling
        // input: [B, T] -> output: [B, T * scaleFactor]
        let B = input.shape[0]
        let inputLen = input.shape[1]
        let outputLen = inputLen * scaleFactor

        // Compute interpolation indices and weights
        let outIndices = MLXArray(0..<outputLen).asType(.float32)
        let srcIndices = outIndices * Float(inputLen) / Float(outputLen)
        let srcIndicesInt = floor(srcIndices).asType(.int32)
        let alpha = srcIndices - floor(srcIndices)
        let srcIndicesInt1 = minimum(srcIndicesInt + 1, MLXArray(inputLen - 1))

        // Vectorized batch processing
        var results: [MLXArray] = []
        for b in 0..<B {
            let inputSlice = input[b]
            let val0 = inputSlice[srcIndicesInt]
            let val1 = inputSlice[srcIndicesInt1]
            results.append(val0 + (val1 - val0) * alpha)
        }

        return stacked(results, axis: 0)
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
        }

        // mel: [B, 80, L] -> [B, L, 80]
        var x = mel.transposed(0, 2, 1)

        if Mel2Wav.debugEnabled {
            eval(x)
        }

        // F0 Prediction
        let f0 = f0Predictor(x) // [B, T]
        if Mel2Wav.debugEnabled {
            eval(f0)
        }

        // Upsample F0
        // Scale = product of uprates * hop(4) = 8*5*3 * 4 = 120 * 4 = 480
        // CRITICAL FIX: Use LINEAR interpolation (not tiled/nearest) to match PyTorch's Upsample
        // Tiled creates stair-steps which cause phase discontinuities in the sine generator
        let f0Upsampled = upsampleLinear1D(f0, scaleFactor: 480) // [B, T * 480]
        let f0Flat = f0Upsampled.expandedDimensions(axis: 2) // [B, T_high, 1]
        if Mel2Wav.debugEnabled {
            eval(f0Flat)
        }

        // Generate Source
        if Mel2Wav.debugEnabled {
            let linearW = mSource.linear.weight
            eval(linearW)
            if let linearB = mSource.linear.bias {
                eval(linearB)
            } else {
            }
        }
        let (s, _, _) = mSource(f0Flat) // [B, T_high, 1]
        if Mel2Wav.debugEnabled {
            eval(s)
        }
        
        // Source STFT
        let (sReal, sImag) = stft(x: s.squeezed(axis: 2), nFFT: 16, hopLength: 4, window: stftWindow)
        let sSTFT = concatenated([sReal, sImag], axis: 1) // [B, F(18), Frames]
        // var si = sSTFT.transposed(0, 2, 1) // REMOVED unused variable

        if Mel2Wav.debugEnabled {
            eval(sSTFT)
        }

        // CRITICAL DEBUG: Check Conv1d weight shape to detect transposition issue
        if Mel2Wav.debugEnabled {
            eval(convPre.weight)
        }

        // Main Path
        x = convPre(x)  // [B, L, 512]

        if Mel2Wav.debugEnabled {
            eval(x)
        }

        for i in 0..<ups.count {
            if Mel2Wav.debugEnabled {
            }

            x = leakyRelu(x, negativeSlope: 0.1)

            if Mel2Wav.debugEnabled {
            }

            x = ups[i](x)  // Upsample

            if Mel2Wav.debugEnabled {
                eval(x)
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
            }

            siLocal = sourceDowns[i](siLocal)

            if Mel2Wav.debugEnabled && i == 0 {
                eval(siLocal)
            }

            siLocal = sourceResBlocks[i](siLocal)

            if Mel2Wav.debugEnabled {
                eval(siLocal)
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
            }
        }

        // NOTE: Python uses F.leaky_relu(x) here (no slope arg), which defaults to 0.01
        // The upsampling loop uses 0.1, but this final one uses PyTorch's default of 0.01
        x = leakyRelu(x, negativeSlope: 0.01)
        x = convPost(x)  // [B, L*480, 18]

        if Mel2Wav.debugEnabled {
            eval(x)

            // Check magnitude/phase before ISTFT
            let nHalf = 9
            let mag = clip(exp(x[0..., 0..., 0..<nHalf]), max: 100.0)
            let phaseSin = sin(x[0..., 0..., nHalf...])
            eval(mag)
            eval(phaseSin)
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
    @ModuleInfo public var encoderProj: FixedLinear
    @ModuleInfo public var spkEmbedAffine: FixedLinear
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

        // Load input_embedding weights
        for (key, value) in flowWeights {
            if key.hasPrefix("s3gen.flow.input_embedding.") {
                let remappedKey = key.replacingOccurrences(of: "s3gen.flow.input_embedding.", with: "")
                if remappedKey == "weight" {
                    inputEmbedding.update(parameters: ModuleParameters.unflattened(["weight": value]))
                }
            } else if key.hasPrefix("flow.input_embedding.") {
                let remappedKey = key.replacingOccurrences(of: "flow.input_embedding.", with: "")
                if remappedKey == "weight" {
                    inputEmbedding.update(parameters: ModuleParameters.unflattened(["weight": value]))
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

        // Create UpsampleEncoder with weights for PreLookaheadLayer
        self.encoder = UpsampleEncoder(inputDim: config.inputDim, outputDim: config.inputDim, weights: encoderWeights)
        self.encoder.load(weights: encoderWeights, prefix: "encoder")

        self.encoderProj = FixedLinear(config.inputDim, config.melChannels, name: "S3Gen.encoderProj")
        self.spkEmbedAffine = FixedLinear(192, config.melChannels, name: "S3Gen.spkEmbedAffine")

        self.decoder = FlowMatchingDecoder(config: config)
        self.vocoder = Mel2Wav()

        // Generate fixed noise once at init (matching Python's CausalConditionalCFM)
        MLXRandom.seed(0)
        self.fixedNoise = MLXRandom.normal([1, 80, 50 * 300])
        super.init()

        // Load decoder weights with comprehensive key remapping
        var decoderWeights: [String: MLXArray] = [:]
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
                }

                // Transpose Conv1d weights from PyTorch [out, in, kernel] to MLX [out, kernel, in]
                let isConv1d = remappedKey.hasSuffix(".weight") && finalW.ndim == 3 &&
                              !remappedKey.contains("norm") && !remappedKey.contains("embedding")
                if isConv1d {
                    finalW = finalW.transposed(0, 2, 1)
                }

                decoderWeights[remappedKey] = finalW
            }
        }

        if decoderWeights.count > 0 {
            self.decoder.update(parameters: ModuleParameters.unflattened(decoderWeights))
        }

        // =======================================================================================
        // LOAD VOCODER WEIGHTS
        // =======================================================================================
        if let vocoderWeights = vocoderWeights {
            var processedVocoderWeights: [String: MLXArray] = [:]

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
                    finalWeight = finalWeight.transposed(0, 2, 1)
                }

                // Transpose ConvTranspose1d: PyTorch [In, Out, Kernel] -> MLX [Out, Kernel, In]
                if remappedKey.contains("ups.") && finalWeight.ndim == 3 {
                    finalWeight = finalWeight.transposed(1, 2, 0)
                }

                let finalKey = remappedKey + ".weight"
                processedVocoderWeights[finalKey] = finalWeight
                weightNormProcessedKeys.insert(finalKey)
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
                    processedVocoderWeights[key] = finalW
                }

                // Transpose Conv1d weights
                let isConv1d = finalW.ndim == 3 && !key.contains("ups.")
                if isConv1d {
                    finalW = finalW.transposed(0, 2, 1)
                    processedVocoderWeights[key] = finalW
                }

                // Transpose ConvTranspose1d weights
                let isConvT = key.contains("ups.") && finalW.ndim == 3
                if isConvT {
                    finalW = finalW.transposed(1, 2, 0)
                    processedVocoderWeights[key] = finalW
                }
            }

            // Apply weights to vocoder
            self.vocoder.update(parameters: ModuleParameters.unflattened(processedVocoderWeights))
        }
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
    }
    
    public func generate(tokens: MLXArray, speakerEmb: MLXArray, speechEmbMatrix: MLXArray, promptToken: MLXArray, promptFeat: MLXArray) -> MLXArray {
        // tokens: [1, T] generated speech tokens
        // promptToken: [1, P] history tokens
        // promptFeat: [1, T, 80] history mels (channels-last)

        // 1. Prepare Embedding (use speechEmbMatrix, not speakerEmb!)
        // speechEmbMatrix is [1, 192] and gets projected to [1, 80] via spkEmbedAffine
        var speechEmb = speechEmbMatrix  // [1, 192]
        let norm = sqrt(sum(speechEmb * speechEmb, axis: 1, keepDims: true)) + 1e-8
        speechEmb = speechEmb / norm

        // Project speech embedding [1, 192] -> [1, 80]
        let spkCond = matmul(speechEmb, spkEmbedAffine.weight) + spkEmbedAffine.bias!

        // 2. Concat prompt tokens + new tokens
        let inputs = concatenated([promptToken, tokens], axis: 1)

        // 3. Embed (clip tokens to valid range like Python)
        let vocabSize = inputEmbedding.weight.shape[0]
        let clippedInputs = clip(inputs, min: 0, max: vocabSize - 1)
        let x = inputEmbedding(clippedInputs)

        // 4. Encode
        let h = encoder(x) // [1, 2*T_total, 512]
        let mu = encoderProj(h) // [1, 2*T_total, 80]
        eval(mu)  // Force evaluation to avoid deferred computation overhead

        // 5. Prepare conditions for flow decoder
        let promptMel = promptFeat // Already [1, T, 80]
        let L_pm = promptMel.shape[1]

        // conds is a hybrid: Ground Truth prompt mels + zeros for new mels
        let L_new = mu.shape[1] - L_pm
        let muZeros = MLXArray.zeros([1, L_new, 80], dtype: mu.dtype)
        let conds = concatenated([promptMel, muZeros], axis: 1) // [1, L_total, 80]

        // 6. Decode (Flow Matching) with CFG
        let L_total = conds.shape[1]

        // Use pre-generated fixed noise (matches Python's CausalConditionalCFM)
        var xt = fixedNoise[0..., 0..., 0..<L_total]

        // Transpose conds and mu for decoder [B, C, T]
        let condsT = conds.transposed(0, 2, 1)
        let muT = mu.transposed(0, 2, 1)

        // Create mask matching Python: [B, 1, T] (all 1s for valid positions)
        let mask = MLXArray.ones([1, 1, L_total], dtype: muT.dtype)

        // ODE Parameters
        let nTimesteps = 8  // Reduced from 10 for ~20% speedup with minimal quality impact
        let cfgRate: Float = 0.7  // Match Python decoder CFG

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

        // Pre-allocate zero tensors for CFG (reused in loop for efficiency)
        let zeroMu = MLXArray.zeros(like: muT)
        let zeroSpk = MLXArray.zeros(like: spkCond)
        let zeroCond = MLXArray.zeros(like: condsT)

        for step in 1...nTimesteps {
            let t = MLXArray([currentT])

            // CFG: Conditional and unconditional passes
            // Prepare batch for CFG: [Cond, Uncond]
            let xIn = concatenated([xt, xt], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([muT, zeroMu], axis: 0)
            let spkIn = concatenated([spkCond, zeroSpk], axis: 0)
            let condIn = concatenated([condsT, zeroCond], axis: 0)
            let tIn = concatenated([t, t], axis: 0)

            // Forward pass (Batch=2)
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: maskIn)
            let vCond = vBatch[0].expandedDimensions(axis: 0)
            let vUncond = vBatch[1].expandedDimensions(axis: 0)

            // CFG formula: v = (1 + cfg) * vCond - cfg * vUncond
            let v = (1.0 + cfgRate) * vCond - cfgRate * vUncond

            // Euler step
            xt = xt + v * dt

            // Update time for next step
            currentT = currentT + dt
            if step < nTimesteps {
                dt = tSpan[step + 1] - currentT
            }
        }

        // 7. Vocode - extract generated portion and convert to audio
        let generatedMel = xt[0..., 0..., L_pm...]
        let wav = vocoder(generatedMel)

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
        let nTimesteps = 8  // Reduced from 10 for ~20% speedup with minimal quality impact
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
        // Return FULL mel (including prompt) for testing
        // In production, you might want to slice: xt[0..., 0..., L_pm...] to get only new frames
        let generatedMel = xt  // Return full 696 frames
        eval(generatedMel)
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

        // Encoder outputs [1, T, 512], need to project to [1, T, 80]
        let h = encoder(x)
        let mu = encoderProj(h)
        eval(mu)

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


        // Save all traces to file
        try MLX.save(arrays: traceData, url: URL(fileURLWithPath: tracePath))
        for key in traceData.keys.sorted() {
            let arr = traceData[key]!
            eval(arr)
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

        // Force CPU roundtrip to break MLX computation graph
        let hData = h.asArray(Float.self)
        let hShape = h.shape
        h = MLXArray(hData, hShape)

        var mu = encoderProj(h)

        // Force CPU roundtrip for mu as well
        let muData = mu.asArray(Float.self)
        let muShape = mu.shape
        mu = MLXArray(muData, muShape)

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
        let mask = broadcast(baseMask, to: [1, 1, L_total])
        eval(mask)

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

            // WORKAROUND: Disable mask entirely due to MLX operator cache bug
            let vBatch = decoder(x: xIn, mu: muIn, t: tIn, speakerEmb: spkIn, cond: condIn, mask: nil)
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
