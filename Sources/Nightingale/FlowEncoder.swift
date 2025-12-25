import MLX
import MLXNN
import Foundation

// MARK: - Relative Position Multi-Head Attention (Conformer-style)

/// Attention with relative position bias (pos_bias_u, pos_bias_v)
public class RelativePositionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var linearQ: Linear
    @ModuleInfo var linearK: Linear
    @ModuleInfo var linearV: Linear
    @ModuleInfo var linearOut: Linear
    @ModuleInfo var linearPos: Linear

    // Learnable position biases
    var posBiasU: MLXArray  // [numHeads, headDim]
    var posBiasV: MLXArray  // [numHeads, headDim]

    public init(dim: Int, numHeads: Int = 8, weights: [String: MLXArray], prefix: String) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        // Load projections using LinearFactory to handle quantized weights
        self.linearQ = LinearFactory.load("\(prefix).linear_q", inputDim: dim, outputDim: dim, weights: weights, bias: true)
        self.linearK = LinearFactory.load("\(prefix).linear_k", inputDim: dim, outputDim: dim, weights: weights, bias: true)
        self.linearV = LinearFactory.load("\(prefix).linear_v", inputDim: dim, outputDim: dim, weights: weights, bias: true)
        self.linearOut = LinearFactory.load("\(prefix).linear_out", inputDim: dim, outputDim: dim, weights: weights, bias: true)
        self.linearPos = LinearFactory.load("\(prefix).linear_pos", inputDim: dim, outputDim: dim, weights: weights, bias: false)

        // Initialize position biases
        self.posBiasU = MLXArray.zeros([numHeads, headDim])
        self.posBiasV = MLXArray.zeros([numHeads, headDim])

        super.init()

        // Load position biases (these are not quantized - just regular tensors)
        if let u = weights["\(prefix).pos_bias_u"] {
            self.posBiasU = u
        }
        if let v = weights["\(prefix).pos_bias_v"] {
            self.posBiasV = v
        }
    }

    public func callAsFunction(_ x: MLXArray, posEmb: MLXArray? = nil) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        // Project Q, K, V
        let q = linearQ(x)  // [B, L, dim]
        let k = linearK(x)
        let v = linearV(x)

        // Reshape to [B, L, numHeads, headDim] then transpose to [B, numHeads, L, headDim]
        // q: [B, T1, h, d_k] -> transpose -> [B, h, T1, d_k]
        let qR = q.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kR = k.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vR = v.reshaped([B, L, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Python: q_with_bias_u = q + pos_bias_u (broadcast)
        // Python: q_with_bias_v = q + pos_bias_v
        // posBiasU/V: [numHeads, headDim] -> [1, numHeads, 1, headDim]
        let uExpanded = posBiasU.reshaped([1, numHeads, 1, headDim])
        let vExpanded = posBiasV.reshaped([1, numHeads, 1, headDim])
        let qWithU = qR + uExpanded  // [B, numHeads, L, headDim]
        let qWithV = qR + vExpanded

        // Compute content-based attention: matrix_ac = q_with_bias_u @ k.T (NO scaling yet)
        let matrixAC = matmul(qWithU, kR.transposed(0, 1, 3, 2))  // [B, numHeads, L, L]

        var scores = matrixAC

        // Compute position-based attention if posEmb provided
        if let posEmb = posEmb {
            // Python: p = self.linear_pos(pos_emb).reshape(n_batch_pos, -1, h, d_k)
            // pos_emb: [B, 2*L-1, dim] or [B, L, dim]
            let p = linearPos(posEmb)  // [B, T, dim]
            let T = p.shape[1]

            // Reshape: [B, T, numHeads, headDim] -> transpose -> [B, numHeads, T, headDim]
            let pR = p.reshaped([B, T, numHeads, headDim]).transposed(0, 2, 1, 3)

            // Python: matrix_bd = q_with_bias_v @ p.T
            var matrixBD = matmul(qWithV, pR.transposed(0, 1, 3, 2))  // [B, numHeads, L, T]

            // Apply rel_shift if shapes don't match
            if matrixAC.shape != matrixBD.shape {
                matrixBD = relShift(matrixBD)
            }

            // Python: scores = (matrix_ac + matrix_bd) / sqrt(d_k)
            // Combine and scale ONCE at the end
            scores = matrixAC + matrixBD
        }

        // Apply scale factor once to final scores
        scores = scores * scale

        // Apply softmax
        let attnWeights = softmax(scores, axis: -1)

        // Apply attention to values
        var output = matmul(attnWeights, vR)  // [B, numHeads, L, headDim]

        // Reshape back
        output = output.transposed(0, 2, 1, 3).reshaped([B, L, numHeads * headDim])

        return linearOut(output)
    }

    // Python's rel_shift function
    private func relShift(_ x: MLXArray) -> MLXArray {
        // Input: x [B, head, T1, 2*T1-1]
        // Output: [B, head, T1, T1]

        let B = x.shape[0]
        let head = x.shape[1]
        let T1 = x.shape[2]
        let T2 = x.shape[3]

        // Add zero padding on the left: [B, head, T1, 1]
        let zeroPad = MLXArray.zeros([B, head, T1, 1], dtype: x.dtype)
        let xPadded = concatenated([zeroPad, x], axis: 3)  // [B, head, T1, T2+1]

        // Reshape to [B, head, T2+1, T1]
        let xReshaped = xPadded.reshaped([B, head, T2 + 1, T1])

        // Take [1:] on axis 2, reshape back to [B, head, T1, T2+1]
        let xShifted = xReshaped[0..., 0..., 1..., 0...].reshaped([B, head, T1, T2])

        // Take first T1 columns: [:, :, :, :T1//2+1]
        let targetCols = T2 / 2 + 1
        return xShifted[0..., 0..., 0..., 0..<targetCols]
    }
}

// MARK: - Feed Forward

public class ConformerFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear

    public init(dim: Int, hiddenDim: Int = 2048, weights: [String: MLXArray], prefix: String) {
        // Use LinearFactory to handle quantized weights
        self.w1 = LinearFactory.load("\(prefix).w_1", inputDim: dim, outputDim: hiddenDim, weights: weights, bias: true)
        self.w2 = LinearFactory.load("\(prefix).w_2", inputDim: hiddenDim, outputDim: dim, weights: weights, bias: true)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = w1(x)
        h = silu(h)
        return w2(h)
    }
}

// MARK: - Conformer Block

public class ConformerEncoderBlock: Module {
    let normMha: LayerNorm
    let selfAttn: RelativePositionAttention
    let normFf: LayerNorm
    let feedForward: ConformerFeedForward

    public init(dim: Int, numHeads: Int = 8, ffHiddenDim: Int = 2048,
                weights: [String: MLXArray], prefix: String) {

        self.normMha = LayerNorm(dimensions: dim)
        self.selfAttn = RelativePositionAttention(dim: dim, numHeads: numHeads,
                                                   weights: weights, prefix: "\(prefix).self_attn")
        self.normFf = LayerNorm(dimensions: dim)
        self.feedForward = ConformerFeedForward(dim: dim, hiddenDim: ffHiddenDim,
                                                 weights: weights, prefix: "\(prefix).feed_forward")

        super.init()

        // Load norm weights
        if let w = weights["\(prefix).norm_mha.weight"] {
            normMha.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).norm_mha.bias"] {
            normMha.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
        if let w = weights["\(prefix).norm_ff.weight"] {
            normFf.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).norm_ff.bias"] {
            normFf.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }
    }

    public func callAsFunction(_ x: MLXArray, posEmb: MLXArray? = nil) -> MLXArray {
        // Pre-norm attention with positional embeddings
        var h = x + selfAttn(normMha(x), posEmb: posEmb)
        // Pre-norm feed-forward
        h = h + feedForward(normFf(h))
        return h
    }
}

// MARK: - Embedding Layer (Linear + Norm + Positional Encoding)

public class FlowEmbedding: Module {
    @ModuleInfo var linear: Linear
    let norm: LayerNorm
    let posEnc: EspnetRelPositionalEncoding

    public init(inputDim: Int, outputDim: Int, weights: [String: MLXArray], prefix: String) {
        // Use LinearFactory to handle quantized weights
        self.linear = LinearFactory.load("\(prefix).linear", inputDim: inputDim, outputDim: outputDim, weights: weights, bias: true)
        self.norm = LayerNorm(dimensions: outputDim)
        self.posEnc = EspnetRelPositionalEncoding(dModel: outputDim)

        super.init()

        // Load norm weights (not quantized)
        if let w = weights["\(prefix).norm.weight"] {
            norm.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).norm.bias"] {
            norm.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

        // Load positional encoding weights
        if let pe = weights["\(prefix).pos_enc.pe"] {
            posEnc.pe = pe
        }
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Apply linear + norm
        let h = norm(linear(x))
        // Apply positional encoding (scales by xscale = sqrt(512) = 22.627)
        let (xScaled, posEmb) = posEnc(h)
        return (xScaled, posEmb)
    }
}

// MARK: - Pre-Lookahead Layer (Two Convolutions)

public class PreLookaheadLayer: Module {
    let conv1: Conv1d
    let conv2: Conv1d
    let preLookaheadLen: Int

    public init(dim: Int, weights: [String: MLXArray], prefix: String, preLookaheadLen: Int = 3) {
        self.preLookaheadLen = preLookaheadLen

        // Python uses padding=0 and manual padding with mx.pad()
        // conv1: kernel = pre_lookahead_len + 1 = 4
        self.conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: preLookaheadLen + 1, padding: 0)
        // conv2: kernel = 3
        self.conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 0)

        super.init()
        // NOTE: conv1 and conv2 weights are loaded later via ChatterboxEngine.update()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L, C]
        // Python implementation with manual padding to maintain sequence length

        let inputs = x

        // 1. Pad RIGHT with pre_lookahead_len frames for conv1
        // mx.pad(outputs, [(0, 0), (0, self.pre_lookahead_len), (0, 0)])
        var outputs = padded(x, widths: [.init((0, 0)), .init((0, preLookaheadLen)), .init((0, 0))])

        // 2. Apply conv1 (kernel=4, padding=0): (L+3) - 4 + 1 = L
        outputs = leakyRelu(conv1(outputs))

        // 3. Pad LEFT with 2 frames for conv2
        // mx.pad(outputs, [(0, 0), (2, 0), (0, 0)])
        outputs = padded(outputs, widths: [.init((0, 0)), .init((2, 0)), .init((0, 0))])

        // 4. Apply conv2 (kernel=3, padding=0): (L+2) - 3 + 1 = L
        outputs = conv2(outputs)

        // 5. Residual connection
        outputs = outputs + inputs

        return outputs
    }
}

// MARK: - Up Layer (Single Convolution)

public class UpLayer: Module {
    let conv: Conv1d

    public init(dim: Int, weights: [String: MLXArray], prefix: String) {
        // Use standard zero padding (padding=2) to match Python mlx-audio-plus
        // The default Conv1d padding is zero padding
        self.conv = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 5, padding: 2)

        super.init()

        // NOTE: conv weights are loaded later via ChatterboxEngine.update()
        // DO NOT transpose weights here to avoid double-transpose bug
        // Conv1d weights will be transposed once in remapS3Keys() and applied via update()
        if let w = weights["\(prefix).conv.weight"] {
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Standard Conv1d with zero padding
        return conv(x)
    }
}

// MARK: - Flow Encoder (Complete)

/// Conformer-style encoder for processing speech tokens into mel-like features
/// Architecture: embed -> 6 blocks -> pre_lookahead -> up_embed -> 4 blocks -> up_layer -> norm -> proj
public class FlowEncoder: Module {
    let hiddenDim: Int
    let melDim: Int

    // Input embedding
    let embed: FlowEmbedding

    // Main encoder blocks (6)
    let encoders: [ConformerEncoderBlock]

    // Mid processing
    let preLookaheadLayer: PreLookaheadLayer

    // Upsampling path
    let upEmbed: FlowEmbedding
    public let upEncoders: [ConformerEncoderBlock]
    public let upLayer: UpLayer

    // Output
    public let afterNorm: LayerNorm
    @ModuleInfo public var encoderProj: Linear

    public init(hiddenDim: Int = 512, melDim: Int = 80, numHeads: Int = 8,
                weights: [String: MLXArray]) {
        self.hiddenDim = hiddenDim
        self.melDim = melDim

        // Use "encoder" prefix to match remapped keys (ChatterboxEngine remaps s3gen.flow.encoder.* -> encoder.*)
        let prefix = "encoder"

        // Embedding
        self.embed = FlowEmbedding(inputDim: hiddenDim, outputDim: hiddenDim,
                                    weights: weights, prefix: "\(prefix).embed")

        // 6 encoder blocks
        var encs: [ConformerEncoderBlock] = []
        for i in 0..<6 {
            encs.append(ConformerEncoderBlock(
                dim: hiddenDim,
                numHeads: numHeads,
                ffHiddenDim: 2048,
                weights: weights,
                prefix: "\(prefix).encoders.\(i)"
            ))
        }
        self.encoders = encs

        // Pre-lookahead layer
        self.preLookaheadLayer = PreLookaheadLayer(dim: hiddenDim, weights: weights,
                                                    prefix: "\(prefix).pre_lookahead_layer")

        // Up embedding
        self.upEmbed = FlowEmbedding(inputDim: hiddenDim, outputDim: hiddenDim,
                                      weights: weights, prefix: "\(prefix).up_embed")

        // 4 up encoder blocks
        var upEncs: [ConformerEncoderBlock] = []
        for i in 0..<4 {
            upEncs.append(ConformerEncoderBlock(
                dim: hiddenDim,
                numHeads: numHeads,
                ffHiddenDim: 2048,
                weights: weights,
                prefix: "\(prefix).up_encoders.\(i)"
            ))
        }
        self.upEncoders = upEncs

        // Up layer
        self.upLayer = UpLayer(dim: hiddenDim, weights: weights, prefix: "\(prefix).up_layer")

        // Final norm
        self.afterNorm = LayerNorm(dimensions: hiddenDim)

        // Output projection to mel dimension (use LinearFactory for quantized weights)
        // Use "encoderProj" to match remapped keys (ChatterboxEngine remaps s3gen.flow.encoder_proj.* -> encoderProj.*)
        self.encoderProj = LinearFactory.load("encoderProj", inputDim: hiddenDim, outputDim: melDim, weights: weights, bias: true)

        super.init()

        // Load final norm weights (not quantized)
        if let w = weights["\(prefix).after_norm.weight"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["weight": w]))
        }
        if let b = weights["\(prefix).after_norm.bias"] {
            afterNorm.update(parameters: ModuleParameters.unflattened(["bias": b]))
        }

    }

    /// Forward pass: speech token embeddings -> mel-like features
    /// Input: [B, L, hiddenDim] - token embeddings from speech embedding layer
    /// Output: [B, L*2, melDim] - mel-like conditioning features (2x upsampled)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Initial embedding (applies linear + norm + pos_enc with xscale=22.627)
        let (h, posEmb) = embed(x)  // posEmb: [B, 2*L-1, dim] for relative attention
        var hCurrent = h

        // Main encoder blocks - pass posEmb to each layer for relative attention
        for encoder in encoders {
            hCurrent = encoder(hCurrent, posEmb: posEmb)
        }

        // Pre-lookahead processing
        hCurrent = preLookaheadLayer(hCurrent)

        // 2x upsampling: repeat each frame to match mel length (token_mel_ratio = 2)
        // h: [B, L, C] -> [B, L*2, C]
        let B = hCurrent.shape[0]
        let L = hCurrent.shape[1]
        let C = hCurrent.shape[2]
        hCurrent = hCurrent.expandedDimensions(axis: 2)  // [B, L, 1, C]
        hCurrent = tiled(hCurrent, repetitions: [1, 1, 2, 1])  // [B, L, 2, C]
        hCurrent = hCurrent.reshaped([B, L * 2, C])  // [B, L*2, C]

        // Upsampling path (applies linear + norm + pos_enc with xscale)
        let (hUp, posEmbUp) = upEmbed(hCurrent)  // posEmbUp: [B, 2*L*2-1, dim]
        hCurrent = hUp

        // Up encoder blocks - pass posEmbUp for relative attention
        for upEncoder in upEncoders {
            hCurrent = upEncoder(hCurrent, posEmb: posEmbUp)
        }

        hCurrent = upLayer(hCurrent)

        // Final norm and projection
        hCurrent = afterNorm(hCurrent)
        hCurrent = encoderProj(hCurrent)

        return hCurrent
    }
}
