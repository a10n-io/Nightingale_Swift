import MLX
import MLXNN
import MLXRandom
import Foundation

// MARK: - Relative Positional Encoding

/// Relative Positional Encoding (ESPNet style)
public class EspnetRelPositionalEncoding: Module {
    public let dModel: Int
    public let dropoutRate: Float
    public let xscale: Float
    
    // We keep pe as state, typically regular PE just computes it.
    // ESPNet implementation extends it.
    // Non-optional so Module.update() can load weights into it
    public var pe: MLXArray
    
    public init(dModel: Int, dropoutRate: Float = 0.1) {
        // 🚨 RED HANDED CHECK
        if dModel == 80 {
            fatalError("EspnetRelPositionalEncoding: dModel cannot be 80 (mel_channels)! Should be 512 or 256.")
        }

        self.dModel = dModel
        self.dropoutRate = dropoutRate
        self.xscale = sqrt(Float(dModel))
        // Initialize with empty pe - will be extended on first use or loaded from weights
        self.pe = MLXArray.zeros([1, 1, dModel])
        super.init()
    }
    
    private func extendPe(size: Int) {
        if pe.shape[1] >= size * 2 - 1 {
            return
        }
        
        let position = MLXArray(Array(0..<size).map { Float($0) }).expandedDimensions(axis: 1) // [size, 1]
        let divTerm = exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) }) *
            -(log(10000.0) / Float(dModel))
        )
        
        // Positive positions
        let posSin = sin(position * divTerm)
        let posCos = cos(position * divTerm)
        
        // Interleave manually: 0::2 = sin, 1::2 = cos
        // MLX doesn't support sophisticated slicing assignment easily in one go for strides
        // Construct interleaved array
        // [size, dModel/2] each
        // Stack along last axis [size, dModel/2, 2] then flatten?
        let posStack = stacked([posSin, posCos], axis: 2).reshaped([size, dModel])
        let pePositive = posStack
        
        // Negative positions
        let negPosition = position * -1
        let negSin = sin(negPosition * divTerm)
        let negCos = cos(negPosition * divTerm)
        let negStack = stacked([negSin, negCos], axis: 2).reshaped([size, dModel])
        let peNegative = negStack
        
        // Reverse positive: [0, 1, 2] -> [2, 1, 0]
        // pePositive is [T, D]
        // Create indices for reversal
        let revIndices = MLXArray(Array(stride(from: size - 1, through: 0, by: -1)).map { Int32($0) })
        let pePosRev = pePositive[revIndices]
        
        // Negative: skip first (0), take rest
        // peNegative is [size, dModel]
        let peNegRest = peNegative[1...]
        
        // Concatenate: [PosRev, NegRest] -> [2*T - 1, D]
        let peFull = concatenated([pePosRev, peNegRest], axis: 0)
        
        // Add batch dim [1, 2*T-1, D]
        self.pe = peFull.expandedDimensions(axis: 0)
    }
    
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // x: [B, T, D]
        let size = x.shape[1]
        extendPe(size: size)
        
        let xScaled = x * xscale
        // pos_emb: [1, 2*T-1, D] center crop
        let center = pe.shape[1] / 2
        let start = center - size + 1
        let end = center + size
        // Use Range for slicing
        let posEmb = pe[0..., start..<end, 0...]
        
        return (xScaled, posEmb)
    }
}

// MARK: - Relative Multi-Head Attention

public class RelPositionMultiHeadAttention: Module {
    let numHeads: Int
    let dModel: Int
    let dHead: Int
    let scale: Float

    // Use same property names as remapping expects: linear_q -> queryProj, etc.
    @ModuleInfo public var queryProj: FixedLinear
    @ModuleInfo public var keyProj: FixedLinear
    @ModuleInfo public var valueProj: FixedLinear
    @ModuleInfo public var outProj: FixedLinear

    // RelPos specific
    @ModuleInfo public var linearPos: FixedLinear
    public var posBiasU: MLXArray // [H, D_h] - var so weights can be loaded
    public var posBiasV: MLXArray // [H, D_h] - var so weights can be loaded

    public init(dModel: Int, numHeads: Int, dropout: Float = 0.1) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.dHead = dModel / numHeads
        self.scale = 1.0 / sqrt(Float(dHead))

        // 🔍 DEBUG: Print ALL RelPositionMultiHeadAttention initializations
        fflush(stdout)

        // 🚨 RED HANDED CHECK
        if dHead == 80 {
            fflush(stdout)
            fatalError("RelPositionMultiHeadAttention: dHead cannot be 80 (mel_channels)!")
        }
        if dModel == 80 {
            fflush(stdout)
        }
        if dModel == 640 {
            fflush(stdout)
            fatalError("RelPositionMultiHeadAttention: dModel=640 is wrong! Should be 512.")
        }

        self.queryProj = FixedLinear(dModel, dModel, name: "RelPosAttn.queryProj")
        self.keyProj = FixedLinear(dModel, dModel, name: "RelPosAttn.keyProj")
        self.valueProj = FixedLinear(dModel, dModel, name: "RelPosAttn.valueProj")
        self.outProj = FixedLinear(dModel, dModel, name: "RelPosAttn.outProj")

        self.linearPos = FixedLinear(dModel, dModel, bias: false, name: "RelPosAttn.linearPos")

        // Initialize biases
        self.posBiasU = MLXRandom.uniform(low: -1, high: 1, [numHeads, dHead])
        self.posBiasV = MLXRandom.uniform(low: -1, high: 1, [numHeads, dHead])

        super.init()
    }
    
    private func relShift(_ x: MLXArray) -> MLXArray {
        // x: [B, H, T, 2*T-1]
        let B = x.shape[0]
        let H = x.shape[1]
        let T = x.shape[2]
        
        // Prepend column of zeros: [B, H, T, 2*T]
        let zeroPad = MLXArray.zeros([B, H, T, 1], dtype: x.dtype)
        let xPad = concatenated([zeroPad, x], axis: -1) // [B, H, T, 2T]
        
        let xR = xPad.reshaped([B, H, 2 * T, T])
        // Slice: drop first "row"
        let xS = xR[0..., 0..., 1...] // [B, H, 2T-1, T] 
        let xFinalFull = xS.reshaped([B, H, T, 2 * T - 1])
        
        // Take first T columns
        return xFinalFull[0..., 0..., 0..., 0..<T]
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, posEmb: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[1]

        let q = queryProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3) // [B, H, T, D_h]
        let k = keyProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3)
        let v = valueProj(x).reshaped([B, T, numHeads, dHead]).transposed(0, 2, 1, 3)

        // Pos embeddings
        let nBatchPos = posEmb.shape[0]
        let posSeqLen = posEmb.shape[1]
        let pProj = linearPos(posEmb)  // [nBatchPos, posSeqLen, dModel]

        // Ensure pProj has the right shape before reshaping
        let expectedDim = numHeads * dHead
        if pProj.shape[2] != expectedDim {
            fatalError("RelPos: linearPos output dim mismatch! Got \(pProj.shape[2]), expected \(expectedDim) (numHeads=\(numHeads) * dHead=\(dHead))")
        }

        let p = pProj.reshaped([nBatchPos, posSeqLen, numHeads, dHead]).transposed(0, 2, 1, 3)

        // Add biases
        let qWithBiasU = q + posBiasU.reshaped([1, numHeads, 1, dHead])
        let qWithBiasV = q + posBiasV.reshaped([1, numHeads, 1, dHead])

        // Attention Scores
        let matrixAC = matmul(qWithBiasU, k.transposed(0, 1, 3, 2)) // [B, H, T, T]
        let matrixBD = matmul(qWithBiasV, p.transposed(0, 1, 3, 2)) // [B, H, T, 2T-1]

        // RelShift matrixBD
        let matrixBDBshifted = relShift(matrixBD) // [B, H, T, T]

        var scores = (matrixAC + matrixBDBshifted) * scale

        if let m = mask {
            scores = scores + m
        }

        let probs = softmax(scores, axis: -1)

        var output = matmul(probs, v) // [B, H, T, D_h]
        output = output.transposed(0, 2, 1, 3).reshaped([B, T, dModel])

        return outProj(output)
    }

    /// Load weights for relative position attention
    /// NOTE: Linear weights are loaded later via ChatterboxEngine.update()
    /// DO NOT transpose weights here to avoid double-transpose bug
    public func load(weights: [String: MLXArray], prefix: String) {
        // NOTE: Q, K, V, Out, and Pos projection weights are loaded via ChatterboxEngine.update()
        // DO NOT load them here to avoid double-transpose bug
        // Only verify they exist and log for debugging
        if let w = weights["\(prefix).linear_q.weight"] {
        }
        if let w = weights["\(prefix).linear_k.weight"] {
        }
        if let w = weights["\(prefix).linear_v.weight"] {
        }
        if let w = weights["\(prefix).linear_out.weight"] {
        }
        if let w = weights["\(prefix).linear_pos.weight"] {
        }

        // Load positional biases
        if let u = weights["\(prefix).pos_bias_u"] {
            eval(u)
            let expectedShape = [numHeads, dHead]
            if u.shape.count != 2 || u.shape[0] != expectedShape[0] || u.shape[1] != expectedShape[1] {
                fatalError("pos_bias_u has wrong shape - expected [\(numHeads), \(dHead)], got \(u.shape)")
            }
            posBiasU = u
        }
        if let v = weights["\(prefix).pos_bias_v"] {
            eval(v)
            let expectedShape = [numHeads, dHead]
            if v.shape.count != 2 || v.shape[0] != expectedShape[0] || v.shape[1] != expectedShape[1] {
                fatalError("pos_bias_v has wrong shape - expected [\(numHeads), \(dHead)], got \(v.shape)")
            }
            posBiasV = v
        }
    }
}
