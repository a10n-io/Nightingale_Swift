import Foundation
import MLX
import MLXNN

/// A Linear layer that works correctly with transposed PyTorch weights by storing
/// them directly and using manual matmul.
///
/// CRITICAL: This class exists because Linear.update() does NOT persist transposed
/// PyTorch weights. We store weights as direct MLXArray properties so Module.update()
/// can discover and update them, then use manual matmul in forward pass.
public class FixedLinear: Module {
    // Store weights directly as properties so Module.update() can discover them
    // These MUST be mutable vars for Module.update() to work
    public var weight: MLXArray
    public var bias: MLXArray?

    let inputDim: Int
    let outputDim: Int
    let name: String

    public init(_ inputDim: Int, _ outputDim: Int, bias: Bool = true, name: String = "FixedLinear") {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.name = name

        // Initialize with small random weights (MLX format: [In, Out])
        let k = 1.0 / sqrt(Float(inputDim))
        self.weight = MLXRandom.uniform(low: -k, high: k, [inputDim, outputDim])
        self.bias = bias ? MLXRandom.uniform(low: -k, high: k, [outputDim]) : nil

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // CRITICAL: Use manual matmul instead of Linear forward pass
        // This ensures correct behavior with transposed PyTorch weights
        var out = matmul(x, weight)
        if let b = bias {
            out = out + b
        }
        return out
    }
}
