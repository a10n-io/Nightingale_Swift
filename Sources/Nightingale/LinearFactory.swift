import MLX
import MLXNN
import Foundation

/// Factory for creating Linear layers that automatically handles quantized vs FP16 weights.
/// Checks for `_scales` and `_biases` keys to determine if a layer is quantized.
public enum LinearFactory {

    /// Load a Linear layer, automatically detecting if it's quantized (4-bit) or FP16.
    ///
    /// - Parameters:
    ///   - name: The key prefix in the weights dictionary (e.g., "layers.0.selfAttn.qProj")
    ///   - inputDim: Input dimension of the linear layer
    ///   - outputDim: Output dimension of the linear layer
    ///   - weights: The full weights dictionary from safetensors
    ///   - bias: Whether this layer has a bias term
    ///   - forceFP16: If true, forces FP16 loading even if Q4 weights exist (for precision-critical layers)
    /// - Returns: A Module that is either QuantizedLinear (4-bit) or Linear (FP16)
    public static func load(
        _ name: String,
        inputDim: Int,
        outputDim: Int,
        weights: [String: MLXArray],
        bias: Bool = false,
        forceFP16: Bool = false
    ) -> Linear {
        let weightKey = "\(name).weight"

        // Check for quantization keys - support both formats:
        // Format 1 (Q4): {name}.weight_scales, {name}.weight_biases
        // Format 2 (INT8): {name}.scales, {name}.biases
        let hasQ4Weights = weights["\(weightKey)_scales"] != nil &&
                           weights["\(weightKey)_biases"] != nil &&
                           weights[weightKey] != nil

        let hasINT8Weights = weights["\(name).scales"] != nil &&
                             weights["\(name).biases"] != nil &&
                             weights[weightKey] != nil

        // If forceFP16 is requested but only Q4 weights exist, dequantize them
        if forceFP16 && hasQ4Weights {
            guard let scales = weights["\(weightKey)_scales"],
                  let biases = weights["\(weightKey)_biases"],
                  let packedWeight = weights[weightKey] else {
                fatalError("Missing Q4 weights for \(name)")
            }


            // Dequantize: reconstruct FP16 weights from Q4
            var dequantized = dequantizeQ4(packed: packedWeight, scales: scales, biases: biases, groupSize: 64, bits: 4)

            // MLX Linear expects weights shaped [in_features, out_features]
            // Dequantized weights are in PyTorch format [out_features, in_features]
            if dequantized.shape[0] == outputDim && dequantized.shape[1] == inputDim {
                dequantized = dequantized.transposed(0, 1)
            }

            let linear = Linear(inputDim, outputDim, bias: bias)
            var params: [String: MLXArray] = ["weight": dequantized]
            if bias, let b = weights["\(name).bias"] {
                params["bias"] = b
            }
            linear.update(parameters: ModuleParameters.unflattened(params))
            return linear
        }

        // Otherwise load Q4 if available (and not forcing FP16)
        if !forceFP16 && hasQ4Weights {
            guard let scales = weights["\(weightKey)_scales"],
                  let biases = weights["\(weightKey)_biases"],
                  let packedWeight = weights[weightKey] else {
                fatalError("Missing Q4 weights for \(name)")
            }


            // Create QuantizedLinear with positional arguments (no labels for dimensions)
            let qLinear = QuantizedLinear(
                inputDim,
                outputDim,
                bias: bias,
                groupSize: 64,
                bits: 4
            )

            // Use update(parameters:) - MLX properties are immutable 'let' constants
            // ModuleParameters is NestedDictionary<String, MLXArray>
            var params: [String: MLXArray] = [
                "weight": packedWeight,
                "scales": scales,
                "biases": biases
            ]
            if bias, let b = weights["\(name).bias"] {
                params["bias"] = b
            }
            qLinear.update(parameters: ModuleParameters.unflattened(params))

            return qLinear
        }

        // Load INT8 quantized weights (format 2: {name}.scales, {name}.biases)
        if !forceFP16 && hasINT8Weights {
            guard let scales = weights["\(name).scales"],
                  let biases = weights["\(name).biases"],
                  let packedWeight = weights[weightKey] else {
                fatalError("Missing INT8 weights for \(name)")
            }


            // Create QuantizedLinear for INT8
            let qLinear = QuantizedLinear(
                inputDim,
                outputDim,
                bias: bias,
                groupSize: 64,
                bits: 8
            )

            var params: [String: MLXArray] = [
                "weight": packedWeight,
                "scales": scales,
                "biases": biases
            ]
            if bias, let b = weights["\(name).bias"] {
                params["bias"] = b
            }
            qLinear.update(parameters: ModuleParameters.unflattened(params))

            return qLinear
        }

        // Fallback: Standard FP16 Linear
        let linear = Linear(inputDim, outputDim, bias: bias)

        if let w = weights[weightKey] {
            // MLX Linear expects weights shaped [in_features, out_features]
            // PyTorch saves them as [out_features, in_features]
            // Need to transpose if shape doesn't match expected [inputDim, outputDim]
            var weightToUse = w
            if w.shape[0] == outputDim && w.shape[1] == inputDim {
                // Weight is in PyTorch format [out, in], transpose to MLX format [in, out]
                weightToUse = w.transposed(0, 1)
            } else {
            }
            // Use update(parameters:) - MLX properties are immutable 'let' constants
            var params: [String: MLXArray] = ["weight": weightToUse]
            if bias, let b = weights["\(name).bias"] {
                params["bias"] = b
            }
            linear.update(parameters: ModuleParameters.unflattened(params))
        } else {
            // Load bias even if no weight found
            if bias, let b = weights["\(name).bias"] {
                linear.update(parameters: ModuleParameters.unflattened(["bias": b]))
            }
        }

        return linear
    }

    /// Check if a given layer name has quantized weights in the dictionary
    public static func isQuantized(_ name: String, in weights: [String: MLXArray]) -> Bool {
        let weightKey = "\(name).weight"
        // Check both Q4 format ({name}.weight_scales) and INT8 format ({name}.scales)
        let hasQ4 = weights["\(weightKey)_scales"] != nil && weights["\(weightKey)_biases"] != nil
        let hasINT8 = weights["\(name).scales"] != nil && weights["\(name).biases"] != nil
        return hasQ4 || hasINT8
    }

    /// Get statistics about quantized vs FP16 layers in a weights dictionary
    public static func analyzeWeights(_ weights: [String: MLXArray]) -> (quantized: Int, fp16: Int, other: Int) {
        var quantized = 0
        var fp16 = 0
        var other = 0

        // Find all unique layer names (strip _scales, _biases suffixes)
        var layerNames = Set<String>()
        for key in weights.keys {
            let baseName = key
                .replacingOccurrences(of: "_scales", with: "")
                .replacingOccurrences(of: "_biases", with: "")
            if baseName.hasSuffix(".weight") {
                layerNames.insert(String(baseName.dropLast(7))) // Remove ".weight"
            }
        }

        for name in layerNames {
            if isQuantized(name, in: weights) {
                quantized += 1
            } else if weights["\(name).weight"] != nil {
                fp16 += 1
            } else {
                other += 1
            }
        }

        return (quantized, fp16, other)
    }

    /// Dequantize Q4 weights back to FP16 for precision-critical layers.
    ///
    /// Q4 format: packed 4-bit values with per-group scales and biases.
    /// Formula: dequantized[i] = (packed[i] - bias) * scale
    ///
    /// - Parameters:
    ///   - packed: Packed Q4 weights [outDim, inDim/2] (2 values per byte)
    ///   - scales: Per-group scaling factors [outDim, numGroups]
    ///   - biases: Per-group bias values [outDim, numGroups]
    ///   - groupSize: Size of each quantization group (typically 64)
    ///   - bits: Number of bits per value (4 for Q4)
    /// - Returns: Dequantized FP16 weights [outDim, inDim]
    private static func dequantizeQ4(
        packed: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> MLXArray {

        let outDim = scales.shape[0]
        let numGroups = scales.shape[1]
        let inDim = numGroups * groupSize


        // The packed array uses uint32 where each uint32 contains 8 4-bit values
        // Format: [outDim, inDim/8] with 8 values per uint32
        let packedFlat = packed.flattened().asArray(UInt32.self)

        // Unpack: each uint32 contains 8 4-bit values
        var unpackedValues: [UInt32] = []
        unpackedValues.reserveCapacity(packedFlat.count * 8)

        for packedInt in packedFlat {
            // Extract 8 4-bit values from the 32-bit integer
            for shift in stride(from: 0, to: 32, by: 4) {
                let val = (packedInt >> shift) & 0xF
                unpackedValues.append(val)
            }
        }


        // Reshape to [outDim, inDim]
        let unpacked = MLXArray(unpackedValues.map { Float($0) }).reshaped([outDim, inDim])

        // Apply scales and biases per group
        // scales/biases: [outDim, numGroups]
        // Expand to [outDim, inDim] by repeating each group value groupSize times
        // Reshape to [outDim, numGroups, 1], then tile to [outDim, numGroups, groupSize], then flatten
        let scalesReshaped = scales.reshaped([outDim, numGroups, 1])
        let biasesReshaped = biases.reshaped([outDim, numGroups, 1])
        let scalesTiled = MLXArray.repeated(scalesReshaped, count: groupSize, axis: 2)
        let biasesTiled = MLXArray.repeated(biasesReshaped, count: groupSize, axis: 2)
        let scalesExpanded = scalesTiled.reshaped([outDim, inDim])
        let biasesExpanded = biasesTiled.reshaped([outDim, inDim])

        // Dequantize: MLX uses w = scale * quantized + bias
        // Source: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html
        let dequantized = scalesExpanded * unpacked + biasesExpanded


        return dequantized
    }
}
