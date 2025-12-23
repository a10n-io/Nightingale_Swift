import Foundation
import MLX
import MLXNN
import Nightingale

@main
struct QuantizeAndSave {
    static func main() async throws {
        print("🔧 Quantize and Save T3 Model")
        print(String(repeating: "=", count: 60))

        let modelDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/models/chatterbox")
        let inputPath = modelDir.appendingPathComponent("t3_mtl23ls_v2.safetensors")
        let outputPath = modelDir.appendingPathComponent("t3_mtl23ls_v2_int8.safetensors")

        // Check input exists
        guard FileManager.default.fileExists(atPath: inputPath.path) else {
            print("❌ Input file not found: \(inputPath.path)")
            return
        }

        print("📥 Loading FP16 weights from: \(inputPath.lastPathComponent)")
        let startLoad = Date()
        let fp16Weights = try MLX.loadArrays(url: inputPath)
        let loadTime = Date().timeIntervalSince(startLoad)
        print("   Loaded \(fp16Weights.count) arrays in \(String(format: "%.2f", loadTime))s")

        // Calculate original size
        var originalSize: Int = 0
        for (_, array) in fp16Weights {
            originalSize += array.nbytes
        }
        print("   Original size in memory: \(originalSize / 1024 / 1024) MB")

        // Debug: print some key names
        let sortedKeys = fp16Weights.keys.sorted()
        print("\n📋 Sample weight keys:")
        for key in sortedKeys.prefix(10) {
            print("   \(key)")
        }
        print("   ...")
        let attnKeys = sortedKeys.filter { $0.contains("self_attn") || $0.contains("mlp") }
        print("   Attention/MLP keys (\(attnKeys.count) total):")
        for key in attnKeys.prefix(10) {
            print("   \(key)")
        }

        print("\n⚡️ Quantizing weights to INT8...")
        let startQuant = Date()

        var quantizedWeights: [String: MLXArray] = [:]
        var quantizedCount = 0
        var skippedCount = 0
        let groupSize = 64

        for (key, weight) in fp16Weights {
            // Only quantize large linear layers (attention and MLP projections)
            // Keys use snake_case: self_attn.q_proj, mlp.gate_proj, etc.
            let shouldQuantize = (
                key.contains("self_attn") && (
                    key.contains("q_proj") ||
                    key.contains("k_proj") ||
                    key.contains("v_proj") ||
                    key.contains("o_proj")
                )
            ) || (
                key.contains("mlp") && (
                    key.contains("gate_proj") ||
                    key.contains("up_proj") ||
                    key.contains("down_proj")
                )
            )

            if shouldQuantize && key.hasSuffix(".weight") {
                // Check dimensions are compatible with group size
                let shape = weight.shape
                if shape.count == 2 && shape[1] >= groupSize && shape[1] % groupSize == 0 {
                    // Quantize this weight
                    let (quantized, scales, biases) = MLX.quantized(weight, groupSize: groupSize, bits: 8)

                    // Store quantized weight and metadata
                    let baseKey = String(key.dropLast(7)) // Remove ".weight"
                    quantizedWeights["\(baseKey).weight"] = quantized
                    quantizedWeights["\(baseKey).scales"] = scales
                    quantizedWeights["\(baseKey).biases"] = biases
                    quantizedCount += 1

                    if quantizedCount <= 3 {
                        print("   ✅ Quantized: \(key) \(shape) -> INT8")
                    }
                } else {
                    quantizedWeights[key] = weight
                    skippedCount += 1
                }
            } else {
                // Keep as-is (embeddings, norms, biases, etc.)
                quantizedWeights[key] = weight
                skippedCount += 1
            }
        }

        let quantTime = Date().timeIntervalSince(startQuant)
        print("   ... and \(quantizedCount - 3) more layers")
        print("   Quantized \(quantizedCount) layers, kept \(skippedCount) as FP16")
        print("   Quantization took \(String(format: "%.2f", quantTime))s")

        // Calculate new size
        var newSize: Int = 0
        for (_, array) in quantizedWeights {
            newSize += array.nbytes
        }
        print("   New size in memory: \(newSize / 1024 / 1024) MB")
        print("   Compression: \(String(format: "%.1f", Float(originalSize) / Float(newSize)))x")

        print("\n💾 Saving quantized weights to: \(outputPath.lastPathComponent)")
        let startSave = Date()
        try MLX.save(arrays: quantizedWeights, url: outputPath)
        let saveTime = Date().timeIntervalSince(startSave)
        print("   Saved in \(String(format: "%.2f", saveTime))s")

        // Check file size
        let attrs = try FileManager.default.attributesOfItem(atPath: outputPath.path)
        let fileSize = attrs[.size] as? Int ?? 0
        print("   File size: \(fileSize / 1024 / 1024) MB")

        print("\n✅ Quantization complete!")
        print("   Original: \(inputPath.lastPathComponent) (2.0 GB)")
        print("   Quantized: \(outputPath.lastPathComponent) (\(fileSize / 1024 / 1024) MB)")
    }
}
