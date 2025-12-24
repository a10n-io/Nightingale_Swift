import Foundation
import MLX
import MLXNN
import Nightingale

/// Compare FP16 vs INT8 performance directly
@main
struct QuantCompare {
    static func main() async throws {
        print("🔬 FP16 vs INT8 Performance Comparison")
        print(String(repeating: "=", count: 60))
        print("")

        let projectRoot = FileManager.default.currentDirectoryPath
        let modelDir = URL(fileURLWithPath: "\(projectRoot)/models/chatterbox")
        let voicesDir = URL(fileURLWithPath: "\(projectRoot)/baked_voices")
        let testText = "Hello, this is a quick performance test."

        // Warmup GPU
        let warmup = MLXArray([1, 2, 3])
        eval(warmup * warmup)

        // ============================================
        // TEST 1: FP16 (no quantization)
        // ============================================
        print("📊 Test 1: FP16 (useQuantization: false)")
        print(String(repeating: "-", count: 60))

        let engineFP16 = ChatterboxEngine()
        let loadFP16Start = Date()
        try await engineFP16.loadModels(modelsURL: modelDir, useQuantization: false)
        try await engineFP16.loadVoice("sujano", voicesURL: voicesDir)
        let loadFP16Time = Date().timeIntervalSince(loadFP16Start)
        print("  Load time: \(String(format: "%.2f", loadFP16Time))s")

        // Warmup run
        let _ = try await engineFP16.generateAudio(testText, temperature: 0.0001)

        // Benchmark runs
        var fp16Times: [Double] = []
        for i in 1...3 {
            let start = Date()
            let audio = try await engineFP16.generateAudio(testText, temperature: 0.0001)
            let time = Date().timeIntervalSince(start)
            fp16Times.append(time)
            let duration = Double(audio.count) / 24000.0
            print("  Run \(i): \(String(format: "%.2f", time))s (audio: \(String(format: "%.2f", duration))s)")
        }
        let fp16Avg = fp16Times.reduce(0, +) / Double(fp16Times.count)
        print("  Average: \(String(format: "%.2f", fp16Avg))s")
        print("")

        // ============================================
        // TEST 2: INT8 (with quantization)
        // ============================================
        print("📊 Test 2: INT8 (useQuantization: true)")
        print(String(repeating: "-", count: 60))

        let engineINT8 = ChatterboxEngine()
        let loadINT8Start = Date()
        try await engineINT8.loadModels(modelsURL: modelDir, useQuantization: true)
        try await engineINT8.loadVoice("sujano", voicesURL: voicesDir)
        let loadINT8Time = Date().timeIntervalSince(loadINT8Start)
        print("  Load time: \(String(format: "%.2f", loadINT8Time))s")

        // Warmup run
        let _ = try await engineINT8.generateAudio(testText, temperature: 0.0001)

        // Benchmark runs
        var int8Times: [Double] = []
        for i in 1...3 {
            let start = Date()
            let audio = try await engineINT8.generateAudio(testText, temperature: 0.0001)
            let time = Date().timeIntervalSince(start)
            int8Times.append(time)
            let duration = Double(audio.count) / 24000.0
            print("  Run \(i): \(String(format: "%.2f", time))s (audio: \(String(format: "%.2f", duration))s)")
        }
        let int8Avg = int8Times.reduce(0, +) / Double(int8Times.count)
        print("  Average: \(String(format: "%.2f", int8Avg))s")
        print("")

        // ============================================
        // COMPARISON
        // ============================================
        print(String(repeating: "=", count: 60))
        print("📊 COMPARISON")
        print(String(repeating: "=", count: 60))
        print("")
        print("                    │ FP16       │ INT8       │ Ratio")
        print("────────────────────┼────────────┼────────────┼────────────")
        print("  Load time         │ \(String(format: "%6.2f", loadFP16Time))s    │ \(String(format: "%6.2f", loadINT8Time))s    │ \(String(format: "%.2f", loadFP16Time / loadINT8Time))x")
        print("  Avg gen time      │ \(String(format: "%6.2f", fp16Avg))s    │ \(String(format: "%6.2f", int8Avg))s    │ \(String(format: "%.2f", fp16Avg / int8Avg))x")
        print("")

        if int8Avg < fp16Avg {
            print("  ✅ INT8 is \(String(format: "%.1f", (fp16Avg / int8Avg - 1) * 100))% faster")
        } else {
            print("  ⚠️ INT8 is \(String(format: "%.1f", (int8Avg / fp16Avg - 1) * 100))% SLOWER")
            print("")
            print("  Possible causes:")
            print("  1. MLX QuantizedLinear may not be optimized for 8-bit on this hardware")
            print("  2. Dequantization overhead at runtime")
            print("  3. Try 4-bit quantization instead (better MLX support)")
        }
    }
}
