import Foundation
import MLX
import MLXNN
import Nightingale

/// Quick benchmark - minimal output, accurate timing
@main
struct QuickBench {
    static func main() async throws {
        let modelDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/models/chatterbox")
        let voicesDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/baked_voices")

        // Suppress loading output by redirecting stderr temporarily
        let testText = "Hello, this is a quick performance test for Nightingale TTS."

        print("Loading models (INT8)...")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelDir, useQuantization: true)
        try await engine.loadVoice("sujano", voicesURL: voicesDir)
        print("Models loaded.\n")

        // Warmup
        print("Warmup run...")
        let _ = try await engine.generateAudio(testText, temperature: 0.0001)
        print("Warmup complete.\n")

        // Benchmark
        print("Running 5 benchmark iterations...")
        var times: [Double] = []

        for i in 1...5 {
            let start = Date()
            let audio = try await engine.generateAudio(testText, temperature: 0.0001)
            let time = Date().timeIntervalSince(start)
            let duration = Double(audio.count) / 24000.0
            times.append(time)
            print("  Run \(i): \(String(format: "%.2f", time))s (audio: \(String(format: "%.2f", duration))s, RTF: \(String(format: "%.2f", time/duration)))")
        }

        let avg = times.reduce(0, +) / Double(times.count)
        let min = times.min()!
        let max = times.max()!

        print("")
        print("Results:")
        print("  Min: \(String(format: "%.2f", min))s")
        print("  Max: \(String(format: "%.2f", max))s")
        print("  Avg: \(String(format: "%.2f", avg))s")
    }
}
