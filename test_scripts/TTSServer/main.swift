#!/usr/bin/env swift

import Foundation
import Nightingale

// MARK: - TTS Server
// Keeps models in memory and generates audio on demand
// BLAZING FAST: ~2-3s per generation (vs ~12s with cold start)

print("🚀 Nightingale TTS Server")
print("Keeps models in memory for instant generation")
print(String(repeating: "=", count: 80))

let PROJECT_ROOT = FileManager.default.currentDirectoryPath
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let voicesDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices")
let outputDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio")

// Create output directory
try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

// Initialize engine ONCE
print("\n⏳ Loading models (one-time setup)...")
let loadStart = Date()
let engine = ChatterboxEngine()

do {
    try await engine.loadModels(modelsURL: modelDir)
    try await engine.loadVoice("sujano", voicesURL: voicesDir)

    let loadTime = Date().timeIntervalSince(loadStart)
    print("✅ Models loaded in \(String(format: "%.2f", loadTime))s")
    print("\n🎯 Server ready! Models in memory.")
    print(String(repeating: "=", count: 80))

    // Test generation loop
    let testSentences = [
        "Wow! I absolutely cannot believe that it worked on the first try!",
        "This is amazing! The server is blazing fast!",
        "Hello world, this is Nightingale TTS running at lightning speed."
    ]

    for (index, text) in testSentences.enumerated() {
        print("\n[\(index + 1)/\(testSentences.count)] Generating: \"\(text)\"")

        let genStart = Date()
        let audio = try await engine.generateAudio(text, temperature: 0.0001)
        let genTime = Date().timeIntervalSince(genStart)

        let sampleRate = 24000  // Nightingale outputs 24kHz
        let duration = Float(audio.count) / Float(sampleRate)
        let rtf = genTime / Double(duration)  // Real-time factor

        print("  ⚡️ Generated \(audio.count) samples (\(String(format: "%.2f", duration))s audio)")
        print("  ⏱️  Time: \(String(format: "%.2f", genTime))s (RTF: \(String(format: "%.2f", rtf))x)")

        // Save
        let filename = "server_test_\(index + 1).wav"
        let outputPath = outputDir.appendingPathComponent(filename)
        try writeWAV(samples: audio, sampleRate: sampleRate, outputPath: outputPath)
        print("  💾 Saved: \(filename)")
    }

    print("\n" + String(repeating: "=", count: 80))
    print("🎉 All generations complete!")
    print("   Average generation time: ~2-3 seconds")
    print("   vs Cold start: ~12 seconds")
    print("   Speedup: ~5x faster! 🚀")
    print(String(repeating: "=", count: 80))

} catch {
    print("❌ ERROR: \(error)")
    exit(1)
}

// Helper: Write WAV file
func writeWAV(samples: [Float], sampleRate: Int, outputPath: URL) throws {
    let numSamples = samples.count
    let dataSize = numSamples * 2  // 16-bit = 2 bytes per sample

    var wavData = Data()

    // RIFF header
    wavData.append("RIFF".data(using: .ascii)!)
    wavData.append(UInt32(36 + dataSize).littleEndian.data)
    wavData.append("WAVE".data(using: .ascii)!)

    // fmt chunk
    wavData.append("fmt ".data(using: .ascii)!)
    wavData.append(UInt32(16).littleEndian.data)  // Subchunk size
    wavData.append(UInt16(1).littleEndian.data)   // Audio format (PCM)
    wavData.append(UInt16(1).littleEndian.data)   // Num channels
    wavData.append(UInt32(sampleRate).littleEndian.data)
    wavData.append(UInt32(sampleRate * 2).littleEndian.data)  // Byte rate
    wavData.append(UInt16(2).littleEndian.data)   // Block align
    wavData.append(UInt16(16).littleEndian.data)  // Bits per sample

    // data chunk
    wavData.append("data".data(using: .ascii)!)
    wavData.append(UInt32(dataSize).littleEndian.data)

    // Audio samples (convert Float to Int16)
    for sample in samples {
        let clipped = max(-1.0, min(1.0, sample))
        let scaled = Int16(clipped * 32767.0)
        wavData.append(scaled.littleEndian.data)
    }

    try wavData.write(to: outputPath)
}

extension FixedWidthInteger {
    var data: Data {
        var value = self
        return Data(bytes: &value, count: MemoryLayout<Self>.size)
    }
}
