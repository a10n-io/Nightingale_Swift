import Foundation
import MLX
import MLXNN
import MLXRandom
import Nightingale

// MARK: - WAV File Writing

func writeWAV(audio: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate * Int(numChannels) * Int(bitsPerSample) / 8)
    let blockAlign = UInt16(numChannels * bitsPerSample / 8)
    let dataSize = UInt32(audio.count * Int(bitsPerSample) / 8)
    let fileSize = 36 + dataSize

    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    for sample in audio {
        let scaled = Int16(max(-1.0, min(1.0, sample)) * 32767.0)
        data.append(contentsOf: withUnsafeBytes(of: scaled.littleEndian) { Array($0) })
    }
    try data.write(to: url)
}

// MARK: - Main

print(String(repeating: "=", count: 80))
print("NIGHTINGALE TTS - GenerateAudio Test (using ChatterboxEngine)")
print(String(repeating: "=", count: 80))

// Dynamically find project root (current working directory when running via swift run)
let PROJECT_ROOT = FileManager.default.currentDirectoryPath
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let voicesDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices")

// Create ChatterboxEngine - all weight loading/transposition handled internally
print("\nInitializing ChatterboxEngine...")
let engine = ChatterboxEngine()

do {
    // Load models (T3 + S3Gen with all weight transpositions)
    print("Loading models from: \(modelDir.path)")
    try await engine.loadModels(modelsURL: modelDir)
    print("✅ Models loaded successfully")

    // Load voice
    print("\nLoading voice: sujano")
    try engine.loadVoice("sujano", voicesURL: voicesDir)
    print("✅ Voice loaded successfully")

    // Test text - MUST MATCH Python test for verification
    let testText = "Wow! I absolutely cannot believe that it worked on the first try!"
    print("\n" + String(repeating: "=", count: 80))
    print("GENERATING AUDIO")
    print(String(repeating: "=", count: 80))
    print("Text: \"\(testText)\"")

    // Generate audio using ChatterboxEngine
    // temperature=0.0001 for deterministic output (matches Python)
    print("\nRunning T3 + S3Gen pipeline...")
    let audioSamples = try await engine.generateAudio(testText)  // Uses default temp=0.0001

    let duration = Float(audioSamples.count) / 24000.0
    print("✅ Generated \(audioSamples.count) samples (\(String(format: "%.2f", duration))s)")

    // Frequency analysis
    var lowEnergy: Float = 0
    var highEnergy: Float = 0
    let sampleRate: Float = 24000
    for freq in stride(from: 100, through: 500, by: 50) {
        var realSum: Float = 0, imagSum: Float = 0
        for (i, sample) in audioSamples.prefix(10000).enumerated() {
            let angle = 2.0 * Float.pi * Float(freq) * Float(i) / sampleRate
            realSum += sample * cos(angle)
            imagSum += sample * sin(angle)
        }
        lowEnergy += sqrt(realSum * realSum + imagSum * imagSum)
    }
    for freq in stride(from: 5000, through: 10000, by: 500) {
        var realSum: Float = 0, imagSum: Float = 0
        for (i, sample) in audioSamples.prefix(10000).enumerated() {
            let angle = 2.0 * Float.pi * Float(freq) * Float(i) / sampleRate
            realSum += sample * cos(angle)
            imagSum += sample * sin(angle)
        }
        highEnergy += sqrt(realSum * realSum + imagSum * imagSum)
    }
    let totalEnergy = lowEnergy + highEnergy
    print("\nFrequency analysis:")
    print("  Low freq (100-500 Hz): \(String(format: "%.1f", 100 * lowEnergy / totalEnergy))%")
    print("  High freq (5k-10k Hz): \(String(format: "%.1f", 100 * highEnergy / totalEnergy))%")
    if lowEnergy > highEnergy {
        print("  ✅ Correct: Low frequency dominant (speech)")
    } else {
        print("  ⚠️  Warning: High frequency dominant")
    }

    // Save to file
    let outputPath = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/chatterbox_engine_test.wav")
    try writeWAV(audio: audioSamples, sampleRate: 24000, to: outputPath)
    print("\n✅ Saved: \(outputPath.path)")

    print("\n" + String(repeating: "=", count: 80))
    print("✅ TEST COMPLETE!")
    print("   Output: \(outputPath.path)")
    print(String(repeating: "=", count: 80))

} catch {
    print("\n❌ ERROR: \(error)")
    exit(1)
}
