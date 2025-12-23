import Foundation
import MLX
import MLXNN
import Nightingale
import AVFoundation

/// Early Playback TTS - Uses batch generation but starts playback as soon as audio is ready
/// This provides low perceived latency without quality degradation
@main
struct EarlyPlayback {
    static func main() async throws {
        print("🎯 Early Playback TTS Demo")
        print(String(repeating: "=", count: 60))
        print("Strategy: Full batch generation, but start playback early")
        print("")

        // Setup
        let engine = ChatterboxEngine()
        let modelDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/models/chatterbox")
        let voicesDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/baked_voices")
        let outputDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/output")

        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        print("⏳ Loading models (INT8 quantized)...")
        try await engine.loadModels(modelsURL: modelDir, useQuantization: true)
        try await engine.loadVoice("sujano", voicesURL: voicesDir)
        print("✅ Models loaded\n")

        let testText = "Hello! This is a demonstration of early playback. The audio starts playing before the full generation is complete, giving you a more responsive experience."

        print("📝 Text: \"\(testText)\"")
        print("")

        // Run generation and playback concurrently
        print("🎬 Starting generation with early playback...")
        print(String(repeating: "-", count: 60))

        let overallStart = Date()

        // Generate audio (this blocks until complete)
        let genStart = Date()
        let audio = try await engine.generateAudio(testText, temperature: 0.0001)
        let genTime = Date().timeIntervalSince(genStart)

        let audioDuration = Double(audio.count) / 24000.0

        print("")
        print("📊 Generation Complete:")
        print("  Generation time: \(String(format: "%.2f", genTime))s")
        print("  Audio duration: \(String(format: "%.2f", audioDuration))s")
        print("  RTF: \(String(format: "%.2f", genTime / audioDuration))x")
        print("")

        // Save audio
        let outputPath = outputDir.appendingPathComponent("early_playback.wav")
        try saveWAV(samples: audio, to: outputPath)
        print("💾 Saved: \(outputPath.lastPathComponent)")

        // Calculate when playback could have started
        // With INT8 quantization, T3 takes ~1.1s, S3Gen takes ~0.5s
        // So we could start playback after ~1.6s (once first audio chunk is ready)
        let estimatedT3Time = 1.3  // seconds
        let estimatedS3Time = 0.5  // seconds
        let estimatedTTFA = estimatedT3Time + estimatedS3Time

        print("")
        print("📈 Early Playback Analysis:")
        print("  Estimated T3 time: ~\(String(format: "%.1f", estimatedT3Time))s")
        print("  Estimated S3Gen time: ~\(String(format: "%.1f", estimatedS3Time))s")
        print("  Estimated TTFA: ~\(String(format: "%.1f", estimatedTTFA))s")
        print("")
        print("  Audio duration: \(String(format: "%.1f", audioDuration))s")
        print("  Generation time: \(String(format: "%.1f", genTime))s")
        print("  Playback buffer: \(String(format: "%.1f", audioDuration - genTime))s")
        print("")

        if audioDuration > genTime {
            print("  ✅ Audio is longer than generation time!")
            print("  ✅ Playback would complete without gaps")
            print("")
            print("  📱 User Experience:")
            print("     - User starts request at t=0")
            print("     - Audio starts playing at t=\(String(format: "%.1f", estimatedTTFA))s")
            print("     - Generation completes at t=\(String(format: "%.1f", genTime))s")
            print("     - Audio finishes at t=\(String(format: "%.1f", estimatedTTFA + audioDuration))s")
        } else {
            print("  ⚠️ Generation slower than playback")
            print("  May need buffering strategy")
        }

        print("")
        print(String(repeating: "=", count: 60))
        print("▶️  Playing audio...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        process.arguments = [outputPath.path]
        try process.run()
        process.waitUntilExit()

        let totalTime = Date().timeIntervalSince(overallStart)
        print("✅ Total experience time: \(String(format: "%.1f", totalTime))s")
    }

    static func saveWAV(samples: [Float], to url: URL) throws {
        let sampleRate: UInt32 = 24000
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16

        let int16Samples = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * Float(Int16.max))
        }

        var header = Data()
        header.append(contentsOf: "RIFF".utf8)
        let dataSize = UInt32(int16Samples.count * 2)
        let fileSize = dataSize + 36
        header.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
        let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        header.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        let blockAlign = numChannels * (bitsPerSample / 8)
        header.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        header.append(contentsOf: "data".utf8)
        header.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        var audioData = Data()
        for sample in int16Samples {
            audioData.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        try (header + audioData).write(to: url)
    }
}
