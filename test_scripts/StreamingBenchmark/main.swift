import Foundation
import MLX
import MLXNN
import Nightingale

/// Benchmark streaming vs batch S3Gen synthesis
/// Compares TTFA, total time, and audio quality
@main
struct StreamingBenchmark {
    static func main() async throws {
        print("🎯 Streaming vs Batch S3Gen Benchmark")
        print(String(repeating: "=", count: 70))
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

        let testText = "Hello! This is a benchmark comparing streaming and batch synthesis for Nightingale TTS."

        // ============================================
        // BENCHMARK 1: Batch Generation (Baseline)
        // ============================================
        print("📊 Benchmark 1: Batch Generation (Baseline)")
        print(String(repeating: "-", count: 70))

        let batchStart = Date()
        let batchAudio = try await engine.generateAudio(testText, temperature: 0.0001)
        let batchTotal = Date().timeIntervalSince(batchStart)

        let batchDuration = Double(batchAudio.count) / 24000.0
        print("  Total time: \(String(format: "%.2f", batchTotal))s")
        print("  Audio duration: \(String(format: "%.2f", batchDuration))s")
        print("  RTF: \(String(format: "%.2f", batchTotal / batchDuration))x")
        print("  TTFA: \(String(format: "%.0f", batchTotal * 1000))ms (full wait)")
        print("")

        // Save batch audio
        let batchPath = outputDir.appendingPathComponent("benchmark_batch.wav")
        try saveWAV(samples: batchAudio, to: batchPath)
        print("  💾 Saved: \(batchPath.lastPathComponent)")
        print("")

        // ============================================
        // BENCHMARK 2: Streaming Generation
        // ============================================
        print("📊 Benchmark 2: Streaming Generation")
        print(String(repeating: "-", count: 70))

        // Get speech tokens first
        let t3Start = Date()
        let allTokens = try await engine.runT3Only(testText, temperature: 0.0001)
        let t3Time = Date().timeIntervalSince(t3Start)
        print("  T3: \(allTokens.count) tokens in \(String(format: "%.2f", t3Time))s")

        // Get voice conditioning
        guard let voiceCond = await engine.getVoiceConditioning(),
              let s3gen = await engine.getS3Gen() else {
            print("❌ Failed to get voice conditioning or S3Gen")
            return
        }

        // Create streaming wrapper
        let streaming = S3GenStreaming(s3gen: s3gen)

        // Initialize with prompt (cached)
        let initStart = Date()
        streaming.initializeWithPrompt(
            promptTokens: voiceCond.promptToken,
            promptFeat: voiceCond.promptFeat,
            speechEmbMatrix: voiceCond.s3Soul
        )
        let initTime = Date().timeIntervalSince(initStart)
        print("  Prompt cached in \(String(format: "%.0f", initTime * 1000))ms")

        // Generate in chunks
        let initialChunkSize = 20
        let subsequentChunkSize = 15

        var streamingAudio: [Float] = []
        var chunkTimes: [(tokens: Int, time: Double, audioMs: Double)] = []
        var ttfa: Double = 0

        var tokenIndex = 0
        var chunkNum = 0
        let streamStart = Date()

        while tokenIndex < allTokens.count {
            let chunkSize = (chunkNum == 0) ? initialChunkSize : subsequentChunkSize
            let endIndex = min(tokenIndex + chunkSize, allTokens.count)
            let chunkTokens = Array(allTokens[tokenIndex..<endIndex])

            let chunkStart = Date()
            let chunkAudio = streaming.generateIncremental(newTokens: chunkTokens)
            let chunkTime = Date().timeIntervalSince(chunkStart)

            let audioMs = Double(chunkAudio.count) / 24.0

            if chunkNum == 0 {
                ttfa = Date().timeIntervalSince(streamStart)
            }

            chunkTimes.append((chunkTokens.count, chunkTime, audioMs))
            streamingAudio.append(contentsOf: chunkAudio)

            print("  Chunk \(chunkNum): \(chunkTokens.count) tokens → \(String(format: "%.0f", audioMs))ms audio in \(String(format: "%.0f", chunkTime * 1000))ms")

            tokenIndex = endIndex
            chunkNum += 1
        }

        // Flush any remaining audio
        let remainingAudio = streaming.flush()
        streamingAudio.append(contentsOf: remainingAudio)

        let streamTotal = Date().timeIntervalSince(streamStart)
        let streamDuration = Double(streamingAudio.count) / 24000.0

        print("")
        print("  Total time: \(String(format: "%.2f", streamTotal))s")
        print("  Audio duration: \(String(format: "%.2f", streamDuration))s")
        print("  RTF: \(String(format: "%.2f", streamTotal / max(0.001, streamDuration)))x")
        print("  TTFA: \(String(format: "%.0f", ttfa * 1000))ms")
        print("")

        // Save streaming audio
        let streamPath = outputDir.appendingPathComponent("benchmark_streaming.wav")
        try saveWAV(samples: streamingAudio, to: streamPath)
        print("  💾 Saved: \(streamPath.lastPathComponent)")
        print("")

        // ============================================
        // COMPARISON SUMMARY
        // ============================================
        print(String(repeating: "=", count: 70))
        print("📊 COMPARISON SUMMARY")
        print(String(repeating: "=", count: 70))
        print("")
        print("                    │ Batch      │ Streaming  │ Improvement")
        print("────────────────────┼────────────┼────────────┼────────────")
        print("  TTFA              │ \(String(format: "%6.0f", batchTotal * 1000))ms   │ \(String(format: "%6.0f", ttfa * 1000))ms   │ \(String(format: "%5.1f", batchTotal / max(0.001, ttfa)))x faster")
        print("  Total Time        │ \(String(format: "%6.2f", batchTotal))s   │ \(String(format: "%6.2f", streamTotal))s   │ \(streamTotal < batchTotal ? "✅" : "⚠️ ") \(String(format: "%.1f", batchTotal / max(0.001, streamTotal)))x")
        print("  Audio Length      │ \(String(format: "%6.2f", batchDuration))s   │ \(String(format: "%6.2f", streamDuration))s   │ \(abs(batchDuration - streamDuration) < 0.5 ? "✅ Match" : "⚠️ Diff")")
        print("")

        // Calculate if streaming is viable
        if chunkTimes.count > 1 {
            let avgChunkTime = chunkTimes.dropFirst().map { $0.time }.reduce(0, +) / Double(chunkTimes.count - 1)
            let avgAudioMs = chunkTimes.dropFirst().map { $0.audioMs }.reduce(0, +) / Double(chunkTimes.count - 1)

            print("📈 Streaming Viability Analysis:")
            print("   Avg chunk processing: \(String(format: "%.0f", avgChunkTime * 1000))ms")
            print("   Avg chunk audio: \(String(format: "%.0f", avgAudioMs))ms")
            print("   Margin: \(String(format: "%.0f", avgAudioMs - avgChunkTime * 1000))ms")

            if avgAudioMs > avgChunkTime * 1000 {
                print("   ✅ STREAMING VIABLE - Audio buffer grows during playback")
            } else {
                print("   ⚠️ Streaming may stutter - Processing slower than playback")
            }
        }

        print("")
        print(String(repeating: "=", count: 70))
        print("🎧 Listen to compare quality:")
        print("   afplay \(batchPath.path)")
        print("   afplay \(streamPath.path)")
        print(String(repeating: "=", count: 70))

        // Auto-play batch audio
        print("\n▶️  Playing BATCH audio...")
        let batchProcess = Process()
        batchProcess.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        batchProcess.arguments = [batchPath.path]
        try batchProcess.run()
        batchProcess.waitUntilExit()

        print("▶️  Playing STREAMING audio...")
        let streamProcess = Process()
        streamProcess.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        streamProcess.arguments = [streamPath.path]
        try streamProcess.run()
        streamProcess.waitUntilExit()

        print("✅ Playback complete!")
    }

    /// Save audio samples to a WAV file
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
