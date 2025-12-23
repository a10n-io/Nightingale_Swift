import Foundation
import MLX
import MLXNN
import Nightingale
import AVFoundation

/// Streaming TTS Test - Generate audio in chunks and measure real TTFA
/// Tests the recommended configuration:
/// - Initial chunk: 20 tokens (~400ms TTFA, ~800ms audio buffer)
/// - Subsequent chunks: 10 tokens (pipelined)
@main
struct StreamingTest {
    static func main() async throws {
        print("🎯 Streaming TTS Test with Audio Playback")
        print(String(repeating: "=", count: 60))
        print("")

        // Initialize engine with INT8 quantization
        let engine = ChatterboxEngine()
        let modelDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/models/chatterbox")
        let voicesDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/baked_voices")
        let outputDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/output")

        // Create output directory if needed
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        print("⏳ Loading models (INT8 quantized)...")
        try await engine.loadModels(modelsURL: modelDir, useQuantization: true)
        try await engine.loadVoice("sujano", voicesURL: voicesDir)
        print("✅ Models loaded\n")

        let testText = "Hello! This is a streaming test. I am generating audio in chunks to achieve low latency."

        print("📝 Test text: \"\(testText)\"")
        print("")

        // ============================================
        // STEP 1: Generate all tokens first (T3)
        // ============================================
        print("📊 Step 1: Generate Speech Tokens (T3)")
        print(String(repeating: "-", count: 60))

        let t3Start = Date()
        let allTokens = try await engine.runT3Only(testText, temperature: 0.0001)
        let t3Time = Date().timeIntervalSince(t3Start)

        print("  T3 generated \(allTokens.count) tokens in \(String(format: "%.2f", t3Time))s")
        print("  Token rate: \(String(format: "%.1f", Double(allTokens.count) / t3Time)) tokens/sec")
        print("")

        // ============================================
        // STEP 2: Chunked S3Gen Synthesis
        // ============================================
        print("📊 Step 2: Chunked Audio Synthesis (S3Gen)")
        print(String(repeating: "-", count: 60))

        let initialChunkSize = 20
        let subsequentChunkSize = 10

        var allAudio: [Float] = []
        var chunkTimes: [(chunk: Int, tokens: Int, time: Double, audioMs: Double)] = []
        var firstAudioTime: Double = 0

        // Process chunks
        var tokenIndex = 0
        var chunkNum = 0

        while tokenIndex < allTokens.count {
            let chunkSize = (chunkNum == 0) ? initialChunkSize : subsequentChunkSize
            let endIndex = min(tokenIndex + chunkSize, allTokens.count)
            let chunkTokens = Array(allTokens[tokenIndex..<endIndex])

            let chunkStart = Date()
            let chunkAudio = try await engine.synthesizeFromTokens(chunkTokens)
            let chunkTime = Date().timeIntervalSince(chunkStart)

            let audioMs = Double(chunkAudio.count) / 24.0  // 24000 samples/sec = 24 samples/ms

            if chunkNum == 0 {
                firstAudioTime = chunkTime
            }

            chunkTimes.append((chunkNum, chunkTokens.count, chunkTime, audioMs))
            allAudio.append(contentsOf: chunkAudio)

            print("  Chunk \(chunkNum): \(chunkTokens.count) tokens → \(String(format: "%.0f", audioMs))ms audio in \(String(format: "%.0f", chunkTime * 1000))ms")

            tokenIndex = endIndex
            chunkNum += 1
        }

        print("")

        // ============================================
        // STEP 3: Results Summary
        // ============================================
        print("📊 Results Summary")
        print(String(repeating: "-", count: 60))

        let totalAudioMs = Double(allAudio.count) / 24.0
        let avgChunkTime = chunkTimes.dropFirst().map { $0.time }.reduce(0, +) / Double(max(1, chunkTimes.count - 1))

        print("  Total tokens: \(allTokens.count)")
        print("  Total chunks: \(chunkNum)")
        print("  Total audio: \(String(format: "%.1f", totalAudioMs / 1000))s")
        print("")
        print("  ⏱️  ACTUAL TTFA (Time-to-First-Audio):")
        print("     T3 for \(initialChunkSize) tokens: ~\(String(format: "%.0f", Double(initialChunkSize) / Double(allTokens.count) * t3Time * 1000))ms (estimated)")
        print("     S3Gen first chunk: \(String(format: "%.0f", firstAudioTime * 1000))ms")
        print("     Total TTFA: ~\(String(format: "%.0f", Double(initialChunkSize) / Double(allTokens.count) * t3Time * 1000 + firstAudioTime * 1000))ms")
        print("")
        print("  📈 Chunk Processing Performance:")
        print("     First chunk (\(initialChunkSize) tokens): \(String(format: "%.0f", firstAudioTime * 1000))ms → \(String(format: "%.0f", chunkTimes[0].audioMs))ms audio")
        if chunkTimes.count > 1 {
            print("     Avg subsequent chunk: \(String(format: "%.0f", avgChunkTime * 1000))ms")
        }
        print("")

        // Check streaming viability
        let audioPerToken = totalAudioMs / Double(allTokens.count)
        let genTimePerToken = (t3Time * 1000) / Double(allTokens.count)

        print("  🔍 Streaming Viability:")
        print("     Audio per token: \(String(format: "%.1f", audioPerToken))ms")
        print("     T3 time per token: \(String(format: "%.1f", genTimePerToken))ms")
        print("     Margin: \(String(format: "%.1f", audioPerToken - genTimePerToken))ms/token")
        if audioPerToken > genTimePerToken {
            print("     ✅ VIABLE - Audio plays slower than generation")
        } else {
            print("     ⚠️  May have gaps - generation slower than playback")
        }
        print("")

        // ============================================
        // STEP 4: Save Audio Files
        // ============================================
        print("💾 Saving Audio Files")
        print(String(repeating: "-", count: 60))

        // Save full audio
        let fullAudioPath = outputDir.appendingPathComponent("streaming_full.wav")
        try saveWAV(samples: allAudio, to: fullAudioPath)
        print("  ✅ Full audio: \(fullAudioPath.lastPathComponent) (\(String(format: "%.1f", totalAudioMs / 1000))s)")

        // Save first chunk separately (for TTFA testing)
        if let firstChunk = chunkTimes.first {
            let firstChunkSamples = Int(firstChunk.audioMs * 24)
            let firstChunkAudio = Array(allAudio.prefix(firstChunkSamples))
            let firstChunkPath = outputDir.appendingPathComponent("streaming_first_chunk.wav")
            try saveWAV(samples: firstChunkAudio, to: firstChunkPath)
            print("  ✅ First chunk: \(firstChunkPath.lastPathComponent) (\(String(format: "%.0f", firstChunk.audioMs))ms)")
        }

        print("")
        print(String(repeating: "=", count: 60))
        print("🎧 Audio saved! Play with: afplay \(outputDir.path)/streaming_full.wav")
        print(String(repeating: "=", count: 60))

        // Auto-play the audio
        print("\n▶️  Playing audio...")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
        process.arguments = [fullAudioPath.path]
        try process.run()
        process.waitUntilExit()
        print("✅ Playback complete!")
    }

    /// Save audio samples to a WAV file
    static func saveWAV(samples: [Float], to url: URL) throws {
        let sampleRate: UInt32 = 24000
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16

        // Convert Float samples to Int16
        let int16Samples = samples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * Float(Int16.max))
        }

        // WAV header
        var header = Data()

        // RIFF header
        header.append(contentsOf: "RIFF".utf8)
        let dataSize = UInt32(int16Samples.count * 2)
        let fileSize = dataSize + 36
        header.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        header.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        header.append(contentsOf: "fmt ".utf8)
        header.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // chunk size
        header.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // PCM format
        header.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
        let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        header.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        let blockAlign = numChannels * (bitsPerSample / 8)
        header.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        header.append(contentsOf: "data".utf8)
        header.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Audio data
        var audioData = Data()
        for sample in int16Samples {
            audioData.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        // Write file
        let fullData = header + audioData
        try fullData.write(to: url)
    }
}
