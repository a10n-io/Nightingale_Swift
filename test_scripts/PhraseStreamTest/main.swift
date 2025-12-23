import Foundation
import MLX
import MLXNN
import Nightingale

/// Test phrase-based streaming TTS
/// Compares TTFA and quality vs batch generation
@main
struct PhraseStreamTest {
    static func main() async throws {
        print("🎤 Phrase-Based Streaming TTS Test")
        print(String(repeating: "=", count: 60))
        print("")

        let modelDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/models/chatterbox")
        let voicesDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/baked_voices")
        let outputDir = URL(fileURLWithPath: "/Users/a10n/Projects/Nightingale_Swift/test_output")

        // Create output directory
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // Test text - multiple sentences to demonstrate phrase streaming value
        // Each sentence is 30+ chars to avoid hallucination bug
        let testText = "Wow! I absolutely cannot believe that it worked on the first try! This is amazing technology that will change everything. I can't wait to see what incredible things come next."

        // Load engine
        print("Loading models (INT8)...")
        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelDir, useQuantization: true)
        try await engine.loadVoice("sujano", voicesURL: voicesDir)
        print("Models loaded.\n")

        // Create phrase streaming wrapper
        let phraseStreamer = await engine.createPhraseStreaming()

        // Show phrase splitting
        let phrases = phraseStreamer.splitIntoPhrases(testText)
        print("📝 Text split into \(phrases.count) phrases:")
        for (i, phrase) in phrases.enumerated() {
            print("   \(i + 1). \"\(phrase)\"")
        }
        print("")

        // ============================================
        // TEST 1: Phrase-based streaming
        // ============================================
        print("🔊 Test 1: Phrase-Based Streaming")
        print(String(repeating: "-", count: 60))

        let streamStart = Date()
        let (firstChunk, ttfa) = try await phraseStreamer.startStreaming(testText)
        print("  TTFA: \(String(format: "%.0f", ttfa * 1000))ms")
        print("  First chunk: \(firstChunk.count) samples (\(String(format: "%.2f", Double(firstChunk.count) / 24000.0))s audio)")

        var allStreamedAudio = firstChunk
        var chunkTimes: [Double] = [ttfa]

        while phraseStreamer.hasMoreChunks {
            let chunkStart = Date()
            if let chunk = try await phraseStreamer.getNextChunk() {
                let chunkTime = Date().timeIntervalSince(chunkStart)
                chunkTimes.append(chunkTime)
                allStreamedAudio.append(contentsOf: chunk)
                let progress = phraseStreamer.progress
                print("  Chunk \(progress.generated)/\(progress.total): \(chunk.count) samples in \(String(format: "%.0f", chunkTime * 1000))ms")
            }
        }

        let streamTotalTime = Date().timeIntervalSince(streamStart)
        let streamDuration = Double(allStreamedAudio.count) / 24000.0
        print("")
        print("  Total time: \(String(format: "%.2f", streamTotalTime))s")
        print("  Audio duration: \(String(format: "%.2f", streamDuration))s")
        print("  RTF: \(String(format: "%.2f", streamTotalTime / streamDuration))x")
        print("")

        // Save streamed audio
        let streamOutputPath = outputDir.appendingPathComponent("phrase_streamed.wav")
        try saveWAV(samples: allStreamedAudio, sampleRate: 24000, to: streamOutputPath)
        print("  Saved: \(streamOutputPath.lastPathComponent)")
        print("")

        // ============================================
        // TEST 2: Batch generation (baseline)
        // ============================================
        print("🔊 Test 2: Batch Generation (Baseline)")
        print(String(repeating: "-", count: 60))

        let batchStart = Date()
        let batchAudio = try await engine.generateAudio(testText, temperature: 0.0001)
        let batchTime = Date().timeIntervalSince(batchStart)
        let batchDuration = Double(batchAudio.count) / 24000.0

        print("  Generation time: \(String(format: "%.2f", batchTime))s")
        print("  Audio duration: \(String(format: "%.2f", batchDuration))s")
        print("  RTF: \(String(format: "%.2f", batchTime / batchDuration))x")
        print("")

        // Save batch audio
        let batchOutputPath = outputDir.appendingPathComponent("phrase_batch.wav")
        try saveWAV(samples: batchAudio, sampleRate: 24000, to: batchOutputPath)
        print("  Saved: \(batchOutputPath.lastPathComponent)")
        print("")

        // ============================================
        // COMPARISON
        // ============================================
        print(String(repeating: "=", count: 60))
        print("📊 COMPARISON")
        print(String(repeating: "=", count: 60))
        print("")
        print("                    │ Phrase Stream │ Batch      │ Improvement")
        print("────────────────────┼───────────────┼────────────┼────────────")
        print("  TTFA              │ \(String(format: "%8.0f", ttfa * 1000))ms    │ \(String(format: "%8.0f", batchTime * 1000))ms   │ \(String(format: "%.1f", batchTime / ttfa))x faster")
        print("  Total time        │ \(String(format: "%8.2f", streamTotalTime))s    │ \(String(format: "%8.2f", batchTime))s   │ \(String(format: "%.2f", batchTime / streamTotalTime))x")
        print("  Audio duration    │ \(String(format: "%8.2f", streamDuration))s    │ \(String(format: "%8.2f", batchDuration))s   │ ")
        print("")

        // Check if streaming audio matches expected duration
        let durationDiff = abs(streamDuration - batchDuration)
        let durationPct = (durationDiff / batchDuration) * 100
        if durationPct < 5 {
            print("  ✅ Audio durations match (within \(String(format: "%.1f", durationPct))%)")
        } else {
            print("  ⚠️ Duration mismatch: \(String(format: "%.1f", durationPct))% difference")
        }

        print("")
        print("  🎯 User hears first audio after \(String(format: "%.0f", ttfa * 1000))ms (vs \(String(format: "%.0f", batchTime * 1000))ms for batch)")
        print("")
    }

    /// Save audio samples to WAV file
    static func saveWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        var data = Data()

        // WAV header
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = UInt32(sampleRate * Int(numChannels) * Int(bitsPerSample) / 8)
        let blockAlign = UInt16(numChannels * bitsPerSample / 8)
        let dataSize = UInt32(samples.count * 2)
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
        }

        try data.write(to: url)
    }
}
