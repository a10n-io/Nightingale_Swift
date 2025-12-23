import Foundation
import MLX
import MLXNN
import Nightingale

struct TestSentence: Codable {
    let id: String
    let description: String
    let text_en: String
    let text_nl: String
}

// MARK: - WAV File Writing

func writeWAV(audio: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let numSamples = audio.count
    let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample) / 8
    let blockAlign = numChannels * bitsPerSample / 8
    let dataSize = UInt32(numSamples * 2)

    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: (36 + dataSize).littleEndian) { Array($0) })
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
        let intSample = Int16(max(-32768, min(32767, sample * 32767)))
        data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
    }

    try data.write(to: url)
}

@main
struct GenerateTestSentences {
    static func main() async throws {
        print(String(repeating: "=", count: 80))
        print("GENERATING TEST SENTENCES - SWIFT FLOW ONLY")
        print(String(repeating: "=", count: 80))

        // Load test sentences
        let jsonPath = "../E2E/test_sentences.json"
        let jsonURL = URL(fileURLWithPath: jsonPath)
        let jsonData = try Data(contentsOf: jsonURL)
        let sentences = try JSONDecoder().decode([TestSentence].self, from: jsonData)

        print("\nLoaded \(sentences.count) test sentences")

        // Create output directory
        let outputDir = URL(fileURLWithPath: "../test_audio/test_sentences")
        try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // Initialize ChatterboxEngine
        print("\nInitializing ChatterboxEngine...")
        let modelsPath = URL(fileURLWithPath: "../models/chatterbox")
        let voicesPath = URL(fileURLWithPath: "../baked_voices")

        let engine = ChatterboxEngine()
        try await engine.loadModels(modelsURL: modelsPath)

        // List of voices to test
        let voices = ["samantha", "sujano"]

        // Generate audio for each combination
        var totalFiles = 0
        let startTime = Date()

        for voice in voices {
            print("\n" + String(repeating: "=", count: 80))
            print("VOICE: \(voice.uppercased())")
            print(String(repeating: "=", count: 80))

            // Load voice
            print("\nLoading voice: \(voice)")
            try await engine.loadVoice(voice, voicesURL: voicesPath)

            for sentence in sentences {
                // English version
                print("\n[\(sentence.id)] English:")
                print("  Text: \(sentence.text_en)")

                let enStartTime = Date()
                let enAudio = try await engine.generateAudio(sentence.text_en, temperature: 0.0001)
                let enDuration = Date().timeIntervalSince(enStartTime)

                let enFilename = "\(voice)_en_\(sentence.id).wav"
                let enPath = outputDir.appendingPathComponent(enFilename)
                try writeWAV(audio: enAudio, sampleRate: 24000, to: enPath)

                let enAudioDuration = Float(enAudio.count) / 24000.0
                print("  Generated: \(enAudio.count) samples (\(String(format: "%.2f", enAudioDuration))s)")
                print("  Processing time: \(String(format: "%.2f", enDuration))s")
                print("  Saved: \(enFilename)")
                totalFiles += 1

                // Dutch version
                print("\n[\(sentence.id)] Dutch:")
                print("  Text: \(sentence.text_nl)")

                let nlStartTime = Date()
                let nlAudio = try await engine.generateAudio(sentence.text_nl, temperature: 0.0001)
                let nlDuration = Date().timeIntervalSince(nlStartTime)

                let nlFilename = "\(voice)_nl_\(sentence.id).wav"
                let nlPath = outputDir.appendingPathComponent(nlFilename)
                try writeWAV(audio: nlAudio, sampleRate: 24000, to: nlPath)

                let nlAudioDuration = Float(nlAudio.count) / 24000.0
                print("  Generated: \(nlAudio.count) samples (\(String(format: "%.2f", nlAudioDuration))s)")
                print("  Processing time: \(String(format: "%.2f", nlDuration))s")
                print("  Saved: \(nlFilename)")
                totalFiles += 1
            }
        }

        let totalDuration = Date().timeIntervalSince(startTime)

        print("\n" + String(repeating: "=", count: 80))
        print("GENERATION COMPLETE!")
        print(String(repeating: "=", count: 80))
        print("\nSummary:")
        print("  Total files generated: \(totalFiles)")
        print("  Total processing time: \(String(format: "%.2f", totalDuration))s")
        print("  Output directory: \(outputDir.path)")
        print("\nFiles:")

        let files = try FileManager.default.contentsOfDirectory(at: outputDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        for file in files {
            print("  - \(file.lastPathComponent)")
        }

        print("\n" + String(repeating: "=", count: 80))
    }
}
