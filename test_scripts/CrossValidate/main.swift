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

// MARK: - Safetensors I/O

func saveSpeechTokens(_ tokens: [Int], to url: URL) throws {
    let array = MLXArray(tokens.map { Int32($0) })
    try MLX.save(arrays: ["speech_tokens": array], url: url)
    print("Saved \(tokens.count) tokens to: \(url.lastPathComponent)")
}

func loadSpeechTokens(from url: URL) throws -> [Int] {
    let arrays = try MLX.loadArrays(url: url)
    guard let tokens = arrays["speech_tokens"] else {
        throw NSError(domain: "CrossValidate", code: 1, userInfo: [NSLocalizedDescriptionKey: "speech_tokens not found in file"])
    }
    eval(tokens)
    return tokens.asArray(Int32.self).map { Int($0) }
}

// MARK: - Main

print(String(repeating: "=", count: 80))
print("CROSS-VALIDATION: Swift T3 + S3Gen")
print(String(repeating: "=", count: 80))

// Dynamically find project root (current working directory when running via swift run)
let PROJECT_ROOT = FileManager.default.currentDirectoryPath
let modelDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/models/chatterbox")
let voicesDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/baked_voices")
let outputDir = URL(fileURLWithPath: "\(PROJECT_ROOT)/test_audio/cross_validate")

// Create output directory
try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

// Test text - MUST MATCH Python test
let testText = "Wow! I absolutely cannot believe that it worked on the first try!"
print("\nTest text: \"\(testText)\"")

// Initialize engine
print("\nInitializing ChatterboxEngine...")
let engine = ChatterboxEngine()

do {
    // Load models
    print("Loading models from: \(modelDir.path)")
    try await engine.loadModels(modelsURL: modelDir)
    print("Models loaded successfully")

    // Load voice
    print("\nLoading voice: samantha")
    try engine.loadVoice("samantha", voicesURL: voicesDir)
    print("Voice loaded successfully")

    // =========================================================================
    // STEP 1: Generate speech tokens with Swift T3
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 1: Generate speech tokens with Swift T3")
    print(String(repeating: "=", count: 80))

    let swiftTokens = try engine.runT3Only(testText, temperature: 0.0001)

    // Save Swift tokens
    let swiftTokensPath = outputDir.appendingPathComponent("swift_speech_tokens.safetensors")
    try saveSpeechTokens(swiftTokens, to: swiftTokensPath)

    // =========================================================================
    // STEP 2: Generate audio from Swift tokens with Swift S3Gen
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 2: Swift tokens -> Swift S3Gen")
    print(String(repeating: "=", count: 80))

    let swiftAudio = try engine.synthesizeFromTokens(swiftTokens)

    let swiftSwiftPath = outputDir.appendingPathComponent("swift_tokens_swift_audio.wav")
    try writeWAV(audio: swiftAudio, sampleRate: 24000, to: swiftSwiftPath)
    print("Saved: swift_tokens_swift_audio.wav")

    // =========================================================================
    // STEP 3: Load Python tokens and generate audio with Swift S3Gen
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("STEP 3: Python tokens -> Swift S3Gen")
    print(String(repeating: "=", count: 80))

    let pythonTokensPath = outputDir.appendingPathComponent("python_speech_tokens.safetensors")
    if FileManager.default.fileExists(atPath: pythonTokensPath.path) {
        let pythonTokens = try loadSpeechTokens(from: pythonTokensPath)
        print("Python speech tokens: \(pythonTokens.count)")
        print("  First 20: \(Array(pythonTokens.prefix(20)))")
        print("  Last 20: \(Array(pythonTokens.suffix(20)))")

        // Compare tokens
        let minLen = min(swiftTokens.count, pythonTokens.count)
        var matches = 0
        var firstDiffIdx = -1
        for i in 0..<minLen {
            if swiftTokens[i] == pythonTokens[i] {
                matches += 1
            } else if firstDiffIdx < 0 {
                firstDiffIdx = i
            }
        }
        print("\nToken comparison (first \(minLen)):")
        print("  Matching: \(matches)/\(minLen) (\(String(format: "%.1f", 100.0 * Float(matches) / Float(minLen)))%)")
        if firstDiffIdx >= 0 {
            print("  First diff at index \(firstDiffIdx): Swift=\(swiftTokens[firstDiffIdx]), Python=\(pythonTokens[firstDiffIdx])")
        }

        // Generate audio from Python tokens with Swift S3Gen
        let pythonSwiftAudio = try engine.synthesizeFromTokens(pythonTokens)

        let pythonSwiftPath = outputDir.appendingPathComponent("python_tokens_swift_audio.wav")
        try writeWAV(audio: pythonSwiftAudio, sampleRate: 24000, to: pythonSwiftPath)
        print("Saved: python_tokens_swift_audio.wav")
    } else {
        print("Python tokens not found: \(pythonTokensPath.path)")
        print("Run the Python cross-validation script first:")
        print("  python E2E/cross_validate_python.py")
    }

    // =========================================================================
    // Summary
    // =========================================================================
    print("\n" + String(repeating: "=", count: 80))
    print("CROSS-VALIDATION COMPLETE")
    print(String(repeating: "=", count: 80))
    print("\nOutput directory: \(outputDir.path)")
    print("Files generated:")
    print("  - swift_speech_tokens.safetensors (for Python to load)")
    print("  - swift_tokens_swift_audio.wav (full Swift pipeline)")
    if FileManager.default.fileExists(atPath: pythonTokensPath.path) {
        print("  - python_tokens_swift_audio.wav (tests Swift S3Gen)")
    }
    print("\nTo complete cross-validation, run:")
    print("  python E2E/cross_validate_python.py")

} catch {
    print("\nERROR: \(error)")
    exit(1)
}
