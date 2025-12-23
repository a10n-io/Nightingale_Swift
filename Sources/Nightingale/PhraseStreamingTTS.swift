import Foundation
import MLX

/// Phrase-based streaming TTS that generates complete phrases in batch
/// then queues them for playback. Avoids the bidirectional encoder issues
/// by never splitting mid-phrase.
///
/// Strategy:
/// 1. Split text by punctuation (., ?, !, ;, :, -)
/// 2. Generate each phrase using full batch (perfect quality)
/// 3. Queue audio chunks for seamless playback
/// 4. Pre-fetch next phrase while current one plays
public class PhraseStreamingTTS {

    private let engine: ChatterboxEngine

    /// Audio chunks ready for playback
    private var audioQueue: [[Float]] = []

    /// Phrases waiting to be generated
    private var pendingPhrases: [String] = []

    /// Currently generating phrase index
    private var generatingIndex: Int = 0

    /// Generation temperature
    public var temperature: Float = 0.0001

    /// Minimum phrase length (characters) to avoid hallucination with short text
    /// Model needs ~30+ chars for stable output
    public var minPhraseLength: Int = 30

    public init(engine: ChatterboxEngine) {
        self.engine = engine
    }

    // MARK: - Text Splitting

    /// Split text into natural phrases at punctuation boundaries
    /// Ensures each phrase is long enough to avoid hallucination
    public func splitIntoPhrases(_ text: String) -> [String] {
        // Split on sentence-ending punctuation only (not commas - too aggressive)
        let separators = CharacterSet(charactersIn: ".!?")

        var phrases: [String] = []
        var currentPhrase = ""

        for char in text {
            currentPhrase.append(char)

            // Check if we hit a separator
            if let scalar = char.unicodeScalars.first, separators.contains(scalar) {
                let trimmed = currentPhrase.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    phrases.append(trimmed)
                }
                currentPhrase = ""
            }
        }

        // Don't forget the last phrase (may not end with punctuation)
        let trimmed = currentPhrase.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            phrases.append(trimmed)
        }

        // Merge short phrases to avoid hallucination
        var mergedPhrases: [String] = []
        var buffer = ""

        for phrase in phrases {
            if buffer.isEmpty {
                buffer = phrase
            } else {
                buffer += " " + phrase
            }

            if buffer.count >= minPhraseLength {
                mergedPhrases.append(buffer)
                buffer = ""
            }
        }

        // Add any remaining buffer
        if !buffer.isEmpty {
            if let last = mergedPhrases.popLast() {
                mergedPhrases.append(last + " " + buffer)
            } else {
                mergedPhrases.append(buffer)
            }
        }

        return mergedPhrases
    }

    // MARK: - Streaming Generation

    /// Start streaming generation - returns first audio chunk ASAP
    /// Call `getNextChunk()` to get subsequent chunks
    public func startStreaming(_ text: String) async throws -> (firstChunk: [Float], ttfa: Double) {
        let startTime = Date()

        // Split into phrases
        pendingPhrases = splitIntoPhrases(text)
        audioQueue = []
        generatingIndex = 0

        guard !pendingPhrases.isEmpty else {
            return ([], 0)
        }

        // Generate first phrase immediately
        let firstPhrase = pendingPhrases[0]
        let firstAudio = try await engine.generateAudio(firstPhrase, temperature: temperature)

        let ttfa = Date().timeIntervalSince(startTime)

        audioQueue.append(firstAudio)
        generatingIndex = 1

        return (firstAudio, ttfa)
    }

    /// Get next audio chunk (generates on demand)
    /// Returns nil when all phrases are done
    public func getNextChunk() async throws -> [Float]? {
        guard generatingIndex < pendingPhrases.count else {
            return nil
        }

        let phrase = pendingPhrases[generatingIndex]
        let audio = try await engine.generateAudio(phrase, temperature: temperature)

        generatingIndex += 1
        audioQueue.append(audio)

        return audio
    }

    /// Check if more chunks are available
    public var hasMoreChunks: Bool {
        return generatingIndex < pendingPhrases.count
    }

    /// Get all phrases (for debugging)
    public var phrases: [String] {
        return pendingPhrases
    }

    /// Get total phrases count
    public var totalPhrases: Int {
        return pendingPhrases.count
    }

    /// Get current progress
    public var progress: (generated: Int, total: Int) {
        return (generatingIndex, pendingPhrases.count)
    }
}

// MARK: - Convenience Extension

extension ChatterboxEngine {

    /// Create a phrase-based streaming wrapper
    public func createPhraseStreaming() -> PhraseStreamingTTS {
        return PhraseStreamingTTS(engine: self)
    }
}
