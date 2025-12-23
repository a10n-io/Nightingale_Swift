import Foundation

/// T3 Model Configuration
/// Matches the config.json from mlx-community/Chatterbox-TTS-fp16
public struct T3Config: Codable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let hiddenAct: String
    public let speakerEmbedSize: Int
    public let textVocabSize: Int
    public let speechVocabSize: Int
    public let maxTextTokens: Int
    public let maxSpeechTokens: Int
    public let usePerceiverResampler: Bool
    public let emotionAdv: Bool
    public let ropeScaling: RopeScaling?

    // Special tokens for text input (matches Python t3_config.py)
    public let startTextToken: Int  // SOT = 255
    public let stopTextToken: Int   // EOT = 0

    public struct RopeScaling: Codable {
        public let factor: Float
        public let highFreqFactor: Float
        public let lowFreqFactor: Float
        public let originalMaxPositionEmbeddings: Int
        public let ropeType: String

        enum CodingKeys: String, CodingKey {
            case factor
            case highFreqFactor = "high_freq_factor"
            case lowFreqFactor = "low_freq_factor"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case ropeType = "rope_type"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case hiddenAct = "hidden_act"
        case speakerEmbedSize = "speaker_embed_size"
        case textVocabSize = "text_vocab_size"
        case speechVocabSize = "speech_vocab_size"
        case maxTextTokens = "max_text_tokens"
        case maxSpeechTokens = "max_speech_tokens"
        case usePerceiverResampler = "use_perceiver_resampler"
        case emotionAdv = "emotion_adv"
        case ropeScaling = "rope_scaling"
        case startTextToken = "start_text_token"
        case stopTextToken = "stop_text_token"
    }

    /// Default T3 configuration (English-only, 704 vocab)
    /// Matches Python's T3Config.english_only()
    public static var `default`: T3Config {
        T3Config(
            modelType: "chatterbox_t3",
            hiddenSize: 1024,
            intermediateSize: 4096,
            numHiddenLayers: 30,
            numAttentionHeads: 16,
            numKeyValueHeads: 16,
            headDim: 64,
            rmsNormEps: 1e-5,
            ropeTheta: 500000.0,
            hiddenAct: "silu",
            speakerEmbedSize: 256,
            textVocabSize: 704,
            speechVocabSize: 8194,
            maxTextTokens: 2048,
            maxSpeechTokens: 4096,
            usePerceiverResampler: true,
            emotionAdv: true,
            ropeScaling: RopeScaling(
                factor: 8.0,
                highFreqFactor: 4.0,
                lowFreqFactor: 1.0,
                originalMaxPositionEmbeddings: 8192,
                ropeType: "llama3"
            ),
            startTextToken: 255,  // SOT token
            stopTextToken: 0      // EOT token
        )
    }

    /// Multilingual T3 configuration (2454 vocab)
    /// Matches Python's T3Config.multilingual()
    public static func multilingual() -> T3Config {
        T3Config(
            modelType: "chatterbox_t3_multilingual",
            hiddenSize: 1024,
            intermediateSize: 4096,
            numHiddenLayers: 30,
            numAttentionHeads: 16,
            numKeyValueHeads: 16,
            headDim: 64,
            rmsNormEps: 1e-5,
            ropeTheta: 500000.0,
            hiddenAct: "silu",
            speakerEmbedSize: 256,
            textVocabSize: 2454,  // Multilingual vocab size
            speechVocabSize: 8194,
            maxTextTokens: 2048,
            maxSpeechTokens: 4096,
            usePerceiverResampler: true,
            emotionAdv: true,
            ropeScaling: RopeScaling(
                factor: 8.0,
                highFreqFactor: 4.0,
                lowFreqFactor: 1.0,
                originalMaxPositionEmbeddings: 8192,
                ropeType: "llama3"
            ),
            startTextToken: 255,  // SOT token
            stopTextToken: 0      // EOT token
        )
    }

    /// Check if this is a multilingual config
    public var isMultilingual: Bool {
        textVocabSize == 2454
    }
}
