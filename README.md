# Nightingale TTS - Swift/MLX

**High-quality multilingual text-to-speech running entirely on-device with Swift and MLX.**

Pure Swift implementation of Nightingale TTS for Apple Silicon Macs, iPhones, and iPads. Generate natural-sounding speech in multiple languages without any Python dependencies.

## Features

- **100% Swift/MLX** - No Python runtime required for inference
- **On-Device Processing** - All computation happens locally on Apple Silicon
- **High Quality** - 24kHz audio output with natural prosody
- **Multilingual** - Supports English, Dutch, and many other languages
- **Fast** - Optimized for Apple Silicon with MLX
- **Multiple Voices** - Includes Samantha (female) and Sujano (male) voices

## Requirements

- macOS 14.0+ or iOS 17.0+
- Apple Silicon (M1/M2/M3/M4) or A-series chip
- Swift 5.9+
- Xcode 15.0+

## Quick Start

### 1. Build the Package

```bash
swift build -c release
```

### 2. Generate Audio

```bash
.build/release/GenerateAudio "Hello! Welcome to Nightingale TTS."
```

The audio file will be saved to `test_audio/output.wav`.

### 3. Try Different Voices

```bash
# Use Samantha voice (female, default)
.build/release/GenerateAudio --voice samantha "This is Samantha speaking."

# Use Sujano voice (male)
.build/release/GenerateAudio --voice sujano "This is Sujano speaking."
```

## Generating Test Files

To generate a test .wav file using the Swift TTS pipeline:

```bash
swift run -c release GenerateAudio
```

This runs the [GenerateAudio](test_scripts/GenerateAudio/main.swift) script which:
- Loads the Chatterbox models from `models/chatterbox/`
- Loads the Sujano voice from `baked_voices/sujano/`
- Generates speech for the test sentence: *"Wow! I absolutely cannot believe that it worked on the first try!"*
- Saves the output to `test_audio/chatterbox_engine_test.wav`

The generated file will be:
- **Duration**: ~2.3 seconds
- **Sample Rate**: 24kHz (24,000 Hz)
- **Channels**: Mono
- **Format**: 16-bit PCM WAV

The script uses `temperature=0.0001` (near-deterministic) to match the cross-validation testing parameters. This ensures the output is consistent and can be compared with Python reference implementations.

**Note:** The test text matches the one used in `E2E/cross_validation.md` for verifying Python/Swift parity.

## API Usage

### Basic Usage

```swift
import Nightingale

let engine = ChatterboxEngine()

// Load models (one-time setup)
let modelsURL = URL(fileURLWithPath: "models/chatterbox")
try await engine.loadModels(modelsURL: modelsURL)

// Load voice
let voicesURL = URL(fileURLWithPath: "baked_voices")
try await engine.loadVoice("samantha", voicesURL: voicesURL)

// Generate speech
let audio = try await engine.generateAudio(
    "Hello! This is a test.",
    temperature: 0.667
)

// audio is [Float] at 24kHz sample rate
```

### Advanced: Token-Level Control

```swift
// Generate speech tokens only (for caching or manipulation)
let tokens = try engine.runT3Only("Hello world!", temperature: 0.667)

// Later: Synthesize audio from tokens
let audio = try engine.synthesizeFromTokens(tokens)
```

## Project Structure

```
nightingale_Swift/
├── Package.swift              # Swift package definition
├── Sources/
│   └── Nightingale/          # Main library
│       ├── ChatterboxEngine.swift
│       ├── T3.swift          # Text-to-tokens encoder
│       ├── S3Gen.swift       # Token-to-mel decoder
│       └── Vocoder.swift     # Mel-to-audio synthesis
├── test_scripts/
│   ├── GenerateAudio/        # Command-line tool
│   ├── CrossValidate/        # Verification tool
│   └── GenerateTestSentences/ # Batch generation
├── models/
│   └── chatterbox/           # Model weights (~1GB)
├── baked_voices/
│   ├── samantha/             # Female voice
│   └── sujano/               # Male voice
└── E2E/
    ├── test_sentences.json   # Test sentences
    └── cross_validation.md   # Testing documentation
```

## Executables

### GenerateAudio

Generate speech from text input.

```bash
# Basic usage
.build/release/GenerateAudio "Your text here"

# With options
.build/release/GenerateAudio \
  --voice samantha \
  --temperature 0.8 \
  --output my_audio.wav \
  "Your text here"
```

### GenerateTestSentences

Generate a comprehensive test suite across multiple voices and languages.

```bash
.build/release/GenerateTestSentences
```

Generates audio for all test sentences in `E2E/test_sentences.json` in both English and Dutch.

### CrossValidate

Verify Swift implementation against Python reference (requires Python installation).

```bash
.build/release/CrossValidate
```

## Configuration

### Voice Parameters

Voices are pre-computed speaker embeddings stored in `baked_voices/`:

- **samantha** - Clear female voice, neutral American English accent
- **sujano** - Warm male voice, slight accent

Each voice includes:
- `baked_voice.safetensors` - Speaker embedding
- `ref_audio.wav` - Reference audio sample
- `tuning/` - Example outputs with different parameters

### Generation Parameters

```swift
engine.generateAudio(
    text,
    temperature: 0.667,      // 0.0-1.0: Higher = more variation
    cfgStrength: 0.5,        // 0.0-1.0: Classifier-free guidance
    exaggeration: 0.5        // 0.0-1.0: Prosody emphasis
)
```

- **temperature**: Controls randomness in token generation
  - `0.667` - Default, natural variation
  - `0.001` - Nearly deterministic (for testing)
  - `1.0` - Maximum variation

- **cfgStrength**: Guidance strength for text conditioning
  - `0.5` - Default, balanced
  - `0.3` - Less guided, more expressive
  - `0.7` - More guided, more controlled

- **exaggeration**: Prosody and emphasis
  - `0.5` - Default, natural
  - `0.3` - Flatter, more neutral
  - `0.7` - More expressive, emphasized

## Performance

On Apple M3 Max:

- **Model Loading**: ~2 seconds (one-time)
- **Voice Loading**: ~0.1 seconds (per voice)
- **Generation**: ~1.5-2.5 seconds for typical sentence
- **Real-time Factor**: ~0.3-0.5x (much faster than real-time)

## Verification

The Swift implementation has been thoroughly validated:

- ✅ **100% Token Match** - T3 encoder produces identical tokens to Python
- ✅ **Perfect Audio Correlation** - 1.0 correlation with deterministic noise
- ✅ **Cross-Platform Tested** - Verified on M1, M2, M3, M4 Macs

See `E2E/cross_validation.md` for detailed test results.

## Troubleshooting

### Build Issues

```bash
# Clean build
rm -rf .build
swift build -c release
```

### Model Loading Errors

Ensure model files are present:

```bash
ls models/chatterbox/
# Should show: s3gen.safetensors, t3_mtl23ls_v2.safetensors, etc.
```

### Audio Quality Issues

- Use lower temperature (0.3-0.6) for clearer speech
- Adjust exaggeration (0.3-0.7) for different styles
- Try different voices (samantha vs sujano)

## Architecture

Nightingale uses a three-stage pipeline:

1. **T3 Encoder** (Text → Tokens)
   - Transformer-based text encoder
   - Outputs discrete speech tokens
   - Handles multilingual input

2. **S3Gen Decoder** (Tokens → Mel Spectrogram)
   - Conformer-based decoder
   - Uses flow matching for high quality
   - Guided by speaker embeddings

3. **Vocoder** (Mel → Waveform)
   - ResBlock-based neural vocoder
   - Converts mel spectrogram to 24kHz audio
   - Trained for natural sound quality

## Integration

### iOS App Integration

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "path/to/nightingale_Swift", from: "1.0.0")
]
```

Include model files in your app bundle and reference them:

```swift
let modelsURL = Bundle.main.url(forResource: "chatterbox", withExtension: nil)!
let voicesURL = Bundle.main.url(forResource: "baked_voices", withExtension: nil)!
```

### macOS App Integration

Same as iOS - add the package dependency and bundle the model files with your app.

## License

See the main Nightingale TTS repository for license information.

## Credits

Swift/MLX implementation based on the original Chatterbox TTS model by Resemble AI.

MLX framework by Apple ML Research.
