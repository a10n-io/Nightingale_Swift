# Nightingale TTS - Swift/MLX

**High-quality streaming text-to-speech running entirely on-device with Swift and MLX.**

Pure Swift implementation of Nightingale TTS for Apple Silicon Macs, iPhones, and iPads. Generate natural-sounding speech with low-latency streaming - no Python dependencies required.

---

## Streaming TTS

Nightingale supports **chunked streaming synthesis** for low-latency audio playback. Generate audio in chunks while tokens are still being produced.

### Quick Start - Streaming

```bash
# Build
swift build -c release --product StreamingTest

# Run streaming test
.build/release/StreamingTest
```

### Current Performance

| Metric | Value |
|--------|-------|
| **TTFA (Time-To-First-Audio)** | ~2.1s |
| **Token Generation** | 64 tok/s (2.5x faster than real-time) |
| **Streaming Viable** | Yes - positive 24ms/token margin |

### How Streaming Works

```
Text Input → T3 (tokens) → Chunked S3Gen → Audio Chunks → Playback
                              ↓
                    4-token overlap for
                    seamless boundaries
```

1. **T3** generates all speech tokens first (~64 tokens/sec)
2. **S3Gen** processes tokens in chunks with 4-token overlap
3. Audio chunks are concatenated with overlap samples skipped
4. Result: seamless audio without clicking at boundaries

See [docs/STREAMING_TTS.md](docs/STREAMING_TTS.md) for complete streaming documentation.

---

## Features

- **Streaming Synthesis** - Low-latency chunked audio generation
- **100% Swift/MLX** - No Python runtime required
- **On-Device** - All computation on Apple Silicon
- **High Quality** - 24kHz audio with natural prosody
- **Multilingual** - English, Dutch, and more (2454 token vocab)
- **INT8 Quantized** - Fast T3 inference with minimal quality loss
- **Multiple Voices** - Samantha (female) and Sujano (male)

## Requirements

- macOS 14.0+ or iOS 17.0+
- Apple Silicon (M1/M2/M3/M4/M5) or A-series chip
- Swift 5.9+
- Xcode 15.0+

---

## Installation

### 1. Clone and Build

```bash
git clone <repository-url>
cd Nightingale_Swift

# Build all targets
swift build -c release
```

### 2. Download Models

Place model files in `models/chatterbox/`:
- `t3_mtl23ls_v2_int8.safetensors` (INT8 quantized T3)
- `s3gen.safetensors` (S3Gen decoder + vocoder)

### 3. Add Voice Files

Place voice embeddings in `baked_voices/<voice_name>/`:
- `baked_voice.safetensors`

---

## Usage

### Command Line

```bash
# Basic generation
.build/release/GenerateAudio "Hello! Welcome to Nightingale."

# With options
.build/release/GenerateAudio --voice sujano --temperature 0.5 "Your text here"

# Streaming test
.build/release/StreamingTest
```

### Swift API

```swift
import Nightingale

// Initialize engine
let engine = ChatterboxEngine()

// Load models with INT8 quantization
try await engine.loadModels(
    modelsURL: URL(fileURLWithPath: "models/chatterbox"),
    useQuantization: true
)

// Load voice
try await engine.loadVoice("sujano", voicesURL: URL(fileURLWithPath: "baked_voices"))

// Generate audio
let audio = try await engine.generateAudio("Hello world!", temperature: 0.5)
// audio is [Float] at 24kHz
```

### Streaming API

```swift
// Generate tokens first
let tokens = try await engine.runT3Only(text, temperature: 0.5)

// Synthesize in chunks with overlap
let chunkSize = 10
let overlap = 4

for chunk in tokens.chunked(size: chunkSize, overlap: overlap) {
    let audio = try await engine.synthesizeFromTokens(chunk)
    // Skip overlap samples and play
    playAudio(audio.dropFirst(overlap * 960))
}
```

### Phrase-Based Streaming

For simpler streaming at sentence boundaries:

```swift
let streamer = await engine.createPhraseStreaming()
let (firstChunk, ttfa) = try await streamer.startStreaming(text)

while streamer.hasMoreChunks {
    if let chunk = try await streamer.getNextChunk() {
        playAudio(chunk)
    }
}
```

---

## Available Executables

| Executable | Purpose |
|------------|---------|
| `GenerateAudio` | Basic text-to-speech |
| `StreamingTest` | Streaming synthesis benchmark |
| `PhraseStreamTest` | Phrase-based streaming test |
| `TTSServerQuantized` | HTTP TTS server with INT8 |
| `QuickBench` | Performance benchmarking |
| `CrossValidate` | Python parity verification |

Build specific targets:
```bash
swift build -c release --product StreamingTest
```

---

## Project Structure

```
Nightingale_Swift/
├── Sources/Nightingale/
│   ├── ChatterboxEngine.swift    # Main TTS engine
│   ├── T3Model.swift             # Text-to-tokens (INT8)
│   ├── S3Gen.swift               # Tokens-to-mel decoder
│   ├── S3GenStreaming.swift      # Streaming wrapper
│   ├── PhraseStreamingTTS.swift  # Phrase-based streaming
│   └── Vocoder.swift             # Mel-to-audio
├── test_scripts/
│   ├── StreamingTest/            # Streaming benchmark
│   ├── GenerateAudio/            # CLI tool
│   └── TTSServerQuantized/       # HTTP server
├── docs/
│   └── STREAMING_TTS.md          # Streaming documentation
├── models/chatterbox/            # Model weights
└── baked_voices/                 # Voice embeddings
```

---

## Performance Optimization

See [TTFA_OPTIMIZATION_PLAN.md](TTFA_OPTIMIZATION_PLAN.md) for the optimization roadmap.

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| T3 INT8 | Done | 17.7% speedup |
| Debug cleanup | Done | Removed 300+ debug lines |
| 4-token overlap | Done | Click-free streaming |
| S3Gen INT8 | Planned | Est. 600-900ms savings |

### Target Performance

| Metric | Current | Target |
|--------|---------|--------|
| TTFA | 2135ms | 600ms |
| T3 Rate | 64 tok/s | 80 tok/s |
| S3Gen RTF | 2.3x | 1.0x |

---

## Configuration

### Generation Parameters

```swift
engine.generateAudio(
    text,
    temperature: 0.5,    // 0.0-1.0: randomness
    cfgStrength: 0.5,    // 0.0-1.0: text guidance
    exaggeration: 0.5    // 0.0-1.0: prosody emphasis
)
```

### Streaming Parameters

```swift
// In StreamingTest
let initialChunkSize = 20    // First chunk tokens
let subsequentChunkSize = 10 // Later chunks
let overlapTokens = 4        // Context overlap
```

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│     T3       │     │    S3Gen     │     │   Vocoder    │
│  (INT8)      │────▶│   (ODE)      │────▶│  (HiFi-GAN)  │
│ Text→Tokens  │     │ Tokens→Mel   │     │  Mel→Audio   │
└──────────────┘     └──────────────┘     └──────────────┘
    ~64 tok/s           4 ODE steps         24kHz output
```

- **T3**: 30-layer transformer, INT8 quantized, 2454 token vocab
- **S3Gen**: Flow matching decoder with 4 ODE steps (fast mode)
- **Vocoder**: Neural vocoder producing 24kHz audio

---

## Verification

The Swift implementation is validated against Python:

- 100% token match with Python T3
- 1.0 correlation with deterministic noise
- Cross-platform tested on M1-M5 Macs

See `E2E/cross_validation.md` for test results.

---

## Troubleshooting

### Clicking at Chunk Boundaries

Increase overlap tokens or use phrase-based streaming:
```swift
let overlapTokens = 6  // Default is 4
```

### Slow Generation

Enable INT8 quantization:
```swift
try await engine.loadModels(modelsURL: url, useQuantization: true)
```

### Model Loading Errors

```bash
ls models/chatterbox/
# Should show: s3gen.safetensors, t3_mtl23ls_v2_int8.safetensors
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/STREAMING_TTS.md](docs/STREAMING_TTS.md) | Streaming implementation guide |
| [TTFA_OPTIMIZATION_PLAN.md](TTFA_OPTIMIZATION_PLAN.md) | Performance optimization roadmap |
| [CURRENT_AUDIT_FINDINGS.md](CURRENT_AUDIT_FINDINGS.md) | Code audit and cleanup log |
| [E2E/cross_validation.md](E2E/cross_validation.md) | Python parity testing |

---

## License

See the main Nightingale TTS repository for license information.

## Credits

- Swift/MLX implementation based on Chatterbox TTS by Resemble AI
- MLX framework by Apple ML Research
