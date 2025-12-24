# Streaming TTS Guide

This document explains how to use and understand the streaming TTS implementation in Nightingale.

---

## Quick Start

### Build and Run

```bash
# Build release version
swift build -c release --product StreamingTest

# Run the test
.build/release/StreamingTest
```

### What It Does

The streaming test:
1. Generates all speech tokens first (T3 model)
2. Synthesizes audio in chunks with overlap for continuity
3. Measures TTFA (Time-To-First-Audio)
4. Saves and plays the final audio

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Text Input                                │
│              "Hello! This is a streaming test..."               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     T3 Token Generation                          │
│                                                                  │
│  • Generates ALL speech tokens upfront                          │
│  • ~64 tokens/sec on M-series Mac                               │
│  • 142 tokens for test sentence (~2.2s)                         │
│                                                                  │
│  Output: [token₀, token₁, token₂, ... token₁₄₁]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Chunked S3Gen Synthesis                        │
│                                                                  │
│  Chunk 0: tokens[0:20]     → 800ms audio  (first chunk larger)  │
│  Chunk 1: tokens[16:30]    → 400ms audio  (4-token overlap)     │
│  Chunk 2: tokens[26:40]    → 400ms audio  (4-token overlap)     │
│  ...                                                             │
│                                                                  │
│  Each chunk includes 4 overlap tokens from previous chunk       │
│  to maintain encoder context continuity                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Audio Output                                │
│                                                                  │
│  • 24kHz sample rate                                            │
│  • Chunks concatenated with overlap samples skipped             │
│  • Saved to output/streaming_full.wav                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 4-Token Overlap Technique

### Why Overlap?

The S3Gen encoder uses **bidirectional self-attention**. When processing tokens independently per chunk, each chunk lacks context from surrounding tokens, causing:
- Clicking/popping at chunk boundaries
- Inconsistent prosody between chunks
- Audio discontinuities

### How It Works

```
Token Stream:  [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] ...

Chunk 0:       [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
               └─────────── 10 tokens ───────────────┘
               └─────────── Use all audio ───────────┘

Chunk 1:                         [6] [7] [8] [9] [10] [11] [12] [13] [14] [15]
                                 └─ overlap ─┘  └────── new tokens ──────────┘
                                 └─ skip audio ┘ └────── use audio ──────────┘

Chunk 2:                                              [12] [13] [14] [15] [16] ...
                                                      └─ overlap ─┘  └─ new ─┘
```

### Audio Sample Math

- Each token ≈ 40ms of audio
- 40ms × 24,000 Hz = 960 samples per token
- 4 overlap tokens = 3,840 samples to skip

```swift
let overlapTokens = 4
let samplesToSkip = overlapTokens * 960  // 3840 samples

// For chunks after the first:
let usableAudio = Array(chunkAudio[samplesToSkip...])
```

---

## Configuration Parameters

### Chunk Sizes

```swift
let initialChunkSize = 20    // First chunk: 20 tokens (~800ms audio)
let subsequentChunkSize = 10 // Later chunks: 10 tokens (~400ms audio)
let overlapTokens = 4        // Context overlap between chunks
```

| Parameter | Value | Audio Duration | Trade-off |
|-----------|-------|----------------|-----------|
| Initial chunk | 20 tokens | ~800ms | Larger = more buffer, higher TTFA |
| Subsequent | 10 tokens | ~400ms | Smaller = lower latency, more overhead |
| Overlap | 4 tokens | ~160ms | More = better quality, more redundant work |

### Tuning for Your Use Case

**Lower TTFA (faster start):**
```swift
let initialChunkSize = 10   // Start with less audio
let subsequentChunkSize = 8
```

**Better Quality:**
```swift
let overlapTokens = 6       // More context
let initialChunkSize = 30   // Larger first chunk
```

---

## Key Metrics

### Current Performance (M-series Mac)

| Metric | Value | Notes |
|--------|-------|-------|
| **TTFA** | ~2135ms | Time until first audio ready |
| **T3 rate** | 64 tok/s | Token generation speed |
| **S3Gen RTF** | ~2.3x | 1650ms to generate 400ms audio |
| **Audio/token** | 40ms | Each token = 40ms of speech |
| **Margin** | +24.4ms/tok | T3 is 2.5x ahead of real-time |

### Understanding RTF (Real-Time Factor)

```
RTF = Generation Time / Audio Duration

RTF < 1.0 = Faster than real-time (good for streaming)
RTF = 1.0 = Real-time (borderline)
RTF > 1.0 = Slower than real-time (will have gaps)
```

Current S3Gen RTF of 2.3x means it takes 2.3 seconds to generate 1 second of audio.

---

## Code Walkthrough

### Step 1: Generate All Tokens

```swift
// T3 generates speech tokens from text
let t3Start = Date()
let allTokens = try await engine.runT3Only(testText, temperature: 0.0001)
let t3Time = Date().timeIntervalSince(t3Start)

// Result: [1486, 892, 1103, 445, ...] (142 tokens)
```

### Step 2: Process Chunks with Overlap

```swift
var tokenIndex = 0
var chunkNum = 0

while tokenIndex < allTokens.count {
    // Determine chunk boundaries
    let chunkSize = (chunkNum == 0) ? initialChunkSize : subsequentChunkSize
    let endIndex = min(tokenIndex + chunkSize, allTokens.count)

    // Include overlap tokens from previous chunk
    var chunkTokens: [Int]
    if chunkNum == 0 {
        chunkTokens = Array(allTokens[tokenIndex..<endIndex])
    } else {
        let overlapStart = max(0, tokenIndex - overlapTokens)
        chunkTokens = Array(allTokens[overlapStart..<endIndex])
    }

    // Synthesize audio for this chunk
    let chunkAudio = try await engine.synthesizeFromTokens(chunkTokens)

    // Skip overlap samples (each token ≈ 40ms ≈ 960 samples)
    let samplesToSkip = (chunkNum == 0) ? 0 : overlapTokens * 960
    let usableAudio = Array(chunkAudio[samplesToSkip...])

    allAudio.append(contentsOf: usableAudio)
    tokenIndex = endIndex
    chunkNum += 1
}
```

### Step 3: Save and Play

```swift
// Save to WAV file
let fullAudioPath = outputDir.appendingPathComponent("streaming_full.wav")
try saveWAV(samples: allAudio, to: fullAudioPath)

// Play with system audio
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/afplay")
process.arguments = [fullAudioPath.path]
try process.run()
process.waitUntilExit()
```

---

## Output Files

The test generates two files in `output/`:

| File | Description |
|------|-------------|
| `streaming_full.wav` | Complete concatenated audio |
| `streaming_first_chunk.wav` | Just the first chunk (for TTFA testing) |

---

## Streaming Viability Analysis

The test outputs a viability analysis:

```
🔍 Streaming Viability:
   Audio per token: 40.0ms
   T3 time per token: 15.6ms
   Margin: 24.4ms/token
   ✅ VIABLE - Audio plays slower than generation
```

### What This Means

- **Audio per token (40ms)**: How much audio each token produces
- **T3 time per token (15.6ms)**: How long T3 takes to generate one token
- **Margin (24.4ms)**: Buffer we accumulate per token

**Positive margin** = T3 generates tokens faster than audio plays = streaming is viable

---

## Alternative: Phrase-Based Streaming

For simpler streaming without token-level chunking, use `PhraseStreamingTTS`:

```swift
let phraseStreamer = await engine.createPhraseStreaming()

// Start streaming - returns first audio chunk
let (firstChunk, ttfa) = try await phraseStreamer.startStreaming(text)

// Get subsequent chunks
while phraseStreamer.hasMoreChunks {
    if let chunk = try await phraseStreamer.getNextChunk() {
        // Play or buffer chunk
    }
}
```

Phrase streaming splits text at punctuation (`.`, `!`, `?`) and generates complete phrases, avoiding the complexity of token overlap.

---

## Troubleshooting

### Clicking at Chunk Boundaries

**Cause:** Insufficient overlap or vocoder edge artifacts

**Solutions:**
1. Increase overlap tokens: `let overlapTokens = 6`
2. The current implementation trims 512 samples (~21ms) from chunk ends
3. Ensure vocoder context frames are sufficient (32 frames default)

### Audio Gaps During Playback

**Cause:** S3Gen RTF > 1.0 (slower than real-time)

**Solutions:**
1. Increase first chunk size for larger initial buffer
2. Use INT8 quantization for S3Gen (not yet implemented)
3. Reduce ODE steps in S3GenStreaming (`fastOdeSteps = 2`)

### Hallucination/Gibberish

**Cause:** Text too short (< 30 characters)

**Solution:** Ensure minimum text length or use phrase merging

---

## Related Files

| File | Purpose |
|------|---------|
| `test_scripts/StreamingTest/main.swift` | Main streaming test |
| `Sources/Nightingale/S3GenStreaming.swift` | Streaming S3Gen wrapper |
| `Sources/Nightingale/PhraseStreamingTTS.swift` | Phrase-based streaming |
| `Sources/Nightingale/ChatterboxEngine.swift` | Core TTS engine |

---

## Next Steps

See [TTFA_OPTIMIZATION_PLAN.md](../TTFA_OPTIMIZATION_PLAN.md) for planned optimizations to reduce TTFA from 2135ms to ~600ms.

---

*Last Updated: December 24, 2025*
