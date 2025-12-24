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

## Architecture Overview (Optimized)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Text Input                                │
│              "Hello! This is a streaming test..."               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     T3 Token Generation                          │
│  • Generates tokens ~64/sec                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Optimized S3Gen Streaming                      │
│                                                                 │
│  Step 1: Full Context Encoder (High Quality)                    │
│  • Accumulates ALL tokens received so far                       │
│  • Re-encodes full sequence to ensure perfect prosody/context   │
│  • Cost: Low (~10-20ms)                                         │
│                                                                 │
│  Step 2: Windowed Decoder (High Speed)                          │
│  • Identifying active window (e.g. new tokens + 64 overlapping) │
│  • Runs ODE solver ONLY on this window (O(1) complexity)        │
│  • Cost: Low (~200ms) - constant regardless of sentence length  │
│                                                                 │
│  Output: Seamless audio stream                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Windowed Decoding Technique

### Why it's better than simple chunking

1.  **Perfect Prosody**: The Encoder sees the *entire* history, so it knows the intonation contour of the whole sentence so far. Simple chunking breaks this context.
2.  **Constant Speed**: The Decoder (the heavy part) only processes a fixed-size window.
3.  **Seamless Audio**: Overlap is handled internally by the ODE solver's boundary conditions, ensuring zero clicks or pops.

---

## Code Walkthrough

### Step 1: Initialize Streaming

```swift
// Create the optimized streaming wrapper
guard let s3stream = await engine.createStreamingS3Gen() else { ... }

// Initialize with voice conditioning (computed once)
guard let (s3Soul, promptToken, promptFeat) = await engine.getVoiceConditioning(),
      let speechEmbMatrix = await engine.getSpeechEmbMatrix() else { ... }

s3stream.initializeWithPrompt(
    promptTokens: promptToken,
    promptFeat: promptFeat,
    speechEmbMatrix: speechEmbMatrix
)
```

### Step 2: Push Tokens Incremenally

```swift
var tokenIndex = 0

while tokenIndex < allTokens.count {
    // Determine next batch of tokens (e.g. 10 tokens)
    let endIndex = min(tokenIndex + 10, allTokens.count)
    let newTokens = Array(allTokens[tokenIndex..<endIndex])
    
    // Generate audio!
    // s3stream handles all overlap, caching, and windowing internally.
    let chunkAudio = s3stream.generateIncremental(newTokens: newTokens)
    
    allAudio.append(contentsOf: chunkAudio)
    tokenIndex = endIndex
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

---
 
 ## Key Metrics
 
 ### Current Performance (M-series Mac)
 
 | Metric | Value | Notes |
 |--------|-------|-------|
 | **TTFA** | ~815ms | 300ms (T3) + 515ms (S3Gen First Chunk) |
 | **S3Gen RTF** | ~0.80x | 322ms to generate 400ms audio (Streaming) |
 | **Audio/token** | ~96ms | Observed average (varies by voice) |
 | **Margin** | +80ms/tok | Streaming is comfortably faster than playback |
 
 ### Understanding RTF (Real-Time Factor)
 
 ```
 RTF = Generation Time / Audio Duration
 
 RTF < 1.0 = Faster than real-time (good for streaming)
 RTF = 1.0 = Real-time (borderline)
 RTF > 1.0 = Slower than real-time (will have gaps)
 ```
 
 **Previous S3Gen RTF:** 2.3x (Unusable)
 **Optimized S3Gen RTF:** 0.78x (Viable) -> **3x Speedup!**
 
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
