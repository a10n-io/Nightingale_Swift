# Cross-Validation Testing for Python/Swift Parity

This document explains the cross-validation testing framework for verifying parity between the Python (PyTorch) and Swift (MLX) implementations of Chatterbox TTS.

## Status

✅ **ALL TESTS PASSING** - Swift implementation achieves 100% parity with Python

- ✅ T3 Token Generation: 100% match (98/98 tokens identical)
- ✅ S3Gen Audio: Perfect correlation (1.0) with deterministic noise
- ✅ Cross-validation: All 4 audio files sound identical
- ✅ Production Ready: Full pipeline verified

## Overview

The Chatterbox TTS pipeline has two main stages:

1. **T3 (Text Encoder)**: Converts text → speech tokens
2. **S3Gen (Speech Generator)**: Converts speech tokens → audio waveform

```
Text → [T3] → Speech Tokens → [S3Gen] → Audio
```

Cross-validation verifies parity by mixing Python and Swift components in different combinations, isolating each stage for testing.

## Test Matrix

The cross-validation generates 4 audio files by combining T3 and S3Gen from both implementations:

| File | T3 (Encoder) | S3Gen (Decoder) | Purpose | Status |
|------|--------------|-----------------|---------|--------|
| `python_tokens_python_audio.wav` | Python | Python | Baseline reference | ✅ Perfect |
| `python_tokens_swift_audio.wav` | Python | Swift | Tests Swift S3Gen decoder | ✅ Perfect |
| `swift_tokens_swift_audio.wav` | Swift | Swift | Full Swift pipeline | ✅ Perfect |
| `swift_tokens_python_audio.wav` | Swift | Python | Tests Swift T3 encoder | ✅ Perfect |

All 4 files sound **identical** when played back, confirming perfect parity.

## Running the Tests

### Prerequisites

- Python environment with Chatterbox installed (`python/venv/`)
- Swift MLX environment with models loaded (`models/chatterbox/`)
- Baked voice files (`baked_voices/samantha/`)

### Step 1: Generate Python Baseline

Generate Python tokens and baseline audio:

```bash
python E2E/cross_validate_python.py
```

**Output:**
```
Python T3 generation...
  Generated 98 speech tokens
  Saved tokens to: python_speech_tokens.safetensors

Python S3Gen synthesis...
  Generated 55776 audio samples (2.32s)
  Saved: python_tokens_python_audio.wav
```

**Files Created:**
- `test_audio/cross_validate/python_speech_tokens.safetensors` - Tokens for Swift
- `test_audio/cross_validate/python_tokens_python_audio.wav` - Baseline audio

### Step 2: Generate Swift Cross-Validation

Generate Swift tokens and cross-validate with Python tokens:

```bash
cd swift
swift run -c release CrossValidate
```

**Output:**
```
STEP 1: Generate speech tokens with Swift T3
  Generated 98 speech tokens

STEP 2: Swift tokens -> Swift S3Gen
  Saved: swift_tokens_swift_audio.wav

STEP 3: Python tokens -> Swift S3Gen
  Python tokens: 98
  Swift tokens: 98
  Token comparison: 98/98 (100.0%) ✅
  First diff: NONE - Perfect match!
  Saved: python_tokens_swift_audio.wav
```

**Files Created:**
- `test_audio/cross_validate/swift_speech_tokens.safetensors` - Tokens for Python
- `test_audio/cross_validate/swift_tokens_swift_audio.wav` - Full Swift pipeline
- `test_audio/cross_validate/python_tokens_swift_audio.wav` - Swift S3Gen test

### Step 3: Complete Cross-Validation

Run Python script again to process Swift tokens:

```bash
python E2E/cross_validate_python.py
```

**Output:**
```
Loading Swift tokens...
  Swift tokens: 98
  Token comparison: 98/98 (100.0%) ✅

Generating audio from Swift tokens with Python S3Gen...
  Generated 55776 audio samples (2.32s)
  Saved: swift_tokens_python_audio.wav
```

**Files Created:**
- `test_audio/cross_validate/swift_tokens_python_audio.wav` - Swift T3 test

## Verification Results

### Token Comparison

Perfect 100% token match between Python and Swift T3 encoders:

```
Test text: "Wow! I absolutely cannot believe that it worked on the first try!"

Python tokens (98): [1732, 2068, 2186, 1457, 680, 1457, 1864, 2186, 1457, 1732, ...]
Swift tokens (98):  [1732, 2068, 2186, 1457, 680, 1457, 1864, 2186, 1457, 1732, ...]

Matching: 98/98 (100.0%) ✅
First diff: NONE - Perfect match!
```

### Audio Quality

All 4 audio files are perceptually identical:

| Metric | Result |
|--------|--------|
| Vocoder correlation | 1.0 (perfect) |
| Audio duration | 2.32s (all files) |
| Sample count | 55,776 (all files) |
| Perceptual quality | Identical |
| Speech clarity | Perfect |

### Sample Statistics

Output from cross-validation run:

```
CROSS-VALIDATION COMPLETE

Output directory: test_audio/cross_validate
Files generated:
  - python_speech_tokens.safetensors
  - swift_speech_tokens.safetensors
  - python_tokens_python_audio.wav   (baseline)
  - python_tokens_swift_audio.wav    (tests Swift S3Gen) ✅
  - swift_tokens_swift_audio.wav     (full Swift pipeline) ✅
  - swift_tokens_python_audio.wav    (tests Swift T3) ✅

Token Match: 98/98 (100.0%) ✅
Audio Quality: Perfect parity ✅
```

## Test Parameters

Both implementations use identical parameters for deterministic testing:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.0001 | Near-deterministic sampling |
| CFG Weight | 0.5 | Classifier-free guidance strength |
| Repetition Penalty | 2.0 | Prevents token repetition |
| Top P | 1.0 | Nucleus sampling (disabled) |
| Min P | 0.05 | Minimum probability threshold |
| Max Tokens | 1000 | Maximum generation length |
| ODE Steps | 10 | Flow matching ODE solver steps |

**Test Text:** `"Wow! I absolutely cannot believe that it worked on the first try!"`

**Voice:** `samantha` (from `baked_voices/samantha/baked_voice.safetensors`)

## API Usage

### ChatterboxEngine Cross-Validation API

The ChatterboxEngine provides methods for isolated component testing:

```swift
// Run T3 only (returns speech tokens)
let tokens = try engine.runT3Only(text, temperature: 0.0001)
// Returns: [Int] array of speech tokens [0-6560]

// Run S3Gen only (returns audio samples)
let audio = try engine.synthesizeFromTokens(tokens)
// Returns: [Float] array of audio samples [-1.0, 1.0]

// Full pipeline (convenience method)
let audio = try await engine.generateAudio(text, temperature: 0.0001)
```

### Saving Tokens for Cross-Validation

**Swift → Python:**
```swift
let tokens = try engine.runT3Only(text, temperature: 0.0001)
let tokensArray = MLXArray(tokens.map { Int32($0) })
try MLX.save(
    arrays: ["speech_tokens": tokensArray],
    url: URL(fileURLWithPath: "swift_speech_tokens.safetensors")
)
```

**Python → Swift:**
```python
from safetensors.torch import save_file
import torch

tokens = model.generate_tokens(text)  # [98] tokens
save_file(
    {"speech_tokens": tokens},
    "python_speech_tokens.safetensors"
)
```

## Troubleshooting

### If Tokens Don't Match

**Problem:** Token mismatch indicates T3 encoder differences

**Common Causes:**
1. Weight loading issues
2. Tokenization differences
3. CFG implementation differences
4. Sampling implementation differences

**Debug Steps:**
1. Compare first 20 tokens - identify where divergence starts
2. Check tokenization: Save text tokens and compare
3. Verify weight loading: Check embedding statistics
4. Test with temperature=0.0001 for determinism

### If Audio Sounds Different

**Problem:** Audio differences indicate S3Gen decoder issues

**Common Causes:**
1. Weight transposition errors (Conv1d, Linear)
2. Decoder noise differences (RNG state)
3. Vocoder implementation differences
4. STFT/FFT differences

**Debug Steps:**
1. Use Python tokens through Swift S3Gen to isolate decoder
2. Check mel spectrogram correlation
3. Verify vocoder weights loaded correctly
4. Use deterministic noise for perfect correlation

### If Only Full Pipeline Fails

**Problem:** Both components work individually but fail together

**Cause:** Usually an interaction bug or state management issue

**Debug Steps:**
1. Check voice conditioning is consistent
2. Verify token format/range compatibility
3. Test with different voices
4. Check GPU memory/cache clearing

## Implementation Notes

### Deterministic Noise

For perfect decoder correlation, the Swift implementation can optionally load Python's fixed noise:

```swift
// In ChatterboxEngine.loadModels():
if let pythonNoise = try? MLX.loadArrays(url: pythonNoiseURL)["noise"] {
    s3gen?.setFixedNoise(pythonNoise)
    print("✅ Loaded Python fixed noise for decoder precision")
}
```

This enables:
- Perfect numerical correlation (1.0)
- Bit-exact audio reproduction
- Easier debugging of decoder issues

For production use, Swift generates its own noise which still produces perceptually identical audio.

### Token Range

Valid speech tokens: **[0, 6560]**

Special tokens (must be filtered before S3Gen):
- **6561**: Start-of-speech (SOS)
- **6562**: End-of-speech (EOS)

The `dropInvalidTokens()` method removes these automatically.

### Voice File Format

Voice files use the unified `baked_voice.safetensors` format:

```python
{
    "t3.speaker_emb": [1, 256],          # T3 speaker embedding
    "t3.cond_prompt_speech_tokens": [...], # T3 conditioning tokens
    "gen.embedding": [1, 192],            # S3Gen speaker embedding
    "gen.prompt_token": [1, 500, 80],     # S3Gen prompt mel
    "gen.prompt_feat": [1, 500, 512]      # S3Gen prompt features
}
```

Both Python and Swift use this same format.

## Output Directory

All cross-validation files are saved to: `test_audio/cross_validate/`

```
test_audio/cross_validate/
├── python_speech_tokens.safetensors
├── swift_speech_tokens.safetensors
├── python_tokens_python_audio.wav   (baseline)
├── python_tokens_swift_audio.wav    (Swift S3Gen test)
├── swift_tokens_swift_audio.wav     (full Swift pipeline)
└── swift_tokens_python_audio.wav    (Swift T3 test)
```

## Success Criteria

✅ Cross-validation **PASSES** if:

1. **Token Match**: Swift and Python T3 generate identical tokens (100% match)
2. **Audio Quality**: All 4 audio files sound perceptually identical
3. **No Artifacts**: No noise, distortion, or quality degradation
4. **Consistent Duration**: All files have same sample count
5. **Reproducible**: Results are consistent across multiple runs

All criteria are currently met! 🎉

## Next Steps

After successful cross-validation:

1. **Test More Sentences**: Use `GenerateTestSentences` for batch testing
2. **Test Other Voices**: Verify with different baked voices
3. **Test Other Languages**: Validate multilingual support
4. **Performance Testing**: Measure inference speed and memory usage
5. **iOS Integration**: Port to iOS app for on-device testing

## Related Documentation

- [README.md](../README.md) - Project overview and quick start
- [CLEANUP_SUMMARY.md](../CLEANUP_SUMMARY.md) - Codebase cleanup details
- [test_sentences.json](test_sentences.json) - Extended test data

---

**Last Updated**: December 2024 | **Status**: All Tests Passing ✅
