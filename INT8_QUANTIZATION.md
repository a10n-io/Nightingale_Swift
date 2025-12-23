# INT8 Quantization Implementation
**Date:** December 23, 2025

## Overview

Implemented INT8 quantization using MLX's native quantization support to achieve 1.5-2x speedup with minimal quality degradation (<1%).

### Expected Performance

| Metric | Before (FP16) | After (INT8) | Improvement |
|--------|---------------|--------------|-------------|
| **T3 Generation** | ~3.96s | ~2.0-2.6s | 1.5-2x faster |
| **Memory Usage** | 100% | ~50% | 50% reduction |
| **Quality Loss** | 0% | <1% | Minimal |
| **Total RTF** | 1.70x | ~0.85-1.13x | ~2x faster |

---

## Implementation Details

### 1. What Was Quantized

**T3Model (Main Bottleneck):**
- 30 transformer layers
  - Attention: Q/K/V/O projections (4 Linear layers × 30 = 120 layers)
  - MLP: Gate/Up/Down projections (3 Linear layers × 30 = 90 layers)
- Output head: speechHead (Linear)
- Conditioning: speakerProj, emotionAdvFC (Linear)

**S3Gen:**
- Encoder: 6 + 4 Conformer blocks
- Decoder: 12 UNet blocks with attention
- All Linear layers quantized to INT8

### 2. Quantization Parameters

```swift
MLXNN.quantize(
    model: model,
    groupSize: 64,    // Elements sharing same scale/bias
    bits: 8,          // INT8 quantization
    mode: .affine     // Standard affine quantization
)
```

**groupSize: 64** - Good balance between compression and accuracy
**bits: 8** - INT8 (vs INT4 which has ~3% quality loss)
**mode: .affine** - Uses scale and bias for better precision

### 3. How It Works

**Quantization Formula:**
```
quantized_value = round((fp16_value - bias) / scale)
dequantized_value = quantized_value * scale + bias
```

**Memory Layout:**
- FP16 weights: 2 bytes per parameter
- INT8 weights: 1 byte per parameter (packed)
- Scales: 1 float per group of 64 elements
- Biases: 1 float per group of 64 elements

**Runtime Behavior:**
- Quantized weights stored on GPU
- Matrix multiplication uses optimized Metal kernels
- Automatic dequantization during forward pass
- Zero overhead for activation computation

### 4. Code Changes

**ChatterboxEngine.swift:**
```swift
// Added useQuantization parameter
public func loadModels(
    from bundle: Bundle = .main,
    modelsURL: URL? = nil,
    useQuantization: Bool = false  // NEW
) async throws {
    // ... load models ...

    // Apply INT8 quantization if requested
    if useQuantization {
        print("\n⚡️ Applying INT8 quantization...")

        // Quantize T3Model
        if let t3 = self.t3 {
            MLXNN.quantize(model: t3, groupSize: 64, bits: 8)
        }

        // Quantize S3Gen
        if let s3 = self.s3gen {
            MLXNN.quantize(model: s3, groupSize: 64, bits: 8)
        }

        print("⚡️ Quantization complete")
    }
}
```

**New Test Script:**
- `test_scripts/TTSServerQuantized/main.swift`
- Enables quantization: `loadModels(useQuantization: true)`
- Outputs to `test_audio/quantized_test_*.wav`

---

## Testing Instructions

### 1. Build the Quantized Test

```bash
swift build -c release
```

### 2. Run Quantized Test

```bash
swift run -c release TTSServerQuantized
```

**Expected Output:**
```
🚀 Nightingale TTS Server (INT8 Quantized)
⏳ Loading models (with INT8 quantization)...
⚡️ INT8 quantization enabled (1.5-2x speedup expected)
...
⚡️ Applying INT8 quantization...
  Quantizing T3Model (30 transformer layers)...
  ✅ T3Model quantized
  Quantizing S3Gen (encoder + decoder)...
  ✅ S3Gen quantized
⚡️ Quantization complete in X.XXs
   Memory: ~50% reduction, Speed: 1.5-2x faster expected
✅ Models loaded and quantized in X.XXs

[1/3] Generating: "Wow! I absolutely cannot believe..."
⏱️  T3 token generation: ~2.0-2.6s (was ~3.96s)
⏱️  S3Gen audio synthesis: ~0.3-0.4s (was ~0.59s)
  ⏱️  Time: ~2.5-3.2s (RTF: ~0.85-1.13x)
```

### 3. Compare Results

**Baseline (FP16):**
```bash
swift run -c release TTSServer
```

**Quantized (INT8):**
```bash
swift run -c release TTSServerQuantized
```

**Compare Audio Files:**
```bash
# Listen to both and compare quality
open test_audio/server_test_1.wav          # FP16
open test_audio/quantized_test_1.wav       # INT8

# Should sound nearly identical (<1% quality difference)
```

### 4. Performance Metrics to Check

| Metric | Expected Result |
|--------|----------------|
| **T3 Time** | 3.96s → 2.0-2.6s (1.5-2x faster) |
| **S3Gen Time** | 0.59s → 0.3-0.4s (1.5-2x faster) |
| **Total Time** | 4.55s → 2.3-3.0s (1.5-2x faster) |
| **RTF** | 1.70x → 0.85-1.13x (2x improvement) |
| **Audio Quality** | <1% degradation (perceptual) |
| **File Size** | Identical (189KB for sentence 1) |

---

## Quality Validation

### 1. Automated Checks

The quantized audio should match the FP16 baseline:
- ✅ Same sample count (e.g., 96,956 samples)
- ✅ Same file size (e.g., 189KB)
- ✅ Same frequency distribution (99.6% low freq)
- ✅ Similar waveform shape

### 2. Perceptual Testing

**Listen Test:**
1. Open both FP16 and INT8 audio files
2. Listen for any noticeable differences
3. Expected: Nearly identical, <1% perceptual difference

**What to Listen For:**
- Voice quality (timbre, tone)
- Pronunciation clarity
- Background noise/artifacts
- Prosody (rhythm, intonation)

**Expected Result:** Should be indistinguishable or barely noticeable difference.

### 3. If Quality Is Degraded

**Possible Solutions:**
1. **Increase group size** to 128 (better accuracy, less compression)
2. **Selective quantization** (quantize only large layers)
3. **Use INT6** instead of INT8 (middle ground)

**Adjust in ChatterboxEngine.swift:**
```swift
// More conservative quantization
MLXNN.quantize(model: t3, groupSize: 128, bits: 8)  // Larger groups
```

---

## Technical Notes

### 1. MLX Quantization Support

MLX Swift provides comprehensive quantization through:
- **`MLX.quantized()`** - Low-level weight quantization
- **`MLX.quantizedMatmul()`** - Optimized matrix multiplication
- **`MLXNN.quantize(model:)`** - High-level model quantization
- **`QuantizedLinear`** - Drop-in replacement for Linear layers

### 2. Why INT8 vs INT4?

| Precision | Speedup | Quality Loss | Memory |
|-----------|---------|--------------|--------|
| **FP16** | 1.0x | 0% | 100% |
| **INT8** | 1.5-2x | <1% | 50% |
| **INT4** | 2-3x | ~3% | 25% |

**Chose INT8 because:**
- ✅ <1% quality loss (acceptable for TTS)
- ✅ 1.5-2x speedup (significant)
- ✅ 50% memory reduction
- ✅ Stable and well-tested

### 3. Quantization at Load Time

Quantization happens **once during model loading**, not at runtime:
- Weights are quantized after loading FP16 weights
- Quantized weights stored in GPU memory
- Forward passes use quantizedMatmul() automatically
- Zero runtime overhead (except faster execution)

### 4. Hardware Acceleration

MLX uses optimized Metal kernels for quantized operations:
- Matrix multiplication: ~2x faster on M1/M2/M3/M4/M5
- Memory bandwidth: 50% reduction improves cache efficiency
- Neural Accelerators: Automatic use on M5 chips

---

## Next Steps (If Quantization Succeeds)

**If performance is good and quality is maintained:**

1. **Make quantization default** for TTSServer
2. **Add quantization flag** to command-line args
3. **Update documentation** with performance numbers
4. **Move to next optimization:**
   - Speculative Decoding (2-3x additional speedup)
   - Streaming (instant perceived response)
   - KV Cache Compression (1.2-1.5x speedup)

**Combined Target:**
- INT8: 4.55s → 2.3s (2x faster)
- Speculative Decoding: 2.3s → 0.8s (3x faster)
- **Final: <1.0s** ✅

---

## Troubleshooting

### Build Errors

**Error: `MLXNN.quantize` not found**
- Solution: Update mlx-swift to 0.21.0+
- Check: `swift package show-dependencies | grep mlx`

**Error: Module not found**
- Solution: `swift package clean && swift build`

### Runtime Errors

**Error: Quantization fails**
- Check: Model loaded successfully before quantization
- Check: Weights are FP16/FP32 (not already quantized)

**Error: Audio quality is poor**
- Try: Increase groupSize to 128
- Try: Use bits: 6 instead of bits: 8
- Try: Selective quantization (skip embeddings)

### Performance Issues

**No speedup observed:**
- Check: Using release build (`-c release`)
- Check: GPU acceleration enabled
- Check: Sufficient GPU memory

**Slower than expected:**
- Possible: Quantization overhead on first run
- Try: Run multiple times to warm up caches
- Check: Profiling for bottlenecks

---

## References

- [MLX Quantization Guide](https://ml-explore.github.io/mlx/build/html/usage/quantization.html)
- [MLX Swift Documentation](https://swiftpackageindex.com/ml-explore/mlx-swift)
- [WWDC 2025: Explore LLM on Apple Silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) - Full optimization plan

---

*INT8 quantization implemented: December 23, 2025*
*Target: 1.5-2x speedup with <1% quality loss*
*Ready for user testing and validation*
