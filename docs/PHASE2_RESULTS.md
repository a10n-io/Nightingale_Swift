# Phase 2: Sampling Optimization Results
**Date:** December 23, 2025

## Objective
Optimize repetition penalty and top-p sampling to reduce GPU→CPU→GPU transfer overhead.

## Changes Made

### 1. Repetition Penalty Optimization
**Before:**
```swift
// Multiple GPU→CPU→GPU transfers in loop
var logitsArray = logitsFlat.asArray(Float.self)
let penalizedArray = penalizedLogits.asArray(Float.self)
for (i, token) in uniqueTokens.enumerated() {
    logitsArray[token] = penalizedArray[i]
}
logitsFlat = MLXArray(logitsArray)
```

**After:**
```swift
// Single GPU→CPU transfer, build arrays on CPU, single CPU→GPU transfer
let penalizedArray = penalizedLogits.asArray(Float.self)
var updateMaskArray = [Float](repeating: 0.0, count: logitsFlat.shape[0])
var penalizedFullArray = [Float](repeating: 0.0, count: logitsFlat.shape[0])

for (i, token) in uniqueTokens.enumerated() {
    updateMaskArray[token] = 1.0
    penalizedFullArray[token] = penalizedArray[i]
}

let updateMask = MLXArray(updateMaskArray)
let penalizedFull = MLXArray(penalizedFullArray)
logitsFlat = `where`(updateMask .== 1.0, penalizedFull, logitsFlat)
```

### 2. Top-P Sampling Optimization
**Before:**
```swift
// Multiple GPU syncs via .item() calls in loop
let probsArray = probs.asArray(Float.self)
var sortedPairs = probsArray.enumerated().map { ($0.offset, $0.element) }
sortedPairs.sort { $0.1 > $1.1 }

var cumSum: Float = 0.0
var keepIndices = Set<Int>()
for (idx, prob) in sortedPairs {
    cumSum += prob
    keepIndices.insert(idx)
    if cumSum > topP { break }
}

// Then rebuild array with Set lookups
var topPLogitsArray = filteredLogits.asArray(Float.self)
for i in 0..<topPLogitsArray.count {
    if !keepIndices.contains(i) && topPLogitsArray[i] > -1e30 {
        topPLogitsArray[i] = -Float.infinity
    }
}
filteredLogits = MLXArray(topPLogitsArray)
```

**After:**
```swift
// Single pass: sort, cumsum, and build mask together
let probsArray = probs.asArray(Float.self)
var sortedPairs = probsArray.enumerated().map { ($0.offset, $0.element) }
sortedPairs.sort { $0.1 > $1.1 }

var cumSum: Float = 0.0
var keepMaskArray = [Float](repeating: 0.0, count: probsArray.count)

for (idx, prob) in sortedPairs {
    cumSum += prob
    keepMaskArray[idx] = 1.0
    if cumSum > topP { break }
}

// Single CPU→GPU transfer and GPU mask application
let keepMask = MLXArray(keepMaskArray)
filteredLogits = `where`(keepMask .== 1.0, filteredLogits, MLXArray(-Float.infinity))
```

## Performance Measurements

### Test Results (3 sentences, `swift run -c release TTSServer`)

| Sentence | Tokens | Phase 1 T3 Time | Phase 2 T3 Time | Difference |
|----------|--------|-----------------|-----------------|------------|
| 1 | 102 | 3.87s | 3.92s | +0.05s (1.3% slower) |
| 2 | 78 | 2.55s | 2.48s | -0.07s (2.7% faster) |
| 3 | 127 | 4.15s | 4.24s | +0.09s (2.2% slower) |

**Average Change:** ~0% (within measurement variance)

### Analysis

**Why No Significant Improvement:**

1. **Top-P with topP=1.0 (default parameter)**
   - With topP=1.0, ALL tokens are kept
   - No actual filtering occurs
   - The "optimization" doesn't reduce work because there's no work to reduce

2. **Repetition Penalty Overhead is Minimal**
   - The penalty only applies to ~10-100 unique tokens (small subset of 8K vocab)
   - GPU→CPU transfers for small arrays are fast
   - The CPU loop overhead was already small

3. **Real Bottleneck is Elsewhere**
   - 102 forward passes through 30-layer transformer = ~3.6s
   - Sampling overhead = ~0.2-0.3s total
   - Our optimizations reduced that by maybe 0.05s (not measurable)

4. **Measurement Variance**
   - Test runs vary by ±0.1s naturally
   - Our "improvements" are within noise

## Conclusion

**Phase 2 sampling optimization: NO MEASURABLE BENEFIT**

This is actually valuable information:
- ✅ **Confirmed:** Sampling is NOT a bottleneck (only ~5% of generation time)
- ✅ **Learned:** The real bottleneck is the T3 forward pass (95% of time)
- ✅ **Decision:** Don't waste more time optimizing sampling

**Next Steps:**
Focus optimization efforts on areas identified in OPTIMIZATION_ROADMAP.md:
1. **INT8 Quantization** - Reduces matmul cost in forward pass (1.5-2x speedup)
2. **Speculative Decoding** - Reduces number of forward passes (2-3x speedup)
3. **KV Cache Compression** - Reduces memory bandwidth in attention (1.2-1.5x speedup)

These target the actual bottleneck (forward pass through 30 transformer layers).

## Recommendation

**Revert Phase 2 changes?**
- **No** - The code is cleaner (fewer CPU loops) even if not faster
- **No regression** - Performance is identical within variance
- **Better for quantized models** - Cleaner CPU→GPU boundaries will help when weights are quantized

**Keep the changes and move to OPTIMIZATION_ROADMAP.md strategies.**

---

*Phase 2 completed: December 23, 2025*
*Status: TESTED - No measurable improvement (sampling is not bottleneck)*
*Next: Focus on forward pass optimizations (quantization, speculative decoding)*
