# TTFA Optimization Plan
**Date:** December 24, 2025
**Current TTFA:** 2135ms
**Target TTFA:** 100ms (aspirational), 500-600ms (realistic)

---

## Requirements

| Requirement | Priority | Status |
|-------------|----------|--------|
| Perfect audio quality | Must have | ✅ Maintained |
| Multilingual support (2454 vocab) | Must have | ✅ Maintained |
| As fast as possible (target 100ms TTFA) | Must have | 🔄 In progress |
| Runs on iPhone (future) | Must have | ✅ Compatible path |

---

## Current Performance Breakdown

| Component | Time | Notes |
|-----------|------|-------|
| T3 first 20 tokens | ~310ms | 64 tok/s, 2.5x faster than real-time |
| S3Gen first chunk | ~1823ms | **Bottleneck** - 2.3x RTF |
| **Total TTFA** | **~2135ms** | |

### Streaming Viability
- Audio per token: 40ms
- T3 time per token: 15.6ms
- **Margin: +24.4ms/token** ✅ T3 is ahead of real-time

---

## 100ms TTFA Reality Check

| Component | Current | Best Realistic | Gap to 100ms |
|-----------|---------|----------------|--------------|
| T3 (10 tokens) | 155ms | ~80ms | Still over budget |
| S3Gen (10 tokens) | ~900ms | ~400ms (INT8+MLX) | 4x too slow |
| **Total** | 2135ms | **~500-600ms** | 5-6x off target |

**Hard Truth:** 100ms TTFA requires architectural changes:
- Pre-computed prosody/first syllables
- Different vocoder architecture
- Streaming speculative decoding

**Realistic Target:** 500-600ms TTFA with current architecture

---

## Optimization Priorities

### Priority 1: S3Gen INT8 Quantization
**Estimated savings:** 600-900ms
**Effort:** 1-2 days
**Risk:** Low

| Metric | Before | After |
|--------|--------|-------|
| S3Gen first chunk | 1823ms | 912-1215ms |
| S3Gen RTF | 2.3x | 1.1-1.5x |

- Biggest single optimization
- iPhone Neural Engine compatible
- No quality degradation with INT8
- Proven approach (already done for T3)

### Priority 2: Smaller First Chunk (10 tokens)
**Estimated savings:** 150ms
**Effort:** 30 minutes
**Risk:** None

- Reduces T3 time for first chunk by 50%
- Trade-off: less initial audio buffer (400ms vs 800ms)
- Still viable for streaming playback

### Priority 3: Phase 2 Sampling Optimizations
**Estimated savings:** 200-300ms
**Effort:** 3-4 hours
**Risk:** Low

| Optimization | Savings | Notes |
|--------------|---------|-------|
| Repetition penalty (pure MLX) | 0.1-0.2s | GPU instead of CPU |
| Top-p sampling (GPU ops) | 0.2-0.4s | Avoid CPU sort |

- From CURRENT_AUDIT_FINDINGS.md Phase 2
- Helps T3 which is already fast
- Quality preserved

### Priority 4: MLX Fusion Optimizations
**Estimated savings:** 100-200ms
**Effort:** 1 day
**Risk:** Low

- Operation fusion (matmul + activation)
- Metal kernel optimizations
- Unified memory optimizations

---

## Projected Results

| Scenario | TTFA | Improvement |
|----------|------|-------------|
| Current | 2135ms | baseline |
| + S3Gen INT8 | ~1300ms | 39% faster |
| + Smaller chunk | ~1150ms | 46% faster |
| + Phase 2 sampling | ~900ms | 58% faster |
| + MLX fusion | **~700ms** | **67% faster** |

---

## iPhone Compatibility

All recommended optimizations are iPhone-compatible:

| Optimization | iPhone Support | Notes |
|--------------|----------------|-------|
| INT8 Quantization | ✅ Neural Engine | Optimized for Apple Silicon |
| Smaller chunks | ✅ Yes | No hardware dependency |
| Phase 2 sampling | ✅ Yes | Pure MLX operations |
| MLX fusion | ✅ Yes | Native Metal support |

### iPhone Considerations
- Memory constraint: INT8 reduces model size by 50%
- Neural Engine: 4x speedup for quantized ops
- Thermal: Streaming helps avoid sustained load
- Battery: Faster = less total compute

---

## Implementation Order

```
Week 1:
├── Day 1-2: S3Gen INT8 Quantization
│   ├── Quantize decoder weights
│   ├── Quantize vocoder weights
│   ├── Validate quality (A/B test)
│   └── Benchmark speedup
├── Day 3: Smaller first chunk + Phase 2 sampling
│   ├── Reduce initial chunk to 10 tokens
│   ├── GPU repetition penalty
│   └── GPU top-p sampling
└── Day 4-5: MLX optimizations + testing
    ├── Operation fusion
    ├── Full regression testing
    └── iPhone simulator testing
```

---

## Quality Validation Checklist

After each optimization:
- [ ] Audio sounds identical (blind A/B test)
- [ ] Deterministic output (temp=0.0001 → same hash)
- [ ] Multilingual works ([en], [fr], [es], etc.)
- [ ] Token count stable (±2 tokens)
- [ ] No clicking/artifacts at chunk boundaries
- [ ] iPhone builds and runs

---

## Beyond 600ms: Future Options

To reach 100ms would require:

| Approach | TTFA | Quality | Effort |
|----------|------|---------|--------|
| Pre-baked first syllables | ~100ms | ✅ Perfect | 1 week |
| Speculative T3 decoding | ~300ms | ✅ Perfect | 2 weeks |
| Distilled S3Gen (4 layers) | ~200ms | ⚠️ ~95% | 1 month |
| Alternative TTS (VITS/StyleTTS2) | ~100ms | ⚠️ Different | 2 months |

---

## Files Modified

### Already Optimized
- `Sources/Nightingale/T3Model.swift` - INT8, debug cleanup
- `Sources/Nightingale/S3GenStreaming.swift` - Streaming wrapper
- `test_scripts/StreamingTest/main.swift` - 4-token overlap

### To Be Modified
- `Sources/Nightingale/S3Gen.swift` - INT8 quantization
- `Sources/Nightingale/ChatterboxEngine.swift` - Smaller chunks
- `Sources/Nightingale/T3Model.swift` - GPU sampling

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| TTFA | 2135ms | 600ms | 400ms |
| S3Gen RTF | 2.3x | 1.0x | 0.7x |
| T3 tok/s | 64 | 80 | 100 |
| iPhone TTFA | N/A | 1000ms | 700ms |

---

*Last Updated: December 24, 2025*
*Status: Planning → Ready for Implementation*
