# MLX-Swift Linear Layer Bug Workaround

## Issue

A bug was discovered in MLX-Swift's `Linear` layer where calling the layer with different input tensors (even when they have identical shapes and dtypes) causes a broadcast shape mismatch error.

### Error Message
```
Fatal error: [broadcast_shapes] Shapes (1,80,64) and (1,564,80) cannot be broadcast.
```

### Reproduction Pattern
```swift
let linear = Linear(512, 80)

// First call works
let output1 = linear(input1)  // ✅ Works

// Second call with SAME input works
let output2 = linear(input1)  // ✅ Works

// Third call with DIFFERENT input fails
let output3 = linear(input2)  // ❌ Crashes (even if input2 has same shape as input1)
```

## Root Cause

The MLX-Swift `Linear` layer appears to have internal caching or shape inference logic that becomes corrupted when the layer is called with a different input tensor after being used with an initial input.

## Workaround

Instead of using the `Linear` layer's `callAsFunction` method, perform the matrix multiplication manually:

```swift
// Instead of:
let output = linearLayer(input)

// Use:
let weight = linearLayer.weight  // [out_features, in_features]
let weightT = weight.transposed()  // [in_features, out_features]
var output = matmul(input, weightT)  // [B, T, out_features]
if let bias = linearLayer.bias {
    output = output + bias  // broadcast bias
}
```

## Implementation

In `S3Gen.swift`, a helper function `applyEncoderProj` was added:

```swift
// WORKAROUND: Manual matmul for encoderProj to bypass MLX Linear layer caching bug
public func applyEncoderProj(_ input: MLXArray) -> MLXArray {
    let weight = encoderProj.weight  // [80, 512]
    let weightT = weight.transposed()  // [512, 80]
    var output = matmul(input, weightT)  // [B, T, 80]
    if let bias = encoderProj.bias {
        output = output + bias  // broadcast [80] to [B, T, 80]
    }
    return output
}
```

All calls to `encoderProj(h)` were replaced with `applyEncoderProj(h)`.

## Affected Locations

- `S3Gen.generate()` - main generation function
- `S3Gen.getEncoderOutput()` - encoder output extraction
- `S3Gen.generateWithTracing()` - traced generation
- `S3Gen.generateWithMel()` - generation with mel output

## Status

- **Workaround implemented**: Yes
- **Bug reported to MLX team**: Pending
- **MLX-Swift version affected**: 0.21.0+

## Future Work

1. Report this bug to the MLX-Swift team at https://github.com/ml-explore/mlx-swift
2. Once fixed upstream, remove the workaround and revert to using `Linear` directly
3. Consider applying similar workarounds to other `Linear` layers if needed (e.g., in FlowMatchingDecoder)
