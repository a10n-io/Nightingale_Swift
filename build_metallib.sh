#!/bin/bash
set -e

BUILD_DIR=".build"
CHECKOUTS_DIR="$BUILD_DIR/checkouts/mlx-swift"
GENERATED_DIR="$CHECKOUTS_DIR/Source/Cmlx/mlx-generated/metal"
KERNELS_DIR="$CHECKOUTS_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"
BIN_DIR=".build/arm64-apple-macosx/debug"

echo "Generating default.metallib in $BIN_DIR..."

TEMP_DIR="metal_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

XCFLAGS="-sdk macosx metal -c -std=metal3.0"

echo "Compiling generated shaders..."
find "$GENERATED_DIR" -name "*.metal" 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    xcrun $XCFLAGS "$file" -I "$GENERATED_DIR" -o "$TEMP_DIR/${name}_gen.air" 2>/dev/null || true
done

echo "Compiling core kernels..."
find "$KERNELS_DIR" -name "*.metal" 2>/dev/null | while read -r file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    
    if [ -f "$TEMP_DIR/${name}_gen.air" ]; then
        continue
    fi
    
    INCLUDE_PATH="$CHECKOUTS_DIR/Source/Cmlx/mlx"
    xcrun $XCFLAGS "$file" -I "$INCLUDE_PATH" -o "$TEMP_DIR/${name}.air" 2>/dev/null || true
done

echo "Linking default.metallib..."
xcrun -sdk macosx metallib "$TEMP_DIR"/*.air -o "$BIN_DIR/default.metallib"

echo "Done. Created $BIN_DIR/default.metallib"
rm -rf "$TEMP_DIR"
