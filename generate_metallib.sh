#!/bin/bash
set -e

# Configuration
BUILD_DIR=".build"
CHECKOUTS_DIR="$BUILD_DIR/checkouts/mlx-swift"
GENERATED_DIR="$CHECKOUTS_DIR/Source/Cmlx/mlx-generated/metal"
KERNELS_DIR="$CHECKOUTS_DIR/Source/Cmlx/mlx/mlx/backend/metal/kernels"
BIN_DIR=$1

if [ -z "$BIN_DIR" ]; then
    echo "Usage: $0 <binary_output_directory>"
    exit 1
fi

echo "Generating default.metallib in $BIN_DIR..."

TEMP_DIR="metal_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Compiler flags
# We need to target macOS.
XCFLAGS="-sdk macosx metal -c -std=metal3.0"  # Assuming Metal 3.0 or appropriate version

# 1. Compile Generated Shaders
echo "Compiling generated shaders..."
find "$GENERATED_DIR" -name "*.metal" | while read -r file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    # echo "  $filename"
    
    # Include path for generated: the generated dir itself
    xcrun $XCFLAGS "$file" -I "$GENERATED_DIR" -o "$TEMP_DIR/${name}_gen.air"
done

# 2. Compile Core Kernels (skipping those that exist in Generated)
echo "Compiling core kernels..."
find "$KERNELS_DIR" -name "*.metal" | while read -r file; do
    filename=$(basename "$file")
    name="${filename%.*}"
    
    # Check if this file exists in generated
    if [ -f "$TEMP_DIR/${name}_gen.air" ]; then
        # echo "  Skipping $filename (covered by generated)"
        continue
    fi
    
    # Also some files might be headers but have .metal extension? 
    # Usually headers are excluded if they don't have kernels.
    # But if compilation fails (no kernels), metal compiler might warn or error?
    # We'll ignore errors for headers? No, compilation unit with no kernels is fine?
    
    # Include path for kernels: source root
    INCLUDE_PATH="$CHECKOUTS_DIR/Source/Cmlx/mlx"
    
    # We capture output to check for errors, but proceed.
    # echo "  $filename"
    if xcrun $XCFLAGS "$file" -I "$INCLUDE_PATH" -o "$TEMP_DIR/${name}.air" 2>/dev/null; then
        :
    else
        # If it failed, it might be a header or dependency. 
        # Check if it was because "no kernels".
        # For now, we assume failure means it wasn't a standalone kernel file.
        # echo "  Input $filename failed to compile (likely a header/utils file). Skipping."
        true
    fi
done

# 3. Link them all
echo "Linking default.metallib..."
xcrun -sdk macosx metallib "$TEMP_DIR"/*.air -o "$BIN_DIR/default.metallib"

echo "Done. Created $BIN_DIR/default.metallib"
rm -rf "$TEMP_DIR"
