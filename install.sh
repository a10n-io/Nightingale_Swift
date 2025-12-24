#!/bin/bash
#
# Nightingale TTS - One Command Installer & Launcher
# ===================================================
# Run: curl -sL https://YOUR_URL/install.sh | bash
#
set -e

# ============================================
# CONFIGURATION (downloaded from Google Drive)
# ============================================
TOKEN_GDRIVE_ID="1_xVvnHx0oxws0s3wt60fCwhmktQ-JQBY"
ASSETS_GDRIVE_ID="1AMa2D23SK4F8LcPUmrWpE84mAKaGpSOI"

# ============================================
# Setup
# ============================================
INSTALL_DIR="$HOME/Nightingale_Swift"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              🎤 Nightingale TTS                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================
# If already installed, just launch
# ============================================
if [ -f "$INSTALL_DIR/.build/release/TTSWebUI" ]; then
    echo -e "${GREEN}✓${NC} Nightingale already installed"
    echo -e "${BLUE}==>${NC} Launching TTS Web Server..."
    echo ""
    cd "$INSTALL_DIR"
    exec .build/release/TTSWebUI
fi

# ============================================
# Fresh install
# ============================================
echo -e "${BLUE}==>${NC} Installing Nightingale TTS..."

# Check requirements
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}✗${NC} This only works on macOS"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${RED}✗${NC} Requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

# Install Xcode CLI tools if needed
if ! xcode-select -p &>/dev/null; then
    echo -e "${BLUE}==>${NC} Installing Xcode Command Line Tools..."
    xcode-select --install
    echo ""
    echo "Please complete the Xcode tools installation, then run this script again."
    exit 0
fi

echo -e "${GREEN}✓${NC} Requirements met"

# Download token from Google Drive
echo -e "${BLUE}==>${NC} Fetching access credentials..."
GITHUB_TOKEN=$(curl -sL "https://drive.google.com/uc?export=download&id=${TOKEN_GDRIVE_ID}")

if [ -z "$GITHUB_TOKEN" ] || [[ "$GITHUB_TOKEN" == *"html"* ]]; then
    echo -e "${RED}✗${NC} Failed to fetch credentials. Check Google Drive access."
    exit 1
fi

# Download repository
echo -e "${BLUE}==>${NC} Downloading Nightingale..."

if command -v git &>/dev/null; then
    git clone --depth 1 "https://${GITHUB_TOKEN}@github.com/a10n-io/Nightingale_Swift.git" "$INSTALL_DIR" 2>/dev/null
else
    curl -sL -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/a10n-io/Nightingale_Swift/zipball/main" \
        -o /tmp/nightingale.zip
    unzip -q /tmp/nightingale.zip -d /tmp
    mv /tmp/a10n-io-Nightingale_Swift-* "$INSTALL_DIR"
    rm /tmp/nightingale.zip
fi

echo -e "${GREEN}✓${NC} Code downloaded"

cd "$INSTALL_DIR"

# Function to download large files from Google Drive
download_gdrive() {
    local file_id=$1
    local output=$2

    # First try with gdown (handles large files best)
    if command -v gdown &>/dev/null; then
        gdown "https://drive.google.com/uc?id=${file_id}" -O "$output" --fuzzy
        return $?
    fi

    # Install gdown if pip3 available
    if command -v pip3 &>/dev/null; then
        echo "   Installing download helper..."
        pip3 install --user --break-system-packages gdown 2>/dev/null || pip3 install --user gdown 2>/dev/null
        # Add common Python bin paths
        export PATH="$HOME/Library/Python/3.9/bin:$HOME/Library/Python/3.10/bin:$HOME/Library/Python/3.11/bin:$HOME/Library/Python/3.12/bin:$HOME/.local/bin:$PATH"
        if command -v gdown &>/dev/null; then
            gdown "https://drive.google.com/uc?id=${file_id}" -O "$output" --fuzzy
            return $?
        fi
    fi

    # Fallback: direct curl with new Google Drive URL format
    echo "   Downloading via curl..."
    curl -L -o "$output" \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t" \
        --progress-bar
}

# Check if models already exist
if [ -f "models/chatterbox/s3gen.safetensors" ]; then
    echo -e "${GREEN}✓${NC} Models already installed, skipping download"
else
    # Download models and voices from Google Drive
    echo -e "${BLUE}==>${NC} Downloading AI models & voices (~2.5GB, this may take a few minutes)..."

    ASSETS_ZIP="/tmp/nightingale_assets.zip"
    download_gdrive "$ASSETS_GDRIVE_ID" "$ASSETS_ZIP"

    # Extract the zip
    if [ -f "$ASSETS_ZIP" ]; then
        echo -e "${BLUE}==>${NC} Extracting assets..."
        unzip -q "$ASSETS_ZIP" -d /tmp/nightingale_extract

        # Move contents from any subfolder (handles "Model + Voices" folder in zip)
        if [ -d /tmp/nightingale_extract ]; then
            find /tmp/nightingale_extract -type d -name "models" -exec cp -r {} "$INSTALL_DIR/" \;
            find /tmp/nightingale_extract -type d -name "baked_voices" -exec cp -r {} "$INSTALL_DIR/" \;
            rm -rf /tmp/nightingale_extract
        fi

        rm -f "$ASSETS_ZIP"
        echo -e "${GREEN}✓${NC} Models and voices installed"
    else
        echo -e "${RED}✗${NC} Download failed. Please check your internet connection."
        exit 1
    fi

    # Verify files exist after download
    if [ ! -f "models/chatterbox/s3gen.safetensors" ]; then
        echo -e "${RED}✗${NC} Model files not found after extraction."
        echo "   Expected: models/chatterbox/s3gen.safetensors"
        ls -la models/ 2>/dev/null || echo "   (models/ directory not found)"
        exit 1
    fi
fi

# Build
echo -e "${BLUE}==>${NC} Building Nightingale (this takes 2-3 minutes)..."
swift build -c release --product TTSWebUI 2>&1 | grep -E "(Compiling|Linking|Build complete)" || true

if [ ! -f ".build/release/TTSWebUI" ]; then
    echo -e "${RED}✗${NC} Build failed. Please check Swift/Xcode installation."
    exit 1
fi

echo -e "${GREEN}✓${NC} Build complete"

# Build Metal library
if [ -f "build_metallib.sh" ]; then
    echo -e "${BLUE}==>${NC} Building Metal shaders..."
    chmod +x build_metallib.sh
    ./build_metallib.sh 2>/dev/null || true
fi

# Done - launch!
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Installation Complete!                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${BLUE}==>${NC} Launching TTS Web Server..."
echo ""

exec .build/release/TTSWebUI
