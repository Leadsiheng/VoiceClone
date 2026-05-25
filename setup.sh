#!/usr/bin/env bash
# VoiceClone Mac Setup (M5 / Apple Silicon)
# Run from the VoiceClone project directory:
#   bash setup.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  VoiceClone Setup - macOS (Apple Silicon)${NC}"
echo -e "${CYAN}========================================${NC}"

# ── 0. Check Python ──
echo -e "\n${YELLOW}[0/6] Checking Python...${NC}"
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo -e "${GREEN}[OK] Found: $cmd $ver${NC}"
            break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}[ERROR] Python 3.10+ required. Install from https://python.org${NC}"
    exit 1
fi

# ── 1. Create venv for VoiceClone ──
echo -e "\n${YELLOW}[1/6] Creating VoiceClone virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}[OK] venv ready.${NC}"

# ── 2. Install VoiceClone dependencies ──
echo -e "\n${YELLOW}[2/6] Installing VoiceClone dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}[OK] Dependencies installed.${NC}"

# ── 3. Install uv ──
echo -e "\n${YELLOW}[3/6] Installing uv package manager...${NC}"
if ! command -v uv &>/dev/null; then
    pip install uv
    echo -e "${GREEN}[OK] uv installed.${NC}"
else
    echo -e "${GREEN}[OK] uv already installed.${NC}"
fi

# ── 4. Clone index-tts ──
INDEX_TTS_DIR="../index-tts"
if [ ! -d "$INDEX_TTS_DIR" ]; then
    echo -e "\n${YELLOW}[4/6] Cloning index-tts (IndexTTS2)...${NC}"
    git clone https://github.com/index-tts/index-tts.git "$INDEX_TTS_DIR"
    echo -e "${GREEN}[OK] index-tts cloned.${NC}"
else
    echo -e "\n${YELLOW}[4/6] index-tts already exists, skipping clone.${NC}"
fi

# ── 5. Install index-tts dependencies ──
echo -e "\n${YELLOW}[5/6] Installing IndexTTS2 dependencies (uv sync)...${NC}"
cd "$INDEX_TTS_DIR"
uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" || \
uv sync --all-extras
echo -e "${GREEN}[OK] IndexTTS2 dependencies installed.${NC}"

# ── 5b. Download models ──
echo -e "\n${YELLOW}[5b/6] Downloading IndexTTS2 model weights...${NC}"
if [ ! -f "checkpoints/gpt.pth" ]; then
    echo "Downloading via ModelScope..."
    uv run python -c "
from modelscope import snapshot_download
snapshot_download('IndexTeam/IndexTTS-2', cache_dir='checkpoints')
print('Models downloaded.')
" 2>/dev/null || {
        echo "ModelScope failed, trying huggingface-cli..."
        uv tool install "huggingface-hub[cli,hf_xet]" 2>/dev/null || true
        export HF_ENDPOINT="https://hf-mirror.com"
        hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints 2>/dev/null || \
        hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    }
    echo -e "${GREEN}[OK] Models downloaded.${NC}"
else
    echo -e "${GREEN}[OK] Models already exist.${NC}"
fi

# ── 5c. Copy bridge script ──
echo -e "\n${YELLOW}[5c/6] Installing TTS bridge script...${NC}"
cp ../VoiceClone/indextts_bridge.py indextts_bridge.py 2>/dev/null || \
cp ../../VoiceClone/indextts_bridge.py indextts_bridge.py 2>/dev/null || \
echo -e "${YELLOW}[WARN] Could not auto-copy indextts_bridge.py. Please copy it manually.${NC}"
[ -f indextts_bridge.py ] && echo -e "${GREEN}[OK] Bridge script installed.${NC}"

cd ../VoiceClone 2>/dev/null || cd ../../VoiceClone 2>/dev/null || cd "$(dirname "$0")"

# ── 6. Verify ──
echo -e "\n${YELLOW}[6/6] Verifying...${NC}"
echo -e "  Python: $($PYTHON_CMD --version)"
echo -e "  uv:     $(uv --version)"
echo -e "  venv:   $(which python)"

echo -e "\n${CYAN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit config.yaml and set your DeepSeek API key"
echo "  2. Add real voice samples to voices/biao/, voices/zhe/, voices/xin/"
echo "  3. Edit each voices/*/speaker.yaml with reference_text"
echo "  4. Run: python app.py"
echo ""
echo "Activate venv later with: source venv/bin/activate"
