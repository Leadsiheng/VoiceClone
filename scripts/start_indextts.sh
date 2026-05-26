#!/usr/bin/env bash
# Start IndexTTS2 native Web UI for voice cloning testing
# Usage: bash scripts/start_indextts.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INDEX_DIR="$PROJECT_DIR/../index-tts"

if [ ! -d "$INDEX_DIR" ]; then
    echo "ERROR: index-tts not found at $INDEX_DIR"
    echo "Clone it first: git clone https://github.com/index-tts/index-tts.git $INDEX_DIR"
    exit 1
fi

echo "Starting IndexTTS2 Web UI..."
cd "$INDEX_DIR"
uv run webui.py
