#!/usr/bin/env bash
# Start VoiceClone App (full pipeline)
# Usage: bash scripts/start_voiceclone.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
python3 -u app.py
