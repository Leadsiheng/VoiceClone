---
name: voiceclone
description: Use when working with the VoiceClone project. Covers starting IndexTTS2 native web UI, starting the VoiceClone app, and project conventions. Use ONLY when the user mentions VoiceClone, IndexTTS2, voice cloning, or related topics.
---

# VoiceClone

Zero-shot voice cloning + conversational AI powered by IndexTTS2 + DeepSeek.

## Project Structure

```
VoiceClone/
├── app.py              # Production Gradio app (full pipeline)
├── preview_app.py      # Mock preview (no model loading)
├── engines/            # TTSEngine, LLMEngine, ASREngine
├── prompts.py          # 7 voice styles
├── config.yaml         # DeepSeek API key, paths
├── scripts/
│   ├── start_indextts.sh    # Start IndexTTS2 native Web UI
│   └── start_voiceclone.sh  # Start VoiceClone app
└── voices/             # Speaker reference audio
    ├── biao/
    ├── zhe/
    └── xin/
```

External dependencies:
```
../index-tts/           # Cloned from https://github.com/index-tts/index-tts.git
                        # Contains model checkpoints and indextts/ package
```

## Quick Commands

### Start IndexTTS2 Native Web UI

For testing voice cloning quality directly:

```bash
bash scripts/start_indextts.sh
```

Opens at `http://localhost:7860`. Upload reference WAV → type text → hear cloned voice.

### Start VoiceClone App (Full Pipeline)

TTS + LLM + ASR + custom UI:

```bash
bash scripts/start_voiceclone.sh
```

Opens at `http://localhost:7860`. Desktop UI with waveform, mode tabs, voice/style selection.

## Architecture

```
VoiceClone/app.py  ──(stdin/stdout JSON)──>  indextts_bridge.py  ──>  IndexTTS2
                                               (runs inside uv venv)
```

Bridge protocol:
```
→ {"cmd":"init","cfg_path":"checkpoints/config.yaml","model_dir":"checkpoints",...}
← {"status":"ok"}

→ {"cmd":"infer","spk_audio_prompt":"...wav","text":"...","output_path":"...wav"}
← {"status":"ok","output_path":"...wav"}

→ {"cmd":"quit"}
← {"status":"ok"}
```

## Model Files

All downloaded locally under `../index-tts/checkpoints/` (~15GB total):
- gpt.pth (3.2GB) - main GPT model
- s2mel.pth (1.1GB) - spectrogram-to-mel
- qwen0.6bemo4-merge/ (1.1GB) - emotion model
- bigvgan/ - BigVGAN vocoder
- maskgct/ - MaskGCT semantic codec
- campplus_cn_common.bin - speaker embedding
- w2v-bert-2.0/ - Wav2Vec2Bert

## Key Files Modified for M5 Mac

In `../index-tts/indextts/infer_v2.py`:
- Line 115-117: w2v-bert path patched to `./checkpoints/w2v-bert-2.0`
- Line 126: MaskGCT path patched to local
- Line 155: CAM++ path patched to local
- Line 164-165: BigVGAN path patched to local

In `../index-tts/indextts/utils/maskgct_utils.py`:
- Line 88: w2v-bert path patched

## Voice Styles

Defined in `prompts.py`: 清纯男高, 贴心男友, 幽默, 温柔, 知性, 性感, 色情🔞
