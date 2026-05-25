# VoiceClone

Zero-shot voice cloning + conversational AI.
Powered by **IndexTTS2** (Bilibili) for voice synthesis, **DeepSeek** / **Ollama** for conversation, **FunASR** for speech recognition.

## Features

| Mode | Input | Pipeline |
|------|-------|----------|
| 语音聊天 | Microphone | ASR → LLM (styled) → IndexTTS2 (emotion) |
| 发送短信 | Text | LLM (styled) → IndexTTS2 (emotion) |
| 朗读内容 | Text | IndexTTS2 |

- 7 voice styles (清纯男高, 贴心男友, 幽默, 温柔, 知性, 性感, 色情🔞)
- Real-time waveform animation (Web Audio API driven)
- Multi-speaker support (彪 / 喆 / 鑫)
- Desktop-optimized UI

## Architecture

```
VoiceClone/ (this repo)
├── app.py                    # Gradio Web UI
├── engines/
│   ├── tts_engine.py         # IndexTTS2 bridge (subprocess)
│   ├── llm_engine.py         # DeepSeek / Ollama
│   └── asr_engine.py         # FunASR Paraformer
├── prompts.py                # 7 style system prompts
├── indextts_bridge.py        # Bridge script (copied to index-tts/)
├── config.yaml
└── voices/                   # Speaker reference audio

index-tts/ (cloned separately)
├── indextts/                 # IndexTTS2 Python package
├── indextts_bridge.py        # Copied from VoiceClone/
└── checkpoints/              # Model weights (downloaded)
```

**Communication**: VoiceClone → [stdin/stdout JSON] → indextts_bridge.py → IndexTTS2

## Quick Start (macOS / Apple Silicon)

```bash
git clone https://github.com/Leadsiheng/VoiceClone.git
cd VoiceClone
bash setup.sh
```

This automatically:
1. Creates Python venv
2. Installs VoiceClone dependencies
3. Clones index-tts repo
4. Installs IndexTTS2 (via uv)
5. Downloads model weights (~2GB)
6. Copies the bridge script

### Manual Setup

```bash
# 1. Clone both repos
git clone https://github.com/Leadsiheng/VoiceClone.git
git clone https://github.com/index-tts/index-tts.git ../index-tts

# 2. Set up VoiceClone
cd VoiceClone
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Set up IndexTTS2
cd ../index-tts
uv sync --all-extras
# Download models (choose one):
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
# or via ModelScope:
uv run python -c "from modelscope import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', cache_dir='checkpoints')"

# 4. Copy bridge script
cp ../VoiceClone/indextts_bridge.py .

# 5. Return and configure
cd ../VoiceClone
# Edit config.yaml: set deepseek api_key
```

## Windows (Y9000P RTX 5060)

```powershell
git clone https://github.com/Leadsiheng/VoiceClone.git
cd VoiceClone
powershell -ExecutionPolicy Bypass -File setup.ps1
```

## Voice Samples

Place reference audio (3-15 sec clear speech) in `voices/`:

```
voices/
├── biao/
│   ├── reference.wav
│   └── speaker.yaml
├── zhe/
│   ├── reference.wav
│   └── speaker.yaml
└── xin/
    ├── reference.wav
    └── speaker.yaml
```

`speaker.yaml`:
```yaml
name: "彪"
reference_text: "今天天气真不错，我们出去走走吧"
system_prompt: ""
```

## Launch

```bash
# In VoiceClone directory with venv activated:
python app.py
```

Open: `http://localhost:7860`
Share: Gradio public URL printed in terminal (when `server.share: true`)

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `llm.backend` | `deepseek` or `ollama` | `deepseek` |
| `llm.deepseek.api_key` | DeepSeek API key | (required) |
| `llm.ollama.model` | Ollama model name | `gemma4:latest` |
| `tts.index_tts_dir` | Path to index-tts repo | `../index-tts` |
| `tts.use_fp16` | Half-precision inference | `false` |
| `asr.model` | FunASR model | Paraformer Chinese |
| `server.share` | Gradio public link | `true` |

## Tips for Best Cloning

1. 10-15 sec audio in quiet room
2. Natural speaking style (not reading)
3. Varied phonetic content
4. Provide reference_text in speaker.yaml

## Future: LoRA Fine-tuning

IndexTTS2 supports LoRA fine-tuning for higher-quality voice clones.
Gather 3-5 minutes of clean speech per speaker, then fine-tune a LoRA adapter.
