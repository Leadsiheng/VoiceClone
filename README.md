# VoiceClone

Zero-shot voice cloning + conversational AI.
Upload 10 seconds of someone's voice, then chat with them via text or speech.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM (RTX 2060+) | RTX 3060+ |
| RAM | 16GB | 32GB |
| Storage | 10GB free | 20GB free |
| OS | Windows 10/11, Linux, macOS | Windows 11 (for RTX 5060) |

## Quick Start (Y9000P with RTX 5060)

### 1. Prerequisites

Install **Python 3.10** and **CUDA Toolkit 12.4**:

```powershell
# Check if Python is installed
python --version

# Install CUDA Toolkit (if not installed)
# Download from: https://developer.nvidia.com/cuda-12-4-0-download-archive
```

### 2. Clone & Setup

```powershell
# Clone this project
git clone https://github.com/YOUR_USERNAME/VoiceClone.git
cd VoiceClone

# Clone CosyVoice (required for voice cloning)
git clone https://github.com/FunAudioLLM/CosyVoice.git ../CosyVoice
cd ../CosyVoice

# Install CosyVoice dependencies
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Return to VoiceClone and install our dependencies
cd ../VoiceClone
pip install -r requirements.txt
```

### 3. Configure API Key

Edit `config.yaml` and set your DeepSeek API key:

```yaml
llm:
  backend: deepseek
  deepseek:
    api_key: "sk-your-api-key-here"
```

Get a key from: https://platform.deepseek.com

Or use local Gemma4 (Ollama):

```yaml
llm:
  backend: ollama
```

### 4. Add Voice Samples

Place reference audio files in `voices/`:

```
voices/
└── alice/
    ├── reference.wav        # 3-15 sec clear speech
    └── speaker.yaml         # Name, reference text, system prompt
```

Or use the Web UI's "Voice Management" tab to upload audio.

Example `speaker.yaml`:

```yaml
name: "Alice"
reference_text: "今天天气真不错，我们去公园散步吧"
system_prompt: "你是一个温柔友善的女生，回答简洁自然，每句话不超过30个字。"
```

### 5. Launch

```powershell
python app.py
```

Open in browser: `http://localhost:7860`
Friends access: use the Gradio public URL printed in the terminal

## Architecture

```
Voice input → [FunASR] → text → [DeepSeek/Ollama] → reply text → [CosyVoice] → cloned voice output
                                  ↑
                          Text input (skip ASR)
```

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `llm.backend` | `deepseek` or `ollama` | `deepseek` |
| `llm.deepseek.model` | DeepSeek model | `deepseek-chat` |
| `llm.ollama.model` | Ollama model name | `gemma4:latest` |
| `tts.model` | CosyVoice variant | `CosyVoice-300M-SFT` |
| `asr.model` | FunASR model | Paraformer Chinese |
| `server.share` | Gradio public link | `true` |

## Tips for Best Voice Cloning

1. **Audio quality matters most** — quiet room, no echo, close to mic
2. **10-15 seconds** is the sweet spot
3. **Natural speaking** — don't read like a robot
4. **Varied content** — cover different sounds in 10 seconds
5. **Provide reference text** — tells the model what the audio says
