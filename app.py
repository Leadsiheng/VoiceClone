"""
VoiceClone — Zero-shot voice cloning + conversational AI.
New UI matching prototype design.
"""
import os
import tempfile
import time
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

from prompts import STYLES, DEFAULT_STYLE, get_style_prompt, get_style_list

warnings.filterwarnings("ignore")

_tts = None
_llm = None
_asr = None


def _scan_voices() -> list:
    import yaml
    voices = []
    vdir = Path("voices")
    if not vdir.exists():
        return voices
    for d in sorted(vdir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "reference.wav").exists():
            continue
        sy = d / "speaker.yaml"
        name = d.name
        if sy.exists():
            with open(sy, "r", encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
            name = info.get("name", d.name)
        voices.append({"id": d.name, "name": name})
    return voices


def _voice_id(name: str) -> str:
    for v in _scan_voices():
        if v["name"] == name:
            return v["id"]
    return ""


def _get_tts():
    global _tts
    if _tts is not None:
        return _tts
    from engines.tts_engine import TTSEngine
    _tts = TTSEngine()
    return _tts


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from engines.llm_engine import LLMEngine
    _llm = LLMEngine()
    return _llm


def _get_asr():
    global _asr
    if _asr is not None:
        return _asr
    from engines.asr_engine import ASREngine
    _asr = ASREngine()
    return _asr


def _save_audio(audio_tensor, sample_rate: int) -> str:
    audio_np = audio_tensor.cpu().numpy()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_np, sample_rate)
    tmp.close()
    return tmp.name


# ── Callbacks ──

def on_mode_change(mode: str):
    is_chat = mode in ("语音聊天", "发送短信")
    return (
        gr.Column(visible=(mode == "语音聊天")),
        gr.Column(visible=(mode == "发送短信")),
        gr.Column(visible=(mode == "朗读内容")),
        gr.Dropdown(visible=is_chat),
    )


def on_voice_chat(audio_path: str, voice_name: str, style_name: str):
    if audio_path is None:
        return None, ""

    tts = _get_tts()
    llm = _get_llm()
    asr_engine = _get_asr()
    vid = _voice_id(voice_name)

    if not vid:
        return None, f"未找到声音: {voice_name}"

    transcript = asr_engine.recognize(audio_path)
    if not transcript:
        return None, "未检测到语音。"

    prompt = get_style_prompt(style_name)
    try:
        reply = llm.chat(transcript, prompt)
    except Exception as e:
        return None, f"LLM 错误: {e}"

    try:
        audio = tts.synthesize(reply, vid)
        audio_path_out = _save_audio(audio, tts.sample_rate)
        return audio_path_out, f"「{reply}」"
    except Exception as e:
        return None, f"TTS 错误: {e}"


def on_sms(text: str, voice_name: str, style_name: str):
    if not text.strip():
        return None, ""

    tts = _get_tts()
    llm = _get_llm()
    vid = _voice_id(voice_name)

    if not vid:
        return None, f"未找到声音: {voice_name}"

    prompt = get_style_prompt(style_name)
    try:
        reply = llm.chat(text, prompt)
    except Exception as e:
        return None, f"LLM 错误: {e}"

    try:
        audio = tts.synthesize(reply, vid)
        audio_path_out = _save_audio(audio, tts.sample_rate)
        return audio_path_out, f"「{reply}」"
    except Exception as e:
        return None, f"TTS 错误: {e}"


def on_read(text: str, voice_name: str):
    if not text.strip():
        return None, ""

    tts = _get_tts()
    vid = _voice_id(voice_name)

    if not vid:
        return None, f"未找到声音: {voice_name}"

    try:
        audio = tts.synthesize(text, vid)
        audio_path_out = _save_audio(audio, tts.sample_rate)
        return audio_path_out, f"朗读:「{text}」"
    except Exception as e:
        return None, f"TTS 错误: {e}"


# ── Waveform HTML + JS (Web Audio API driven) ──

WAVEFORM_HTML = """
<div id="waveform-wrap" style="width:100%;height:260px;position:relative;margin-bottom:6px;">
  <canvas id="waveform-canvas" style="width:100%;height:260px;display:block;"></canvas>
</div>

<script>
(function(){
const canvas = document.getElementById('waveform-canvas');
const ctx = canvas.getContext('2d');
const BAR_COUNT = 45;
const DPR = window.devicePixelRatio || 1;

function resizeWave() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * DPR;
  canvas.height = rect.height * DPR;
  ctx.scale(DPR, DPR);
}
resizeWave();
new ResizeObserver(resizeWave).observe(canvas.parentElement);

const phases = Array.from({length:BAR_COUNT},()=>Math.random()*Math.PI*2);
const speeds = Array.from({length:BAR_COUNT},()=>0.5+Math.random()*1.2);

// State
let targetVolume = 0;
let currentVolume = 0;
const BREATH_AMP = 0.055;
const SPEAK_AMP = 0.75;
const VOLUME_LERP = 0.06;

// Audio analyser
let audioCtx = null;
let analyser = null;
let freqData = null;
let isPlaying = false;

function setupAudioContext() {
  // Find Gradio's audio element
  const audios = document.querySelectorAll('audio');
  for (const audio of audios) {
    if (audio.dataset.waveformHooked) continue;
    audio.dataset.waveformHooked = '1';

    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      analyser.smoothingTimeConstant = 0.5;
      freqData = new Uint8Array(analyser.frequencyBinCount);
    }

    try {
      const source = audioCtx.createMediaElementSource(audio);
      source.connect(analyser);
      analyser.connect(audioCtx.destination);
    } catch(e) { /* already connected */ }

    audio.addEventListener('play', () => {
      if (audioCtx.state === 'suspended') audioCtx.resume();
      isPlaying = true;
    });
    audio.addEventListener('ended', () => { isPlaying = false; });
    audio.addEventListener('pause', () => { isPlaying = false; });
  }
}

let time = 0;
function drawWave() {
  requestAnimationFrame(drawWave);
  const dt = 0.016;
  time += dt;

  // Periodically check for new audio elements
  if (Math.floor(time * 10) % 20 === 0) setupAudioContext();

  // Get live volume from analyser if playing
  let liveVol = 0;
  if (isPlaying && analyser && freqData) {
    analyser.getByteFrequencyData(freqData);
    let sum = 0;
    for (let i = 0; i < freqData.length; i++) sum += freqData[i];
    liveVol = (sum / freqData.length) / 255;
    liveVol = Math.pow(liveVol, 0.6); // perceptual curve
  }

  // Blend target: live audio volume when playing, 0 when not
  targetVolume = liveVol > 0.02 ? liveVol : 0;
  currentVolume += (targetVolume - currentVolume) * VOLUME_LERP;

  const W = canvas.width / DPR;
  const H = canvas.height / DPR;
  const CY = H / 2;
  const barW = W / BAR_COUNT * 0.50;
  const gap = W / BAR_COUNT * 0.50;

  ctx.clearRect(0, 0, W, H);

  // Baseline
  ctx.strokeStyle = '#e8e8e3';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, CY);
  ctx.lineTo(W, CY);
  ctx.stroke();

  for (let i = 0; i < BAR_COUNT; i++) {
    const x = i * (barW + gap) + gap / 2;
    const t = time * speeds[i];
    const organic = Math.sin(t * 2.5 + phases[i]);
    const noise = Math.sin(t * 5.1 + phases[i] * 1.7) * 0.4 +
                  Math.sin(t * 9.3 + phases[i] * 0.9) * 0.25 +
                  Math.sin(t * 15.7 + phases[i] * 0.4) * 0.15;
    const vibrato = Math.sin(i * 0.3 + time * 0.4) * 0.06 * currentVolume;

    // Map bar to frequency bin for live audio
    let liveBar = 0;
    if (liveVol > 0.02 && freqData) {
      const binIdx = Math.floor(i / BAR_COUNT * freqData.length);
      liveBar = (freqData[binIdx] || 0) / 255;
      liveBar = Math.pow(liveBar, 0.6);
    }

    const breathe = organic * 0.045 + noise * 0.012;
    const speakLive = liveBar * 0.65 + organic * 0.15 * liveVol;
    const speak = currentVolume > 0.05 ? speakLive : (organic * 0.55 + noise * 0.22 + vibrato) * currentVolume;
    const amplitude = breathe + speak;

    const halfH = Math.abs(amplitude) * H * 0.45;
    const top = CY - halfH;
    const bot = CY + halfH;

    const dist = Math.abs(i - BAR_COUNT / 2) / (BAR_COUNT / 2);
    const alpha = 0.6 - dist * 0.35;

    ctx.fillStyle = `rgba(40,40,40,${alpha})`;
    const r = barW * 0.45;
    ctx.beginPath();
    ctx.moveTo(x, top + r);
    ctx.arcTo(x, top, x + barW, top, r);
    ctx.arcTo(x + barW, top, x + barW, bot, r);
    ctx.arcTo(x + barW, bot, x, bot, r);
    ctx.arcTo(x, bot, x, top, r);
    ctx.closePath();
    ctx.fill();
  }
}
drawWave();
})();
</script>
"""

# ── CSS ──

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500&display=swap');

* { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif !important; }

body, .gradio-container { background: #fafaf7 !important; }

/* Hide default footer */
footer { display: none !important; }

/* ── App container ── */
.gradio-container { max-width: 680px !important; margin: 0 auto !important; }
.contain { padding: 40px 40px 30px !important; }

/* ── Voice dropdown (small, light, centered) ── */
#voice-selector {
  display: flex !important; justify-content: center !important;
  margin-bottom: 242px !important; margin-top: 6px !important;
}
#voice-selector .wrap { max-width: 130px !important; }
#voice-selector select, #voice-selector input[type="text"] {
  background: #f6f6f3 !important; border: 1px solid #e8e8e2 !important;
  color: #aaa !important; font-size: 12px !important;
  padding: 4px 24px 4px 10px !important; border-radius: 6px !important;
  text-align: center !important; letter-spacing: 0.5px !important;
  min-height: auto !important; height: auto !important;
}
#voice-selector label { display: none !important; }

/* ── Control row (mode tabs + style) ── */
#control-row {
  display: flex !important; align-items: center !important;
  justify-content: center !important; gap: 16px !important;
  margin-bottom: 36px !important;
}

/* ── Mode radio (pill style) ── */
#mode-radio {
  background: #ecece6 !important; border-radius: 9px !important;
  padding: 3px !important; border: none !important;
}
#mode-radio .wrap { gap: 0 !important; }
#mode-radio label {
  padding: 8px 22px !important; border-radius: 7px !important;
  font-size: 13px !important; color: #aaa !important;
  cursor: pointer !important; background: transparent !important;
  margin: 0 !important; border: none !important;
  box-shadow: none !important;
}
#mode-radio label.selected {
  background: #fff !important; color: #1a1a1a !important;
  font-weight: 500 !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
#mode-radio input[type="radio"] { display: none !important; }

/* ── Style dropdown ── */
#style-selector .wrap { max-width: 150px !important; }
#style-selector select, #style-selector input[type="text"] {
  background: #f6f6f3 !important; border: 1px solid #e4e4dd !important;
  color: #aaa !important; font-size: 12px !important;
  padding: 6px 28px 6px 12px !important; border-radius: 8px !important;
  text-align: center !important; min-height: auto !important;
}
#style-selector label { display: none !important; }

/* ── Mic button (hidden, we use HTML) ── */
#mic-audio { display: none !important; }

/* ── Text input ── */
#sms-input input, #read-input input {
  padding: 14px 20px !important; border: 1px solid #e0e0da !important;
  border-radius: 24px !important; background: #fff !important;
  font-size: 15px !important; outline: none !important;
}
#sms-input input::placeholder, #read-input input::placeholder { color: #c8c8c2 !important; }
#sms-input input:focus, #read-input input:focus { border-color: #b0b0b0 !important; }

/* ── Send button ── */
#sms-btn, #read-btn {
  width: 48px !important; height: 48px !important; min-width: 48px !important;
  border-radius: 50% !important; border: 1px solid #e0e0da !important;
  background: #fff !important; cursor: pointer !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  font-size: 18px !important; padding: 0 !important;
}
#sms-btn:hover, #read-btn:hover {
  background: #1a1a1a !important; border-color: #1a1a1a !important; color: #fff !important;
}

/* ── Response toast ── */
#response-toast .prose { text-align: center !important; font-size: 14px !important;
  color: #aaa !important; padding: 4px 0 !important; min-height: 22px !important; }
#response-toast { margin-top: 14px !important; }

/* ── Mic custom button ── */
.mic-btn-wrap { display: flex; justify-content: center; }
.mic-btn {
  width: 88px; height: 88px; border-radius: 50%;
  border: 2px solid #e0e0da; background: #fff;
  cursor: pointer; display: flex; align-items: center;
  justify-content: center; transition: all 0.25s;
  position: relative; font-size: 24px; color: #1a1a1a;
}
.mic-btn:hover { border-color: #b0b0b0; background: #f4f4f2; }
.mic-btn.recording {
  border-color: #e04545; background: #fef5f5; color: #e04545;
  animation: mic-pulse 1.2s ease-in-out infinite;
}
@keyframes mic-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(224,69,69,0.2); }
  50% { box-shadow: 0 0 0 16px rgba(224,69,69,0.02); }
}

/* ── Hide Gradio default labels ── */
#voice-chat-panel > label, #sms-panel > label, #read-panel > label { display: none !important; }
"""


def create_demo():
    style_list = get_style_list()
    voice_choices = [v["name"] for v in _scan_voices()] or ["彪", "喆", "鑫"]

    with gr.Blocks(
        title="VoiceClone",
        css=CSS,
        head='<meta name="viewport" content="width=device-width, initial-scale=1.0">',
    ) as demo:

        # ── Waveform ──
        gr.HTML(WAVEFORM_HTML)

        # ── Voice Selector ──
        voice_dd = gr.Dropdown(
            choices=voice_choices,
            value=voice_choices[0] if voice_choices else None,
            elem_id="voice-selector",
            show_label=False,
            interactive=True,
        )

        # ── Mode + Style Row ──
        with gr.Row(elem_id="control-row"):
            mode_radio = gr.Radio(
                choices=["语音聊天", "发送短信", "朗读内容"],
                value="语音聊天",
                elem_id="mode-radio",
                show_label=False,
                interactive=True,
            )
            style_dd = gr.Dropdown(
                choices=style_list,
                value=DEFAULT_STYLE,
                elem_id="style-selector",
                show_label=False,
                visible=True,
                interactive=True,
            )

        # ── Interaction Area ──
        with gr.Column(visible=True, elem_id="voice-chat-panel") as voice_panel:
            gr.HTML("""
            <div class="mic-btn-wrap">
              <button class="mic-btn" id="custom-mic-btn" type="button">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                  <rect x="8" y="2" width="8" height="13" rx="4"/>
                  <path d="M4 11a8 8 0 0 0 16 0"/>
                  <line x1="12" y1="18" x2="12" y2="22"/>
                </svg>
              </button>
            </div>
            <script>
            (function(){
              const micBtn = document.getElementById('custom-mic-btn');
              let recording = false;
              micBtn.addEventListener('click', () => {
                recording = !recording;
                micBtn.classList.toggle('recording', recording);
                // Trigger Gradio's hidden mic component
                const micAudio = document.querySelector('#mic-audio audio') ||
                                 document.querySelector('#mic-audio button');
                if (micAudio) micAudio.click();
              });
            })();
            </script>
            """)
            mic_audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                elem_id="mic-audio",
                show_label=False,
            )

        with gr.Column(visible=False, elem_id="sms-panel") as sms_panel:
            with gr.Row():
                sms_input = gr.Textbox(
                    placeholder="输入消息...",
                    elem_id="sms-input",
                    show_label=False,
                    scale=5,
                )
                sms_btn = gr.Button("➤", elem_id="sms-btn", scale=1, size="sm")

        with gr.Column(visible=False, elem_id="read-panel") as read_panel:
            with gr.Row():
                read_input = gr.Textbox(
                    placeholder="输入要朗读的文字...",
                    elem_id="read-input",
                    show_label=False,
                    scale=5,
                )
                read_btn = gr.Button("➤", elem_id="read-btn", scale=1, size="sm")

        # ── Output ──
        output_audio = gr.Audio(
            autoplay=True,
            visible=False,
            show_label=False,
        )
        response_toast = gr.Markdown("", elem_id="response-toast")

        # ── Events ──

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[voice_panel, sms_panel, read_panel, style_dd],
        )

        mic_audio.stop_recording(
            fn=on_voice_chat,
            inputs=[mic_audio, voice_dd, style_dd],
            outputs=[output_audio, response_toast],
        )

        mic_audio.clear(
            fn=lambda: (None, ""),
            inputs=[],
            outputs=[output_audio, response_toast],
        )

        sms_btn.click(
            fn=on_sms,
            inputs=[sms_input, voice_dd, style_dd],
            outputs=[output_audio, response_toast],
        )
        sms_input.submit(
            fn=on_sms,
            inputs=[sms_input, voice_dd, style_dd],
            outputs=[output_audio, response_toast],
        )

        read_btn.click(
            fn=on_read,
            inputs=[read_input, voice_dd],
            outputs=[output_audio, response_toast],
        )
        read_input.submit(
            fn=on_read,
            inputs=[read_input, voice_dd],
            outputs=[output_audio, response_toast],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(default_concurrency_limit=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
        theme=gr.themes.Soft(),
    )
