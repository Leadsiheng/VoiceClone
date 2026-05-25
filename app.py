"""
VoiceClone — Zero-shot voice cloning + conversational AI.
Web UI powered by Gradio.
"""
import os
import warnings
from pathlib import Path

import gradio as gr
import yaml

warnings.filterwarnings("ignore")

_tts = None
_llm = None
_asr = None


def _scan_voices() -> list:
    voices = []
    voices_dir = Path("voices")
    if not voices_dir.exists():
        return voices
    for d in sorted(voices_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "reference.wav").exists():
            continue
        speaker_yaml = d / "speaker.yaml"
        name = d.name
        if speaker_yaml.exists():
            with open(speaker_yaml, "r", encoding="utf-8") as f:
                info = yaml.safe_load(f) or {}
            name = info.get("name", d.name)
        voices.append({"id": d.name, "name": name})
    return voices


def _speaker_id_from_name(name: str) -> str:
    for sp in _scan_voices():
        if sp["name"] == name:
            return sp["id"]
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


def _speaker_info(speaker_id: str) -> dict:
    tts = _get_tts()
    return tts.speakers.get(speaker_id, {})


def load_voices_on_start():
    voices = _scan_voices()
    choices = [v["name"] for v in voices]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def text_chat(message: str, voice_name: str, chat_history: list):
    if not message.strip():
        return "", chat_history, None, ""

    speaker_id = _speaker_id_from_name(voice_name)
    if not speaker_id:
        return "", chat_history, None, f"Unknown voice: {voice_name}"

    tts = _get_tts()
    llm = _get_llm()

    sp = _speaker_info(speaker_id)
    system_prompt = sp.get("system_prompt")

    try:
        reply = llm.chat(message, speaker_id, system_prompt)
    except Exception as e:
        chat_history.append((message, f"[LLM Error] {e}"))
        return "", chat_history, None, ""

    try:
        audio = tts.synthesize(reply, speaker_id)
        audio_np = audio.cpu().numpy()
        audio_output = (tts.sample_rate, audio_np)
    except Exception as e:
        chat_history.append((message, reply))
        return "", chat_history, None, f"[TTS Error] {e}"

    chat_history.append((message, reply))
    return "", chat_history, audio_output, ""


def voice_chat(audio_input, voice_name: str, chat_history: list):
    empty = None, chat_history, None, ""

    if audio_input is None:
        return empty

    speaker_id = _speaker_id_from_name(voice_name)
    if not speaker_id:
        return None, chat_history, None, f"Unknown voice: {voice_name}"

    tts = _get_tts()
    llm = _get_llm()
    asr_engine = _get_asr()

    try:
        if isinstance(audio_input, str):
            transcript = asr_engine.recognize(audio_input)
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            sr, arr = audio_input
            transcript = asr_engine.recognize_array(arr, sr)
        else:
            return None, chat_history, None, "[ASR] Invalid audio format."
    except Exception as e:
        return None, chat_history, None, f"[ASR Error] {e}"

    if not transcript:
        return None, chat_history, None, "[ASR] No speech detected."

    sp = _speaker_info(speaker_id)
    system_prompt = sp.get("system_prompt")

    try:
        reply = llm.chat(transcript, speaker_id, system_prompt)
    except Exception as e:
        chat_history.append((transcript, f"[LLM Error] {e}"))
        return None, chat_history, None, ""

    try:
        audio = tts.synthesize(reply, speaker_id)
        audio_np = audio.cpu().numpy()
        audio_output = (tts.sample_rate, audio_np)
    except Exception as e:
        chat_history.append((transcript, reply))
        return None, chat_history, None, f"[TTS Error] {e}"

    chat_history.append((transcript, reply))
    return None, chat_history, audio_output, ""


def reset_chat(voice_name: str):
    speaker_id = _speaker_id_from_name(voice_name)
    if speaker_id:
        try:
            llm = _get_llm()
            llm.reset_history(speaker_id)
        except Exception:
            pass
    return [], None, ""


def add_voice(name: str, audio_file, ref_text: str, system_prompt: str):
    if not name.strip():
        return "Please enter a voice name.", gr.Dropdown(), None

    if audio_file is None:
        return "Please upload a reference audio file.", gr.Dropdown(), None

    speaker_id = name.lower().replace(" ", "_")

    if isinstance(audio_file, str):
        audio_path = audio_file
    elif isinstance(audio_file, tuple) and len(audio_file) == 2:
        import tempfile
        import soundfile as sf
        sr, arr = audio_file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, arr, sr)
        audio_path = tmp.name
        tmp.close()
    else:
        return "Invalid audio format.", gr.Dropdown(), None

    tts = _get_tts()
    try:
        tts.add_speaker(speaker_id, name, audio_path, ref_text, system_prompt)
    except Exception as e:
        return f"Error adding voice: {e}", gr.Dropdown(), None

    voices = _scan_voices()
    choices = [v["name"] for v in voices]
    return f"Voice '{name}' added successfully!", gr.Dropdown(choices=choices, value=name), None


def on_voice_change(voice_name: str):
    return reset_chat(voice_name)


css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.status-text { color: #888; font-size: 0.85em; }
footer { display: none !important; }
"""


def create_demo():
    with gr.Blocks(title="VoiceClone", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# VoiceClone")
        gr.Markdown("Zero-shot voice cloning + conversational AI. Text or voice input, cloned voice output.")

        with gr.Row():
            voice_selector = gr.Dropdown(
                label="Voice",
                choices=[],
                interactive=True,
                scale=3,
            )
            reset_btn = gr.Button("Reset Chat", variant="secondary", scale=1)

        status_text = gr.Markdown("", elem_classes=["status-text"])

        with gr.Tabs():
            with gr.TabItem("Text Chat"):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                audio_output = gr.Audio(label="Voice Reply", autoplay=True, type="numpy")

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here and press Enter...",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                text_chat_event = send_btn.click(
                    fn=text_chat,
                    inputs=[text_input, voice_selector, chatbot],
                    outputs=[text_input, chatbot, audio_output, status_text],
                )
                text_input.submit(
                    fn=text_chat,
                    inputs=[text_input, voice_selector, chatbot],
                    outputs=[text_input, chatbot, audio_output, status_text],
                )

            with gr.TabItem("Voice Chat"):
                voice_chatbot = gr.Chatbot(label="Conversation", height=400)
                voice_audio_output = gr.Audio(label="Voice Reply", autoplay=True, type="numpy")

                voice_input = gr.Audio(
                    label="Your voice",
                    type="numpy",
                    sources=["microphone", "upload"],
                )

                voice_input.stop_recording(
                    fn=voice_chat,
                    inputs=[voice_input, voice_selector, voice_chatbot],
                    outputs=[voice_input, voice_chatbot, voice_audio_output, status_text],
                )

            with gr.TabItem("Voice Management"):
                gr.Markdown("### Add a new voice for cloning")
                gr.Markdown(
                    "Upload 3-15 seconds of clear, natural speech. "
                    "Best results: quiet environment, normal speaking style."
                )

                with gr.Row():
                    with gr.Column():
                        new_name = gr.Textbox(label="Voice Name", placeholder="e.g. Alice")
                        new_audio = gr.Audio(
                            label="Reference Audio (3-15 sec)",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )
                    with gr.Column():
                        new_ref_text = gr.Textbox(
                            label="Reference Text",
                            placeholder="What does the person say in the audio? (optional)",
                            lines=2,
                        )
                        new_system_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="Personality / character description for the AI...",
                            lines=4,
                        )

                add_btn = gr.Button("Add Voice", variant="primary")
                add_status = gr.Markdown("")

                add_btn.click(
                    fn=add_voice,
                    inputs=[new_name, new_audio, new_ref_text, new_system_prompt],
                    outputs=[add_status, voice_selector, new_audio],
                )

        reset_btn.click(
            fn=reset_chat,
            inputs=[voice_selector],
            outputs=[chatbot, audio_output, status_text],
        )

        voice_selector.change(
            fn=on_voice_change,
            inputs=[voice_selector],
            outputs=[chatbot, audio_output, status_text],
        )

        demo.load(
            fn=load_voices_on_start,
            outputs=[voice_selector],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue(max_size=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True,
    )
