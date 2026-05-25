"""
VoiceClone Preview — UI demo without model loading.
Run this to see and interact with the interface design.
"""
import time
from pathlib import Path

import gradio as gr
import numpy as np
import yaml


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


def _fake_audio(sample_rate=22050, duration=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return (sample_rate, audio)


def _fake_reply(message: str) -> str:
    return f"[Preview] 这是对「{message}」的模拟回复。实际运行时将由 DeepSeek/Gemma4 生成真实对话内容。"


def text_chat(message: str, voice_name: str, chat_history: list):
    if not message.strip():
        return "", chat_history, None, ""

    time.sleep(0.5)
    reply = _fake_reply(message)
    audio = _fake_audio()
    chat_history.append((message, reply))
    return "", chat_history, audio, ""


def voice_chat(audio_input, voice_name: str, chat_history: list):
    if audio_input is None:
        return None, chat_history, None, ""

    time.sleep(0.8)

    transcript_text = "[预览] 模拟语音识别结果：用户说了一段话。实际运行时将由 FunASR 实时转写。"

    reply = _fake_reply(transcript_text)
    audio = _fake_audio()
    chat_history.append((transcript_text, reply))
    return None, chat_history, audio, ""


def reset_chat(voice_name: str):
    return [], None, ""


def add_voice(name: str, audio_file, ref_text: str, system_prompt: str):
    if not name.strip():
        return "Please enter a voice name.", gr.Dropdown(), None

    if audio_file is None:
        return "Please upload a reference audio file.", gr.Dropdown(), None

    import shutil
    speaker_id = name.lower().replace(" ", "_")
    speaker_dir = Path("voices") / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(audio_file, str):
        shutil.copy(audio_file, speaker_dir / "reference.wav")
    elif isinstance(audio_file, tuple) and len(audio_file) == 2:
        import soundfile as sf
        sr, arr = audio_file
        sf.write(str(speaker_dir / "reference.wav"), arr, sr)

    speaker_info = {
        "name": name,
        "reference_text": ref_text,
        "system_prompt": system_prompt or f"你叫{name}，请自然友好地交谈。",
    }
    with open(speaker_dir / "speaker.yaml", "w", encoding="utf-8") as f:
        yaml.dump(speaker_info, f, allow_unicode=True, default_flow_style=False)

    voices = _scan_voices()
    choices = [v["name"] for v in voices]
    return f"Voice '{name}' added!", gr.Dropdown(choices=choices, value=name), None


def on_voice_change(voice_name: str):
    return reset_chat(voice_name)


def load_voices_on_start():
    voices = _scan_voices()
    choices = [v["name"] for v in voices]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


css = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
.status-text { color: #888; font-size: 0.85em; }
footer { display: none !important; }
"""


def create_demo():
    with gr.Blocks(title="VoiceClone Preview") as demo:
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

                send_btn.click(
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
    demo.queue(default_concurrency_limit=20)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=css,
    )
