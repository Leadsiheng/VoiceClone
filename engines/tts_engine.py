"""
TTS Engine — CosyVoice zero-shot voice cloning.
Supports multiple speakers via reference audio files.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import yaml


class TTSEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tts_config = config["tts"]
        self.voices_dir = Path(tts_config.get("voices_dir", "./voices"))
        self.sample_rate = tts_config.get("sample_rate", 22050)
        self.model_name = tts_config.get("model", "CosyVoice-300M-SFT")
        cosyvoice_path = Path(tts_config.get("cosyvoice_path", "../CosyVoice"))

        sys.path.insert(0, str(cosyvoice_path))
        sys.path.insert(0, str(cosyvoice_path / "third_party" / "Matcha-TTS"))

        from cosyvoice.cli.cosyvoice import CosyVoice

        model_dir = self._resolve_model_path()
        self.cosyvoice = CosyVoice(model_dir, load_jit=False, load_onnx=False, fp16=True)
        self.speakers = self._load_speakers()

    def _resolve_model_path(self) -> str:
        pretrained = Path("pretrained_models") / self.model_name
        if pretrained.exists():
            return str(pretrained)

        from modelscope import snapshot_download

        return snapshot_download(
            f"iic/{self.model_name}",
            cache_dir="models",
        )

    def _load_speakers(self) -> dict:
        speakers = {}
        if not self.voices_dir.exists():
            return speakers

        for speaker_dir in sorted(self.voices_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_yaml = speaker_dir / "speaker.yaml"
            ref_audio = speaker_dir / "reference.wav"

            if not ref_audio.exists():
                continue

            if speaker_yaml.exists():
                with open(speaker_yaml, "r", encoding="utf-8") as f:
                    info = yaml.safe_load(f) or {}
            else:
                info = {}

            name = info.get("name", speaker_dir.name)
            ref_text = info.get("reference_text", "")

            speakers[speaker_dir.name] = {
                "id": speaker_dir.name,
                "name": name,
                "ref_audio": str(ref_audio),
                "ref_text": ref_text,
                "system_prompt": info.get("system_prompt", f"你叫{name}，请自然友好地交谈。回答简洁，每次不超过三句话。"),
            }

        return speakers

    def _load_ref_audio(self, audio_path: str) -> torch.Tensor:
        from cosyvoice.utils.file_utils import load_wav

        return load_wav(audio_path, 16000)

    def reload_speakers(self) -> dict:
        self.speakers = self._load_speakers()
        return self.speakers

    def add_speaker(self, speaker_id: str, name: str, audio_path: str, ref_text: str = "", system_prompt: str = "") -> bool:
        speaker_dir = self.voices_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        ext = Path(audio_path).suffix
        dest_audio = speaker_dir / f"reference{ext}"
        shutil.copy(audio_path, dest_audio)

        speaker_info = {
            "name": name,
            "reference_text": ref_text,
            "system_prompt": system_prompt or f"你叫{name}，请自然友好地交谈。回答简洁，每次不超过三句话。",
        }
        with open(speaker_dir / "speaker.yaml", "w", encoding="utf-8") as f:
            yaml.dump(speaker_info, f, allow_unicode=True, default_flow_style=False)

        self.reload_speakers()
        return True

    def synthesize(self, text: str, speaker_id: str) -> Optional[torch.Tensor]:
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}. Available: {list(self.speakers.keys())}")

        speaker = self.speakers[speaker_id]
        ref_wav = self._load_ref_audio(speaker["ref_audio"])

        output = self.cosyvoice.inference_zero_shot(
            text,
            speaker["ref_text"],
            ref_wav,
            stream=False,
        )

        audio = torch.cat(output["tts_speech"], dim=1).squeeze(0)

        if self.sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, 16000, self.sample_rate)

        return audio

    def extract_envelope(self, audio: torch.Tensor, num_frames: int = 80) -> list:
        """Extract RMS amplitude envelope from audio for waveform animation.
        
        Returns list of floats (0-1) — one per frame.
        """
        audio_np = audio.detach().cpu().numpy()
        total = len(audio_np)
        frame_size = max(total // num_frames, 1)

        envelope = []
        for i in range(num_frames):
            start = i * frame_size
            end = min(start + frame_size, total)
            if start >= total:
                break
            rms = np.sqrt(np.mean(audio_np[start:end] ** 2))
            envelope.append(float(rms))

        if not envelope:
            return [0.0] * num_frames

        max_val = max(envelope) or 1e-8
        envelope = [min(v / max_val, 1.0) for v in envelope]

        if len(envelope) < num_frames:
            envelope.extend([0.0] * (num_frames - len(envelope)))

        return envelope[:num_frames]

    def get_speaker_list(self) -> list:
        return [
            {"id": sp["id"], "name": sp["name"]}
            for sp in self.speakers.values()
        ]
