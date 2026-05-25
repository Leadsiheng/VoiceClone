"""
TTS Engine — IndexTTS2 zero-shot voice cloning via subprocess bridge.
The bridge runs inside the index-tts uv environment for dependency isolation.
Communication via JSON over stdin/stdout pipes.

Supports:
  - Voice cloning from reference audio (3-15s)
  - Emotion control via emo_vector or emo_audio_prompt
  - Multiple speakers via voices/ directory
  - Amplitude envelope extraction for waveform animation
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml


class TTSEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        tts_config = config["tts"]
        self.voices_dir = Path(tts_config.get("voices_dir", "./voices"))
        self.sample_rate = tts_config.get("sample_rate", 24000)

        # IndexTTS2 paths
        self.index_tts_dir = Path(tts_config.get("index_tts_dir", "../index-tts")).resolve()
        self.cfg_path = tts_config.get("cfg_path", "checkpoints/config.yaml")
        self.model_dir = tts_config.get("model_dir", "checkpoints")
        self.use_fp16 = tts_config.get("use_fp16", False)
        self.use_cuda_kernel = tts_config.get("use_cuda_kernel", False)
        self.use_deepspeed = tts_config.get("use_deepspeed", False)

        if not self.index_tts_dir.exists():
            raise FileNotFoundError(
                f"index-tts directory not found: {self.index_tts_dir}\n"
                "Clone it first: git clone https://github.com/index-tts/index-tts.git ../index-tts"
            )

        self.bridge_path = self.index_tts_dir / "indextts_bridge.py"
        if not self.bridge_path.exists():
            self._install_bridge()

        self._proc = None
        self._start_bridge()

        self.speakers = self._load_speakers()
        atexit.register(self._cleanup)

    def _install_bridge(self):
        src = Path(__file__).resolve().parent.parent / "indextts_bridge.py"
        if src.exists():
            shutil.copy(str(src), str(self.bridge_path))
        else:
            raise FileNotFoundError(
                f"indextts_bridge.py not found at {src}. "
                "Please ensure the file exists in the VoiceClone project root."
            )

    def _start_bridge(self):
        self._proc = subprocess.Popen(
            ["uv", "run", "python", str(self.bridge_path)],
            cwd=str(self.index_tts_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._send({
            "cmd": "init",
            "cfg_path": self.cfg_path,
            "model_dir": self.model_dir,
            "use_fp16": self.use_fp16,
            "use_cuda_kernel": self.use_cuda_kernel,
            "use_deepspeed": self.use_deepspeed,
        })

    def _send(self, msg: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("IndexTTS2 bridge process has terminated")

        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = ""
            try:
                stderr = self._proc.stderr.read()
            except Exception:
                pass
            raise RuntimeError(f"Bridge process closed unexpectedly. Stderr: {stderr}")

        resp = json.loads(resp_line.strip())
        if resp.get("status") != "ok":
            raise RuntimeError(f"Bridge error: {resp.get('message', 'unknown')}")

        return resp

    def _cleanup(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"cmd": "quit"})
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

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

            speakers[speaker_dir.name] = {
                "id": speaker_dir.name,
                "name": name,
                "ref_audio": str(ref_audio.resolve()),
                "ref_text": info.get("reference_text", ""),
                "system_prompt": info.get("system_prompt", ""),
            }

        return speakers

    def reload_speakers(self) -> dict:
        self.speakers = self._load_speakers()
        return self.speakers

    def add_speaker(
        self,
        speaker_id: str,
        name: str,
        audio_path: str,
        ref_text: str = "",
        system_prompt: str = "",
    ) -> bool:
        speaker_dir = self.voices_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(audio_path).suffix
        dest = speaker_dir / f"reference{ext}"
        shutil.copy(audio_path, str(dest))

        info = {
            "name": name,
            "reference_text": ref_text,
            "system_prompt": system_prompt,
        }
        with open(speaker_dir / "speaker.yaml", "w", encoding="utf-8") as f:
            yaml.dump(info, f, allow_unicode=True, default_flow_style=False)

        self.reload_speakers()
        return True

    def synthesize(
        self,
        text: str,
        speaker_id: str,
        emo_vector: Optional[list] = None,
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
    ) -> torch.Tensor:
        if speaker_id not in self.speakers:
            raise ValueError(
                f"Unknown speaker: {speaker_id}. "
                f"Available: {list(self.speakers.keys())}"
            )

        speaker = self.speakers[speaker_id]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_path = tmp.name

        infer_msg = {
            "cmd": "infer",
            "spk_audio_prompt": speaker["ref_audio"],
            "text": text,
            "output_path": output_path,
            "verbose": False,
        }

        if emo_vector is not None:
            infer_msg["emo_vector"] = emo_vector
        if emo_audio_prompt is not None:
            infer_msg["emo_audio_prompt"] = emo_audio_prompt
        if emo_alpha != 1.0:
            infer_msg["emo_alpha"] = emo_alpha

        self._send(infer_msg)

        audio_np, file_sr = sf.read(output_path)
        os.unlink(output_path)

        audio = torch.from_numpy(audio_np).float()

        if audio.ndim > 1:
            audio = audio.mean(dim=-1)

        if file_sr != self.sample_rate:
            import torchaudio
            audio = torchaudio.functional.resample(audio, file_sr, self.sample_rate)

        return audio

    def extract_envelope(self, audio: torch.Tensor, num_frames: int = 80) -> list:
        audio_np = audio.detach().cpu().numpy()
        total = len(audio_np)
        frame_size = max(total // num_frames, 1)

        envelope = []
        for i in range(num_frames):
            start = i * frame_size
            end = min(start + frame_size, total)
            if start >= total:
                break
            rms = float(np.sqrt(np.mean(audio_np[start:end] ** 2)))
            envelope.append(rms)

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
