"""
ASR Engine — FunASR Paraformer for Chinese speech recognition.
"""
from pathlib import Path

import numpy as np
import yaml


class ASREngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        asr_config = config["asr"]
        self.model_name = asr_config.get("model", "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        self.vad_model_name = asr_config.get("vad_model", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")

        from funasr import AutoModel

        self.model = AutoModel(
            model=self.model_name,
            vad_model=self.vad_model_name,
            punc_model="ct-punc",
            device="cuda:0" if self._cuda_available() else "cpu",
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def recognize(self, audio_path: str) -> str:
        if isinstance(audio_path, str):
            audio_path = Path(audio_path)

        result = self.model.generate(
            input=str(audio_path),
            batch_size_s=300,
        )

        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text.strip()

        return ""

    def recognize_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        result = self.model.generate(
            input=(sample_rate, audio),
            batch_size_s=300,
        )

        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text.strip()

        return ""
