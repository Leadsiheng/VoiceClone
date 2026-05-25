"""
LLM Engine — supports DeepSeek API and Ollama local models.
Each speaker voice can have its own system prompt.
"""
from typing import Optional

import requests
import yaml
from openai import OpenAI


class LLMEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        llm_config = config["llm"]
        self.backend = llm_config.get("backend", "deepseek")

        if self.backend == "deepseek":
            ds_config = llm_config["deepseek"]
            self.client = OpenAI(
                api_key=ds_config["api_key"],
                base_url=ds_config.get("api_base", "https://api.deepseek.com/v1"),
            )
            self.model = ds_config.get("model", "deepseek-chat")
            self.max_tokens = ds_config.get("max_tokens", 512)
            self.temperature = ds_config.get("temperature", 0.7)
        elif self.backend == "ollama":
            ol_config = llm_config["ollama"]
            self.host = ol_config.get("host", "http://localhost:11434")
            self.model = ol_config.get("model", "gemma4:latest")
            self.num_predict = ol_config.get("num_predict", 256)
            self.temperature = ol_config.get("temperature", 0.7)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

        self.histories: dict[str, list[dict]] = {}

    def _call_deepseek(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": self.num_predict,
                    "temperature": self.temperature,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def chat(self, message: str, speaker_id: str, system_prompt: Optional[str] = None) -> str:
        if speaker_id not in self.histories:
            self.histories[speaker_id] = []

            if system_prompt:
                self.histories[speaker_id].append({
                    "role": "system",
                    "content": system_prompt,
                })

        self.histories[speaker_id].append({"role": "user", "content": message})

        if self.backend == "deepseek":
            reply = self._call_deepseek(self.histories[speaker_id])
        else:
            reply = self._call_ollama(self.histories[speaker_id])

        self.histories[speaker_id].append({"role": "assistant", "content": reply})

        return reply

    def reset_history(self, speaker_id: str):
        self.histories.pop(speaker_id, None)

    def set_system_prompt(self, speaker_id: str, prompt: str):
        if speaker_id in self.histories:
            if self.histories[speaker_id] and self.histories[speaker_id][0]["role"] == "system":
                self.histories[speaker_id][0] = {"role": "system", "content": prompt}
            else:
                self.histories[speaker_id].insert(0, {"role": "system", "content": prompt})
        else:
            self.histories[speaker_id] = [{"role": "system", "content": prompt}]
