"""
LLM Engine — stateless single-turn chat.
Each call is independent, no conversation history.
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
            ds = llm_config["deepseek"]
            self.client = OpenAI(
                api_key=ds["api_key"],
                base_url=ds.get("api_base", "https://api.deepseek.com/v1"),
            )
            self.model = ds.get("model", "deepseek-chat")
            self.max_tokens = ds.get("max_tokens", 256)
            self.temperature = ds.get("temperature", 0.8)
        elif self.backend == "ollama":
            ol = llm_config["ollama"]
            self.host = ol.get("host", "http://localhost:11434")
            self.model = ol.get("model", "gemma4:latest")
            self.num_predict = ol.get("num_predict", 128)
            self.temperature = ol.get("temperature", 0.8)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    def chat(self, message: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        if self.backend == "deepseek":
            return self._call_deepseek(messages)
        return self._call_ollama(messages)

    def _call_deepseek(self, messages: list) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

    def _call_ollama(self, messages: list) -> str:
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
