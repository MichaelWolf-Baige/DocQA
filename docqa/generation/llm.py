"""
Ollama LLM 调用封装
"""
import os
import requests
from .base import LLM


class OllamaLLM(LLM):
    """通过 Ollama 原生 API 调用本地模型"""

    API_URL = 'http://localhost:11434/api/chat'

    def __init__(
        self,
        model: str = 'qwen2.5:7b',
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 300,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @staticmethod
    def _clear_proxy():
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
            os.environ.pop(key, None)

    def generate(self, prompt: str) -> str:
        body = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False,
            'options': {
                'temperature': self.temperature,
                'num_predict': self.max_tokens,
            }
        }

        try:
            self._clear_proxy()
            r = requests.post(self.API_URL, json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            content = data.get('message', {}).get('content', '')
            return content.strip() if content else ''
        except requests.exceptions.Timeout:
            return '[ERROR: LLM 响应超时]'
        except Exception as e:
            return f'[ERROR: {e}]'
