"""
Ollama LLM 调用封装
====================
P1 修复: 响应体校验 + 空内容检测 + 代理绕过不 mutable-env-pop
"""
import os
import json
import requests
from .base import LLM


_NO_PROXY_ENV = {'HTTP_PROXY': '', 'HTTPS_PROXY': '', 'http_proxy': '', 'https_proxy': '',
                 'NO_PROXY': 'localhost,127.0.0.1,::1'}


class OllamaLLM(LLM):
    """通过 Ollama 原生 API 调用本地模型"""

    API_URL = 'http://localhost:11434/api/chat'

    def __init__(self, model: str = 'qwen2.5:7b', temperature: float = 0.1,
                 max_tokens: int = 1024, timeout: int = 300):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._session = requests.Session()
        # 绕过代理——用 Session 级别 trust_env=False，不修改全局 env
        self._session.trust_env = False

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
            r = self._session.post(self.API_URL, json=body, timeout=self.timeout)
            r.raise_for_status()

            ct = r.headers.get('Content-Type', '')
            if 'json' not in ct and ct:
                return f'[ERROR: Ollama returned non-JSON response (Content-Type: {ct})]'

            data = r.json()
            content = data.get('message', {}).get('content', '')

            if not content or not content.strip():
                done_reason = data.get('done_reason', 'unknown')
                eval_count = data.get('eval_count', 0)
                return (
                    f'[ERROR: LLM returned empty response '
                    f'(done_reason={done_reason}, eval_count={eval_count}). '
                    f'Try a larger model or reduce prompt length.]'
                )

            return content.strip()

        except requests.exceptions.Timeout:
            return f'[ERROR: LLM timeout after {self.timeout}s — Ollama may be overloaded]'
        except requests.exceptions.ConnectionError:
            return '[ERROR: Cannot connect to Ollama — is `ollama serve` running?]'
        except json.JSONDecodeError as e:
            return f'[ERROR: Ollama returned invalid JSON: {e}]'
        except Exception as e:
            return f'[ERROR: {type(e).__name__}: {e}]'
