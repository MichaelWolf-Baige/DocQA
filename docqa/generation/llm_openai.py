"""
OpenAI 兼容 LLM 后端
====================
支持所有兼容 OpenAI /v1/chat/completions 的 API：
  - OpenAI 官方 API
  - Ollama (http://localhost:11434/v1)
  - DeepSeek API
  - vLLM / TGI / LocalAI 等自部署服务
  - 任何 OpenAI-compatible endpoint

用法：
  # config.yaml
  generation:
    llm:
      type: openai
      model: qwen2.5:7b              # 模型名
      base_url: http://localhost:11434/v1   # API 地址
      api_key: ollama                 # 可选，本地模型通常不需要
      temperature: 0.1
      max_tokens: 1024
      timeout: 300
"""
import os
from .base import LLM


class OpenAICompatibleLLM(LLM):
    """通过 OpenAI 兼容 API 调用模型（本地或云端）"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "sk-placeholder",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        """延迟加载 OpenAI client，避免导入时的副作用"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url + "/v1" if not self.base_url.endswith("/v1") else self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _clear_proxy():
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)

    def generate(self, prompt: str) -> str:
        self._clear_proxy()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            return f"[ERROR: {e}]"
