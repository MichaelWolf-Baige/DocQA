"""
生成管线：抽象接口
==================
PromptBuilder 和 LLM 接口定义。
"""
from abc import ABC, abstractmethod
from typing import List
from docqa.ingestion.base import Chunk


class PromptBuilder(ABC):
    """Prompt 构建器：检索结果 + 问题 → prompt 字符串"""
    @abstractmethod
    def build(self, query: str, chunks: List[Chunk]) -> str:
        ...


class LLM(ABC):
    """大语言模型：prompt → 回答"""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...
