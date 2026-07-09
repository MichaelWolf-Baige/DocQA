"""
RAG Prompt 构建器
"""
from typing import List
from docqa.ingestion.base import Chunk
from .base import PromptBuilder


class RAGPromptBuilder(PromptBuilder):
    """标准 RAG Prompt：指令 + 上下文 + 问题"""

    DEFAULT_SYSTEM = '你是文档问答助手，只根据用户提供的文档内容作答，引用具体信息，用中文。'

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

    def build(self, query: str, chunks: List[Chunk]) -> str:
        context_parts = []
        for c in chunks:
            # 多文档时显示来源文件名
            if c.source_file:
                part = f'[{c.source_file} 第{c.source_page}页] {c.text}'
            else:
                part = f'[第{c.source_page}页] {c.text}'
            context_parts.append(part)

        context = '\n\n'.join(context_parts)

        prompt = (
            f'从以下文档片段中找出问题的答案，引用文档中的具体信息作答。'
            f'如果确实找不到相关信息，才回复"文档未提及"。\n'
            f'\n'
            f'文档片段：\n{context}\n'
            f'\n'
            f'问题：{query}\n'
            f'答案：'
        )
        return prompt
