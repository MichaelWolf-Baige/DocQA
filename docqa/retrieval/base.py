"""
检索管线：抽象接口
==================
Retriever 和 Reranker 接口定义。
"""
from abc import ABC, abstractmethod
from typing import List
from docqa.ingestion.base import Chunk


class Retriever(ABC):
    """检索器：query → top_k chunks"""
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Chunk]:
        ...


class Reranker(ABC):
    """重排序器：candidates → top_k"""
    @abstractmethod
    def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        ...
