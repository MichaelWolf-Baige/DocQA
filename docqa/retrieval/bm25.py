"""
BM25 关键词检索器
"""
import jieba
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi
from docqa.ingestion.base import Chunk
from .base import Retriever


class BM25Retriever(Retriever):
    """基于 BM25 的关键词检索器"""

    def __init__(self):
        self.bm25 = None
        self.chunk_texts: List[str] = []
        self.chunk_ids: List[int] = []
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in jieba.cut(text) if t.strip()]

    def build_index(self, chunks: List[Chunk]) -> None:
        """对 chunk 列表构建 BM25 索引"""
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._tokenized_corpus = [self.tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> List[Chunk]:
        if self.bm25 is None:
            return []

        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(Chunk(
                    chunk_id=self.chunk_ids[idx],
                    text=self.chunk_texts[idx],
                    source_page=-1,  # BM25 不关心页码
                    metadata={'score': round(float(scores[idx]), 4)}
                ))
        return results
