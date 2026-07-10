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
        self.source_files: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in jieba.cut(text) if t.strip()]

    def build_index(self, chunks: List[Chunk]) -> None:
        """对 chunk 列表构建/重建 BM25 索引"""
        if not chunks:
            self.bm25 = None
            return
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.source_files = [getattr(c, 'source_file', '') for c in chunks]
        self._tokenized_corpus = [self.tokenize(t) for t in self.chunk_texts]

        # 过滤空 token 列表（避免 rank_bm25 的 ZeroDivisionError）
        if len(self._tokenized_corpus) == 0 or all(len(t) == 0 for t in self._tokenized_corpus):
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> List[Chunk]:
        if self.bm25 is None:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0 or scores.max() <= 0:
            return []

        # 归一化到 [0,1]
        max_score = scores.max()
        scores = scores / max_score

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            if scores[idx] <= 0:
                break
            results.append(Chunk(
                chunk_id=self.chunk_ids[idx],
                text=self.chunk_texts[idx],
                source_page=-1,
                source_file=self.source_files[idx] if idx < len(self.source_files) else '',
                metadata={'score': round(float(scores[idx]), 4)}
            ))
        return results
