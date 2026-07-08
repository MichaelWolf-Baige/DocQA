"""
混合检索器：BM25 + 向量，RRF 融合
"""
from typing import List
from docqa.ingestion.base import Chunk
from docqa.ingestion.base import Embedder, VectorStore
from .base import Retriever
from .dense import DenseRetriever
from .bm25 import BM25Retriever


class HybridRetriever(Retriever):
    """BM25 + 稠密向量 → RRF 融合"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        rrf_k: int = 60,
    ):
        self.dense = DenseRetriever(vector_store, embedder)
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k

    def build_bm25_index(self, chunks: List[Chunk]) -> None:
        self.bm25.build_index(chunks)

    def search(self, query: str, top_k: int) -> List[Chunk]:
        # 并行检索
        dense_results = self.dense.search(query, top_k=50)
        bm25_results = self.bm25.search(query, top_k=50)

        # RRF 融合
        rrf_scores = {}
        chunk_map = {}

        for rank, c in enumerate(dense_results, start=1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            chunk_map[c.chunk_id] = c

        for rank, c in enumerate(bm25_results, start=1):
            rrf_scores[c.chunk_id] = rrf_scores.get(c.chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            if c.chunk_id not in chunk_map:
                chunk_map[c.chunk_id] = c

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, rrf_score in sorted_ids[:top_k]:
            c = chunk_map[cid]
            c.metadata['rrf_score'] = round(rrf_score, 6)
            results.append(c)

        return results
