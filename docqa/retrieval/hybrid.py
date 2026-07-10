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
        dense_results = self.dense.search(query, top_k=50)
        bm25_results = self.bm25.search(query, top_k=50)

        # RRF 融合。复合键 (source_file, chunk_id) 避免多文档 id 碰撞
        rrf_scores = {}
        chunk_map = {}

        for rank, c in enumerate(dense_results, start=1):
            key = (getattr(c, 'source_file', '') or '', c.chunk_id)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            chunk_map[key] = c

        for rank, c in enumerate(bm25_results, start=1):
            key = (getattr(c, 'source_file', '') or '', c.chunk_id)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            if key not in chunk_map:
                chunk_map[key] = c

        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, rrf_score in sorted_keys[:top_k]:
            c = chunk_map[key]
            c.metadata['rrf_score'] = round(rrf_score, 6)
            results.append(c)

        return results
