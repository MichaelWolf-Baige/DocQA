"""
稠密向量检索器
"""
from typing import List
from docqa.ingestion.base import VectorStore, Embedder, Chunk
from .base import Retriever


class DenseRetriever(Retriever):
    """基于向量相似度的检索器"""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.store = vector_store
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> List[Chunk]:
        query_vec = self.embedder.embed_query(query)
        return self.store.search(query_vec, top_k)
