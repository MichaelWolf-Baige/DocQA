"""
BGE-small-zh 嵌入器
"""
from typing import List
from sentence_transformers import SentenceTransformer
from .base import Chunk, Embedder


class BGEEmbedder(Embedder):
    """BAAI/bge-small-zh-v1.5 嵌入模型"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model = SentenceTransformer(model_path, device=device, local_files_only=True)

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [c.text for c in chunks]
        vectors = self.model.encode(texts)
        for i, c in enumerate(chunks):
            c.embedding = vectors[i].tolist()
        return chunks

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
