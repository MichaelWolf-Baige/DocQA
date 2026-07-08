"""
Cross-encoder 重排序器
"""
from typing import List
from sentence_transformers import CrossEncoder
from docqa.ingestion.base import Chunk
from .base import Reranker


class BGEReranker(Reranker):
    """BAAI/bge-reranker-v2-m3 cross-encoder 精排"""

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', device: str = 'cpu'):
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query: str, candidates: List[Chunk], top_k: int) -> List[Chunk]:
        if not candidates:
            return candidates

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        for i, c in enumerate(candidates):
            c.metadata['rerank_score'] = round(float(scores[i]), 4)

        ranked = sorted(candidates, key=lambda x: x.metadata.get('rerank_score', 0), reverse=True)
        return ranked[:top_k]
