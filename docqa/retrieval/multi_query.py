"""
多查询检索器
============
对改写后的多个查询变体分别检索，结果 RRF 融合。
解决单查询无法覆盖文档中不同表述的问题。

与 HybridRetriever 的区别：
- HybridRetriever 是单查询的多种检索方式融合（BM25 + 向量）
- MultiQueryRetriever 是多查询的单种/多种检索方式融合
- 两者正交——一个横向（多检索器），一个纵向（多查询变体）
"""

from typing import List
from docqa.ingestion.base import Chunk
from .base import Retriever


class MultiQueryRetriever(Retriever):
    """
    多查询检索：对每个变体独立检索 → RRF 融合。

    用法:
        rewriter = QueryRewriter(llm.generate)
        base_retriever = HybridRetriever(...)
        mq = MultiQueryRetriever(base_retriever, rewriter)
        chunks = mq.search("旅行社全称", top_k=10)
    """

    def __init__(self, retriever: Retriever, rewriter, rrf_k: int = 60):
        self.retriever = retriever
        self.rewriter = rewriter
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> List[Chunk]:
        # 生成变体（含原始查询）
        queries = self.rewriter.rewrite(query, n_variants=3)

        # 每个变体独立检索，取多一些候选留冗余
        per_query_top = max(top_k, 15)

        # RRF 融合：score(chunk) = Σ_variants 1 / (rrf_k + rank_in_variant)
        # 复合键 (source_file, chunk_id) 避免多文档 id 碰撞，与 HybridRetriever 保持一致
        rrf_scores = {}
        chunk_map = {}
        for q in queries:
            chunks = self.retriever.search(q, top_k=per_query_top)
            for rank, c in enumerate(chunks, start=1):
                key = (getattr(c, 'source_file', '') or '', c.chunk_id)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                # 保留首次出现的 Chunk 对象（同一 chunk 跨变体是同一对象或等价副本，取其一即可）
                if key not in chunk_map:
                    chunk_map[key] = c

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (key, score) in enumerate(ranked[:top_k], start=1):
            c = chunk_map[key]
            c.metadata['mq_rrf_score'] = round(score, 6)
            c.metadata['mq_rank'] = rank
            results.append(c)

        return results
