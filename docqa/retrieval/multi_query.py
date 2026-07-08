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

        # 每个变体独立检索
        per_query_top = max(top_k, 15)  # 每个变体取多一些，留冗余

        # 收集所有结果 + 重排序用最高分
        seen = {}  # chunk_id → (best_rank, best_chunk)
        for q in queries:
            chunks = self.retriever.search(q, top_k=per_query_top)
            for rank, c in enumerate(chunks, start=1):
                if c.chunk_id not in seen or rank < seen[c.chunk_id][0]:
                    seen[c.chunk_id] = (rank, c)

        # 按最佳排名排序
        sorted_chunks = sorted(seen.values(), key=lambda x: x[0])

        results = []
        for rank, (_, c) in enumerate(sorted_chunks[:top_k]):
            c.metadata['mq_best_rank'] = rank + 1
            results.append(c)

        return results
