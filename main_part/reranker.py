"""
Cross-encoder 重排序模块
=========================
用 BAAI/bge-reranker-v2-m3 对候选 chunk 做精细语义匹配重排序。

原理：
- 双塔模型（BGE-small）对 query 和 chunk 分别编码后算相似度——快但粗糙
- Cross-encoder 把 query 和 chunk 拼接后一起编码——慢但精准
- 先混合检索取 top-50，再用 cross-encoder 精排取 top-k：兼顾速度和质量

性能注意：
- 首次加载会从 HuggingFace 下载模型（~2.3GB），需联网
- CPU 推理每对约 50-100ms，50 对约 3-5 秒
- GPU (CUDA) 推理会快 10-50 倍
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder


# Cross-encoder 模型名
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'


class Reranker:
    """
    Cross-encoder 重排序器。

    用法:
        reranker = Reranker()
        results = reranker.rerank(query, candidate_chunks, top_k=5)
    """

    def __init__(self, model_name: str = None, device: str = None):
        """
        参数
        ----
        model_name : str
            模型名/路径，默认 BAAI/bge-reranker-v2-m3
        device : str
            推理设备，默认自动检测（'cuda' 或 'cpu'）
        """
        if model_name is None:
            model_name = RERANKER_MODEL

        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

        print(f"加载 Cross-encoder: {model_name} (device={device})...")
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,  # bge-reranker-v2-m3 推荐 512
        )
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        对候选 chunk 列表精排，返回 top_k。

        参数
        ----
        query : str
            用户问题
        candidates : List[Dict]
            候选 chunk 列表，每个需包含 'text' 字段
        top_k : int
            返回 top_k 个

        返回
        ----
        List[Dict] : 精排后的 chunk 列表，新增 'rerank_score' 字段
        """
        if not candidates:
            return candidates

        # 构造 (query, chunk_text) pair 列表
        pairs = [(query, c['text']) for c in candidates]

        # Batch 推理：bge-reranker 对每个 pair 输出一个相关性分数
        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        # 附加分数
        for i, c in enumerate(candidates):
            c['rerank_score'] = round(float(scores[i]), 4)
            # 保留原始检索分数用于调试
            if 'score' not in c:
                c['score'] = c['rerank_score']

        # 按 rerank_score 降序排序，取 top_k
        ranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        return ranked[:top_k]

    def rerank_fast(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        rerank 的别名——语义完全相同，提供一个更直观的方法名。
        """
        return self.rerank(query, candidates, top_k)


# ====== 完整流水线便捷函数 ======

def hybrid_search_then_rerank(
    query: str,
    hybrid_retriever,
    vector_retriever,
    embedder,
    reranker,
    top_k: int = 5,
    candidate_top_k: int = 50,
) -> List[Dict]:
    """
    一步完成：混合检索 → 重排序 → 输出 top_k。

    这是优化后 RAG 流水线的核心检索函数。
    """
    # Step 1: 混合检索（BM25 + 向量，RRF 融合）
    fused = hybrid_retriever.search_hybrid(
        query=query,
        vector_retriever=vector_retriever,
        embedder=embedder,
        top_k=candidate_top_k,
    )

    # Step 2: Cross-encoder 精排
    ranked = reranker.rerank(query, fused, top_k=top_k)

    return ranked
