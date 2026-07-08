"""
混合检索模块
============
BM25 关键词检索 + 向量语义检索，通过 RRF (Reciprocal Rank Fusion) 融合。

原理：
- BM25 对精确匹配敏感（数字、编号、专有名词），向量检索对语义模糊查询敏感
- RRF 将两者的排名转换为分数后融合，免调参
- RRF 公式: score(d) = Σ 1/(k + rank_i(d)),  k=60 (经典取值)
"""

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict


class HybridRetriever:
    """
    混合检索器：BM25 + 向量，RRF 融合。

    用法:
        hr = HybridRetriever()
        hr.build_bm25_index(chunks)           # 对 chunk 文本建 BM25 索引
        results = hr.search_hybrid(query, vector_retriever, embedder, top_k=10)
    """

    def __init__(self, rrf_k: int = 60):
        """
        参数
        ----
        rrf_k : int
            RRF 融合参数。k=60 是经典取值，k 越大排名靠后的结果贡献越小。
        """
        self.rrf_k = rrf_k
        self.bm25 = None
        self.chunk_texts = []
        self.chunk_ids = []
        self._tokenized_corpus = []

    # ------- 中文分词 -------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """jieba 精确模式分词"""
        return [t for t in jieba.cut(text) if t.strip()]

    # ------- BM25 索引 -------

    def build_bm25_index(self, chunks: List[Dict]) -> None:
        """
        对 chunk 列表构建 BM25 索引。

        参数
        ----
        chunks : List[Dict]
            每个元素包含 'chunk_id', 'text'
        """
        self.chunk_texts = [c['text'] for c in chunks]
        self.chunk_ids = [c['chunk_id'] for c in chunks]
        self._tokenized_corpus = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"BM25 索引完成，共 {len(self.chunk_texts)} 条")

    # ------- BM25 检索 -------

    def search_bm25(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        纯 BM25 检索，返回带分数的 chunk 列表。

        返回
        ----
        List[Dict] : [{'chunk_id': int, 'score': float}, ...]
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先调用 build_bm25_index()")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 归一化到 [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        # 排序取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤得分为 0 的结果
                results.append({
                    'chunk_id': self.chunk_ids[idx],
                    'score': float(scores[idx]),
                })
        return results

    # ------- RRF 融合 -------

    @staticmethod
    def _rrf_fusion(
        bm25_results: List[Dict],
        vector_results: List[Dict],
        k: int = 60,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        RRF 融合 BM25 和向量检索结果。

        对每个 chunk：
          rrf_score = 1/(k + bm25_rank) + 1/(k + vector_rank)
        如果某 chunk 只在一个结果集中出现，另一个 rank 记为 +∞（即该项贡献为 0）。

        参数
        ----
        bm25_results : 按分数降序排列的 BM25 结果
        vector_results : 按分数降序排列的向量检索结果
        k : RRF 参数
        top_k : 输出 top_k

        返回
        ----
        List[Dict] : 融合后按 rrf_score 降序的 top_k 个 chunk
        """
        # 构建 rank 映射
        rrf_scores = {}

        for rank, r in enumerate(bm25_results, start=1):
            cid = r['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

        for rank, r in enumerate(vector_results, start=1):
            cid = r['chunk_id']
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank)

        # 排序
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top = sorted_ids[:top_k]

        results = []
        for cid, rrf_score in top:
            results.append({
                'chunk_id': cid,
                'rrf_score': round(rrf_score, 6),
            })
        return results

    # ------- 完整混合检索 -------

    def search_hybrid(
        self,
        query: str,
        vector_retriever,
        embedder,
        top_k: int = 10,
        bm25_candidates: int = 50,
    ) -> List[Dict]:
        """
        混合检索：BM25 + 向量 + RRF 融合。

        参数
        ----
        query : str
            用户问题
        vector_retriever : Retriever
            已有的 Chroma 向量检索器（需要 search 方法返回带 chunk_id 的结果）
        embedder : Embedder
            用于向量化 query
        top_k : int
            最终返回 top_k 个 chunk
        bm25_candidates : int
            BM25 候选数量（大一些，留给 RRF 更多选择）

        返回
        ----
        List[Dict] : 融合后的 chunk 列表，格式与原有 retriever.search() 兼容
            [{chunk_id, text, source_page, score}, ...]
        """
        # 1. BM25 检索
        bm25_results = self.search_bm25(query, top_k=bm25_candidates)

        # 2. 向量检索
        vector_raw = vector_retriever.search(query, embedder, top_k=bm25_candidates)
        vector_results = [
            {'chunk_id': c['chunk_id'], 'score': c['score']}
            for c in vector_raw
        ]

        # 3. RRF 融合
        fused = self._rrf_fusion(
            bm25_results, vector_results,
            k=self.rrf_k, top_k=top_k,
        )

        # 4. 回填 text 和 source_page（从 Chroma 取）
        chunk_data = _get_chunk_data_map(vector_retriever)

        results = []
        for item in fused:
            cid = item['chunk_id']
            info = chunk_data.get(cid, {})
            results.append({
                'chunk_id': cid,
                'text': info.get('text', ''),
                'source_page': info.get('source_page', -1),
                'score': item['rrf_score'],
            })

        return results


def _get_chunk_data_map(retriever) -> Dict[int, Dict]:
    """从 Chroma 取回所有 chunk 的 text 和 metadata，构建 id->data 映射。"""
    try:
        all_data = retriever.collection.get()
        mapping = {}
        for i, cid in enumerate(all_data['ids']):
            mapping[int(cid)] = {
                'text': all_data['documents'][i],
                'source_page': all_data['metadatas'][i].get('source_page', -1),
            }
        return mapping
    except Exception:
        return {}
