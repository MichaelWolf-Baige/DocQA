"""
HyDE 检索器（Hypothetical Document Embeddings）
================================================
论文：Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

核心思想：
  短问题的 query 向量信息密度低，与真实文档的语义距离偏远，召回质量受限。
  HyDE 先用 LLM 针对查询生成一段"假设答案文档"，再对该假设文档做 embedding，
  用这个向量去检索真实语料库。因为假设答案与真实文档处于相似的语义空间，
  相当于在"问题空间 ↔ 文档空间"之间架了一座桥，提升命中相关真实文档的概率。

流程：
  query ──LLM──▶ 假设文档 ──embed──▶ 假设向量 ──向量检索──▶ top_k 真实 chunks

适用场景：
  - 抽象问题、短问题、语义难表达问题
  - 零样本场景（无标注数据训练检索器）
不适用：
  - 含精确编号/产品名/错误码的查询（关键词或混合检索更合适）
  - LLM 不可用或延迟敏感的场景（多一次 LLM 调用）

与 HybridRetriever 的关系：
  HyDE 是"查询侧"增强（改写 query 的 embedding 来源）；
  Hybrid 是"检索侧"增强（多路召回融合）。两者正交，可叠加使用。
"""
from typing import List, Optional, Callable
from docqa.ingestion.base import Chunk, Embedder, VectorStore
from .base import Retriever
from .dense import DenseRetriever


HYDE_PROMPT = (
    "请根据下面的问题，写一段可能的答案文档（120-200 字）。\n"
    "要求：\n"
    "1. 只写答案内容，不要加'答案是'之类的引导语\n"
    "2. 用陈述句描述，就像真实文档里的段落\n"
    "3. 即使不确定细节也要给出合理推测——这段文字只用于检索，不会直接作为答案\n\n"
    "问题：%s"
)


class HyDERetriever(Retriever):
    """HyDE：用 LLM 生成的假设文档做向量检索"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        generate_fn: Callable[[str], str],
        prompt_template: Optional[str] = None,
        fallback_to_query: bool = True,
    ):
        """
        参数
        ----
        vector_store : 向量库（含真实文档的 embedding）
        embedder : 嵌入模型
        generate_fn : LLM 生成函数，签名 (prompt: str) -> str
        prompt_template : 假设文档生成 prompt，含 %s 占位符
        fallback_to_query : LLM 生成失败时是否回退为直接用原 query 检索
        """
        self.dense = DenseRetriever(vector_store, embedder)
        self.embedder = embedder
        self.generate = generate_fn
        self._template = prompt_template or HYDE_PROMPT
        self.fallback_to_query = fallback_to_query

    def search(self, query: str, top_k: int) -> List[Chunk]:
        # Step 1: 用 LLM 生成假设答案文档
        hyp_doc = self._generate_hypothetical_doc(query)

        # Step 2: 决定用哪个向量做检索
        if hyp_doc:
            retrieval_text = hyp_doc
            retrieval_source = 'hyde'
        elif self.fallback_to_query:
            retrieval_text = query
            retrieval_source = 'query_fallback'
        else:
            return []

        # Step 3: 对检索文本做 embedding，送入向量库检索
        query_vec = self.embedder.embed_query(retrieval_text)
        results = self.vector_store_search(query_vec, top_k)

        for i, c in enumerate(results):
            c.metadata['hyde_source'] = retrieval_source
            c.metadata['hyde_rank'] = i + 1
        return results

    def vector_store_search(self, query_vec: List[float], top_k: int) -> List[Chunk]:
        return self.dense.store.search(query_vec, top_k)

    def _generate_hypothetical_doc(self, query: str) -> Optional[str]:
        prompt = self._template % query
        raw = self.generate(prompt)

        if not raw or raw.startswith('[ERROR'):
            return None

        text = raw.strip()
        if len(text) < 5:
            return None
        return text