"""
eval 适配层
============
桥接 docqa/ 新架构与 eval/ 脚本。

旧 main_part/ 提供的接口契约：
  retriever.search(question, embedder, top_k) -> List[dict]  # 三参，返回 dict
  build_rag_prompt(question, chunks)                         # chunks 是 dict 列表

新 docqa/ 架构的接口契约：
  retriever.search(query, top_k) -> List[Chunk]             # 两参，返回 Chunk 对象
  pipeline.retrieve(query, top_k) -> List[Chunk]
  pipeline.prompt_builder.build(query, chunks)               # chunks 是 Chunk 列表

本模块提供三个资产让 eval 脚本一行切入新架构：
  1. build_eval_pipeline()   构建 ingest 好的 DocQAPipeline + 兼容旧 dict 接口的检索 / prompt
  2. dict_search_fn(pipeline) 把二参 Chunk 接口包装成三参 dict 接口
  3. 按块 id 取回文本         用于 Oracle / noise 条件
"""
from __future__ import annotations
import os
from typing import List, Dict, Optional, Callable

from docqa.pipeline import DocQAPipeline
from docqa.config import load_config
from eval.config import PDF_PATH, MODEL_DIR, CHROMA_DIR


def build_eval_pipeline(
    pdf_path: Optional[str] = None,
    rebuild_index: bool = False,
    config_path: Optional[str] = None,
) -> DocQAPipeline:
    """
    构建一个 ingest 好的 DocQAPipeline 用于评估。

    参数
    ----
    pdf_path : 样本 PDF 路径，默认 eval.config.PDF_PATH
    rebuild_index : True 则清空已有向量库重新摄入
    config_path : 覆盖默认 docqa/config.yaml
    """
    pdf_path = pdf_path or PDF_PATH
    pipeline = DocQAPipeline.from_config(config_path)

    # 已有数据时不重复 ingest
    need_ingest = rebuild_index or (pipeline.vector_store.count() == 0)
    if need_ingest:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
        pipeline.ingest(pdf_path, clear=True)
    return pipeline


def dict_search_fn(pipeline: DocQAPipeline) -> Callable[[str, int], List[Dict]]:
    """
    把 pipeline.retrieve 包装成旧三参 dict 接口：
        search(question, top_k, embedder=None) -> List[dict]

    eval 旧脚本传的是 retriever.search(question, embedder, top_k)，
    这里签名放宽，兼容三参和两参调用。
    """
    def search(question: str, top_k_or_embedder, top_k: Optional[int] = None) -> List[Dict]:
        # 兼容两种调用约定：
        #   search(question, top_k)          — 新两参
        #   search(question, embedder, top_k) — 旧三参
        if top_k is None:
            k = top_k_or_embedder  # 第二参数是整数
        else:
            k = top_k
        chunks = pipeline.retrieve(question, top_k=k)
        return [_chunk_to_dict(c) for c in chunks]
    return search


def _chunk_to_dict(c) -> Dict:
    """Chunk 对象 → 旧式 dict。"""
    return {
        'chunk_id': c.chunk_id,
        'text': c.text,
        'source_page': c.source_page,
        'source_file': getattr(c, 'source_file', '') or '',
        'score': c.metadata.get('rrf_score', c.metadata.get('score', 0.0)),
    }


def build_rag_prompt(question: str, chunks: List[Dict]) -> str:
    """
    兼容旧 main_part.prompt_builder.build_rag_prompt 的签名。
    接受 dict 列表（Oracle/noise 条件构造的不规则 chunks）或 Chunk 列表。
    """
    from docqa.ingestion.base import Chunk
    from docqa.generation.prompt import RAGPromptBuilder

    # 将 dict 归一为 Chunk 对象
    normalized = []
    for c in chunks:
        if isinstance(c, Chunk):
            normalized.append(c)
        elif isinstance(c, dict):
            normalized.append(Chunk(
                chunk_id=c.get('chunk_id', -1),
                text=c.get('text', ''),
                source_page=c.get('source_page', -1),
                source_file=c.get('source_file', '') or '',
            ))
        else:
            raise TypeError(f"不支持的 chunk 类型: {type(c)}")

    builder = RAGPromptBuilder()
    return builder.build(question, normalized)


def get_chunk_by_id(pipeline: DocQAPipeline, chunk_id: int) -> Optional[Dict]:
    """按 chunk_id 从向量库取回文本（Oracle / noise 条件用）。"""
    all_chunks = pipeline.vector_store.get_all_chunks()
    for c in all_chunks:
        if c.chunk_id == chunk_id:
            return _chunk_to_dict(c)
    return None


def get_all_chunks(pipeline: DocQAPipeline) -> List[Dict]:
    """返回向量库全部 chunks 的 dict 表示。"""
    return [_chunk_to_dict(c) for c in pipeline.vector_store.get_all_chunks()]


def make_generator(pipeline: DocQAPipeline) -> Callable[[str], str]:
    """用 pipeline 的 LLM 构建 generate(prompt) -> str 函数。"""
    if pipeline.llm is None:
        raise RuntimeError("pipeline 未配置 LLM，无法生成")
    return pipeline.llm.generate


def make_embedder(pipeline):
    """返回 pipeline 的 embedder（兼容旧 eval 脚本传 embedder 给 retriever.search 的调用）。"""
    return pipeline.embedder