"""
DocQA Pipeline 编排器
=====================
组装摄入、检索、生成的完整流水线。

换组件的方式（不需要改任何代码）：
  1. 改 config.yaml 中对应组件的 type
  2. 如果是新 type，在 _build_xxx() 里加一个 elif 分支即可
  3. 或者直接传实例化好的组件给构造函数，完全绕过 config

两条独立的管线:
  摄入: DocumentParser → Chunker → Embedder → VectorStore
  查询: Retriever → Reranker → PromptBuilder → LLM

用法:
  # 从配置构建
  pipeline = DocQAPipeline.from_config()

  # 摄入文档
  pipeline.ingest('doc.pdf')

  # 问答
  answer = pipeline.ask('合同编号是什么？')

  # 查看检索结果
  chunks = pipeline.retrieve('合同编号是什么？', top_k=5)
"""

from typing import List, Optional, Dict, Any

from docqa.config import load_config
from docqa.ingestion import DocumentParser, Chunker, Embedder, VectorStore, Chunk
from docqa.retrieval import Retriever, Reranker
from docqa.generation import PromptBuilder, LLM


class DocQAPipeline:
    """
    DocQA 主流水线编排器。

    可以两种方式构建：
    1. DocQAPipeline.from_config() — 从 config.yaml 自动组装
    2. DocQAPipeline(parser, chunker, ...) — 手动注入组件
    """

    def __init__(
        self,
        parser: DocumentParser = None,
        chunker: Chunker = None,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        retriever: Retriever = None,
        reranker: Reranker = None,
        prompt_builder: PromptBuilder = None,
        llm: LLM = None,
        # 新组件
        query_rewriter = None,          # QueryRewriter
        chunk_expander = None,          # ChunkExpander
        use_multi_query: bool = False,
        use_chunk_expansion: bool = False,
    ):
        # 摄入管线
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

        # 查询管线 — 检索
        self.retriever = retriever
        self.reranker = reranker

        # 查询管线 — 生成
        self.prompt_builder = prompt_builder
        self.llm = llm

        # 高级特性
        self.query_rewriter = query_rewriter
        self.chunk_expander = chunk_expander
        self.use_multi_query = use_multi_query
        self.use_chunk_expansion = use_chunk_expansion

        # 状态
        self._all_chunks: List[Chunk] = []

    # ====== 工厂方法 ======

    @classmethod
    def from_config(cls, config_path: str = None) -> 'DocQAPipeline':
        """从 config.yaml 构建完整流水线。"""
        cfg = load_config(config_path)
        rcfg = cfg.get('retrieval', {})

        # 查询改写
        use_mq = rcfg.get('query_rewrite', {}).get('enabled', False)

        # chunk 扩展
        use_expand = cfg.get('ingestion', {}).get('chunk_expansion', {}).get('enabled', False)

        return cls(
            parser=cls._build_parser(cfg),
            chunker=cls._build_chunker(cfg),
            embedder=cls._build_embedder(cfg),
            vector_store=cls._build_vector_store(cfg),
            retriever=None,
            reranker=cls._build_reranker(cfg),
            prompt_builder=cls._build_prompt_builder(cfg),
            llm=cls._build_llm(cfg),
            use_multi_query=use_mq,
            use_chunk_expansion=use_expand,
        )

    # ====== 摄入管线 ======

    def ingest(self, pdf_path, clear: bool = True) -> int:
        """
        摄入文档: parse → chunk → embed → store → [rebuild BM25]

        参数:
          pdf_path : str | List[str] | 目录路径
            单个 PDF、PDF 路径列表、或包含 .pdf 文件的目录
          clear : bool
            True: 清空旧数据后重新摄入
            False: 追加到已有索引

        返回: 总 chunk 数量
        """
        import os as _os
        import glob

        if self.parser is None:
            raise RuntimeError("Parser 未设置")
        if self.chunker is None:
            raise RuntimeError("Chunker 未设置")
        if self.embedder is None:
            raise RuntimeError("Embedder 未设置")
        if self.vector_store is None:
            raise RuntimeError("VectorStore 未设置")

        # 解析输入：统一为文件列表
        if isinstance(pdf_path, str):
            if _os.path.isdir(pdf_path):
                files = sorted(glob.glob(_os.path.join(pdf_path, '*.pdf')))
                if not files:
                    raise ValueError(f"目录中没有 PDF 文件: {pdf_path}")
            else:
                files = [pdf_path]
        elif isinstance(pdf_path, (list, tuple)):
            files = list(pdf_path)
        else:
            raise TypeError(f"pdf_path 类型不支持: {type(pdf_path)}")

        # 清空
        if clear:
            self.vector_store.clear()
            self._all_chunks = []
            self.retriever = None

        total_count = 0

        for fpath in files:
            if not _os.path.exists(fpath):
                print(f"  [WARN] 文件不存在，跳过: {fpath}")
                continue

            # Step 1: 解析
            pages = self.parser.parse(fpath)
            if not pages:
                print(f"  [WARN] 解析结果为空: {fpath}")
                continue

            # Step 2: 分块
            chunks = self.chunker.chunk(pages)

            # Step 3: 嵌入
            chunks = self.embedder.embed_chunks(chunks)

            # Step 4: 追加到向量库
            self.vector_store.add(chunks)
            self._all_chunks.extend(chunks)

            fname = _os.path.basename(fpath)
            print(f"  {fname}: {len(chunks)} chunks ({len(pages)} pages)")
            total_count += len(chunks)

        # Step 5: 重建 _all_chunks（从向量库获取，保证去重）
        self._all_chunks = self.vector_store.get_all_chunks()

        # Step 6: 重建 BM25 索引（基于全部 chunk）
        self._ensure_retriever()

        return total_count

    def _ensure_retriever(self) -> None:
        """确保 retriever（含 BM25 索引）基于全部 _all_chunks 构建"""
        # 如果 retriever 已存在且当前非 bm25-only 模式，只需更新 BM25
        needs_rebuild = (self.retriever is None)

        if needs_rebuild:
            cfg = load_config()
            mode = cfg.get('retrieval', {}).get('mode', 'hybrid')
            if mode == 'hybrid':
                from docqa.retrieval.hybrid import HybridRetriever
                rrf_k = cfg.get('retrieval', {}).get('hybrid', {}).get('rrf_k', 60)
                base = HybridRetriever(
                    vector_store=self.vector_store,
                    embedder=self.embedder,
                    rrf_k=rrf_k,
                )
                base.build_bm25_index(self._all_chunks)
                self.retriever = base
            elif mode == 'dense':
                from docqa.retrieval.dense import DenseRetriever
                self.retriever = DenseRetriever(self.vector_store, self.embedder)
            elif mode == 'bm25':
                from docqa.retrieval.bm25 import BM25Retriever
                self.retriever = BM25Retriever()
                self.retriever.build_index(self._all_chunks)
        else:
            # Hybrid 模式：更新 BM25 历史 corpse
            if hasattr(self.retriever, 'build_bm25_index'):
                self.retriever.build_bm25_index(self._all_chunks)

        # 可选：查询改写（包装在 retriever 外层）
        if self.use_multi_query and self.query_rewriter is None and self.llm is not None:
            from docqa.retrieval.query_rewriter import QueryRewriter
            from docqa.retrieval.multi_query import MultiQueryRetriever
            self.query_rewriter = QueryRewriter(self.llm.generate)
            self.retriever = MultiQueryRetriever(self.retriever, self.query_rewriter)

        # 可选：chunk 扩展
        if self.use_chunk_expansion and self.chunk_expander is None:
            from docqa.retrieval.chunk_expander import ChunkExpander
            self.chunk_expander = ChunkExpander(self._all_chunks)

    # ====== 查询管线 ======

    def retrieve(self, query: str, top_k: int = None, expand: bool = None) -> List[Chunk]:
        """
        检索（不含生成）: query → chunks

        流程（按配置）：
          query → [改写] → retriever.search → [重排序] → [chunk扩展] → chunks

        多查询模式下，重排序使用每个候选 chunk 在任意变体下的最高分，
        避免 reranker 因原始查询语义不匹配而误杀正确结果。
        """
        if self.retriever is None:
            raise RuntimeError("Retriever 未设置。请先 ingest() 或手动注入 retriever。")

        if top_k is None:
            top_k = load_config().get('retrieval', {}).get('top_k', 10)

        cfg = load_config().get('retrieval', {})
        rerank_enabled = cfg.get('reranker', {}).get('enabled', False)

        # Step 1: 检索（可能包含多查询改写）
        if self.use_multi_query and self.retriever is not None:
            candidate_pool = cfg.get('reranker', {}).get('candidate_pool', 50)
            max_k = max(candidate_pool, top_k) if rerank_enabled else top_k
            candidates = self.retriever.search(query, max_k)
        elif rerank_enabled and self.reranker is not None:
            candidate_pool = cfg.get('reranker', {}).get('candidate_pool', 50)
            candidates = self.retriever.search(query, candidate_pool)
        else:
            candidates = self.retriever.search(query, top_k)

        # Step 2: 重排序
        if rerank_enabled and self.reranker is not None:
            if self.use_multi_query and self.query_rewriter is not None:
                # 多查询模式：对每个变体分别做 rerank，取每个 chunk 的最高分
                queries = self.query_rewriter.rewrite(query, n_variants=3)
                best_score = {}
                for q in queries:
                    ranked = self.reranker.rerank(q, candidates, top_k=len(candidates))
                    for c in ranked:
                        s = c.metadata.get('rerank_score', 0)
                        if c.chunk_id not in best_score or s > best_score[c.chunk_id][0]:
                            best_score[c.chunk_id] = (s, c)
                chunks = sorted(
                    [c for _, c in best_score.values()],
                    key=lambda x: x.metadata.get('rerank_score', 0),
                    reverse=True
                )[:top_k]
            else:
                chunks = self.reranker.rerank(query, candidates, top_k)
        else:
            chunks = candidates[:top_k]

        # Step 3: chunk 上下文扩展（small-to-big）
        if (expand is True) or (expand is None and self.use_chunk_expansion):
            if self.chunk_expander is not None:
                chunks = self.chunk_expander.expand(chunks, window=1)

        return chunks

    def ask(self, query: str) -> str:
        """
        端到端问答: query → answer
        检索结果自动应用 chunk 扩展（如果配置了）
        """
        if self.prompt_builder is None:
            raise RuntimeError("PromptBuilder 未设置")
        if self.llm is None:
            raise RuntimeError("LLM 未设置")

        chunks = self.retrieve(query, expand=True)  # 生成时自动扩展上下文
        prompt = self.prompt_builder.build(query, chunks)
        answer = self.llm.generate(prompt)
        return answer

    def ask_with_chunks(self, query: str):
        """端到端问答 + 检索结果"""
        chunks = self.retrieve(query)
        answer = self.ask(query)
        return answer, chunks

    # ====== 组件构建器（私有） ======

    @staticmethod
    def _build_parser(cfg: Dict) -> DocumentParser:
        icfg = cfg.get('ingestion', {})
        ptype = icfg.get('parser', {}).get('type', 'pymupdf')
        if ptype == 'pymupdf':
            from docqa.ingestion.parser import PyMuPDFParser
            return PyMuPDFParser()
        raise ValueError(f"不支持的 parser type: {ptype}")

    @staticmethod
    def _build_chunker(cfg: Dict) -> Chunker:
        icfg = cfg.get('ingestion', {})
        ccfg = icfg.get('chunker', {})
        ctype = ccfg.get('type', 'sentence')
        if ctype == 'sentence':
            from docqa.ingestion.chunker import SentenceChunker
            return SentenceChunker(
                chunk_size=ccfg.get('chunk_size', 512),
                overlap=ccfg.get('overlap', 128),
            )
        raise ValueError(f"不支持的 chunker type: {ctype}")

    @staticmethod
    def _build_embedder(cfg: Dict) -> Embedder:
        icfg = cfg.get('ingestion', {})
        ecfg = icfg.get('embedder', {})
        etype = ecfg.get('type', 'bge-small-zh')
        if etype == 'bge-small-zh':
            from docqa.ingestion.embedder import BGEEmbedder
            return BGEEmbedder(
                model_path=ecfg.get('model_path', 'models/bge-small-zh-v1.5'),
                device=ecfg.get('device', 'cpu'),
            )
        raise ValueError(f"不支持的 embedder type: {etype}")

    @staticmethod
    def _build_vector_store(cfg: Dict) -> VectorStore:
        icfg = cfg.get('ingestion', {})
        vcfg = icfg.get('vector_store', {})
        vtype = vcfg.get('type', 'chromadb')
        if vtype == 'chromadb':
            from docqa.ingestion.store import ChromaStore
            return ChromaStore(
                persist_dir=vcfg.get('persist_dir', 'chroma_db'),
                collection_name=vcfg.get('collection_name', 'docqa'),
            )
        raise ValueError(f"不支持的 vector_store type: {vtype}")

    @staticmethod
    def _build_reranker(cfg: Dict) -> Reranker:
        rcfg = cfg.get('retrieval', {})
        rrcfg = rcfg.get('reranker', {})
        if not rrcfg.get('enabled', False):
            return None
        rtype = rrcfg.get('type', 'bge-reranker')
        if rtype == 'bge-reranker':
            from docqa.retrieval.reranker import BGEReranker
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            return BGEReranker(model_name=rrcfg.get('model', 'BAAI/bge-reranker-v2-m3'), device=device)
        raise ValueError(f"不支持的 reranker type: {rtype}")

    @staticmethod
    def _build_prompt_builder(cfg: Dict) -> PromptBuilder:
        from docqa.generation.prompt import RAGPromptBuilder
        gcfg = cfg.get('generation', {})
        pcfg = gcfg.get('prompt', {})
        return RAGPromptBuilder(
            system_prompt=pcfg.get('system_prompt', None)
        )

    @staticmethod
    def _build_llm(cfg: Dict) -> LLM:
        """工厂方法：根据 config 创建 LLM 实例。

        支持的后端:
          ollama              → Ollama 原生 API (http://localhost:11434/api/chat)
          openai              → OpenAI 兼容 API (任何 /v1/chat/completions 端点)
                                包括: OpenAI 官方、Ollama /v1、DeepSeek、vLLM 等

        切换模型只需改 config.yaml 的 generation.llm 段，不需要改代码。
        """
        gcfg = cfg.get('generation', {})
        lcfg = gcfg.get('llm', {})
        ltype = lcfg.get('type', 'ollama')
        model = lcfg.get('model', 'qwen2.5:7b')
        temperature = lcfg.get('temperature', 0.1)
        max_tokens = lcfg.get('max_tokens', 1024)
        timeout = lcfg.get('timeout', 300)

        if ltype == 'ollama':
            from docqa.generation.llm import OllamaLLM
            return OllamaLLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        elif ltype in ('openai', 'openai_compatible'):
            from docqa.generation.llm_openai import OpenAICompatibleLLM
            return OpenAICompatibleLLM(
                model=model,
                base_url=lcfg.get('base_url', 'https://api.openai.com/v1'),
                api_key=lcfg.get('api_key', 'sk-placeholder'),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        raise ValueError(
            f"不支持的 llm type: {ltype}。"
            f"支持: ollama | openai | openai_compatible"
        )
