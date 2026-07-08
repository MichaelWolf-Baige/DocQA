"""
DocQA — 模块化解耦 RAG 系统

架构：
  docqa/
  ├── config.yaml         集中配置，换组件改这里
  ├── config.py           YAML 加载器
  ├── ingestion/          摄入管线: Parser→Chunker→Embedder→VectorStore
  ├── retrieval/          查询管线(检索): Retriever→Reranker
  ├── generation/         查询管线(生成): PromptBuilder→LLM
  ├── evaluation/         评估: Metrics / Testset / Oracle
  └── pipeline.py         编排器: 组装管线，暴露统一接口

设计原则：
  - 每个组件是一个接口（抽象基类），实现可任意替换
  - 摄入和查询是两条独立管线
  - 配置集中，代码零硬编码
  - 评估是第一公民，不依赖具体组件实现
"""

from .pipeline import DocQAPipeline
