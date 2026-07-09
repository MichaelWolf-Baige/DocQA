# DocQA 项目当前状态

> 更新日期：2026-07-08

---

## 项目概况

DocQA 是一个中文 PDF 文档智能问答系统，经历了从 Naive RAG MVP → 评估驱动优化 → 架构解耦重构 的完整演进。

---

## 当前进度

### ✅ Phase 0: 评估基线建立（已完成）

- 30 题手工测试集，按题型分类（fact_lookup / exact_match / summary / boundary）
- 手工标注 relevant_chunk_ids 作为检索 ground truth
- 检索指标系统：Recall@k / Precision@k / MRR / NDCG / Hit Rate
- Oracle 四条件瓶颈分析（Actual / Oracle / Bare / Noise）
- 人工审查确认瓶颈在检索层而非生成层

### ✅ Phase 1: 检索优化（已完成）

- BM25 关键词检索 + 稠密向量 → RRF 混合检索
- Cross-encoder 重排序（bge-reranker-v2-m3）
- 查询改写（Query Rewriting）：LLM 生成 3 变体 → 多路检索融合
- 小→大分块（chunk_size=256 检索 + chunk 上下文扩展生成）

### ✅ Phase 2: 架构解耦重构（已完成）

- 从 `main_part/` 单体架构重构为 `docqa/` 模块化架构
- 抽象接口 + 可插拔实现（Parser / Chunker / Embedder / VectorStore / Retriever / Reranker / PromptBuilder / LLM）
- 集中配置管理（config.yaml）
- 摄入管线 和 查询管线 完全分离

### ✅ Phase 4: 效果最终验证（已完成，2026-07-09）

- 在 GPU (RTX 4060) 上完成新架构 5 种检索方案的完整对比
- Hybrid + Reranker 为最优单方案（Recall@5 83%, MRR 0.76）
- 确认 LLM 查询改写在有 Reranker 时反而降低 Recall@5（83→74%）

### ✅ Phase 5: LLM 后端解耦 + CPU 适配（已完成，2026-07-09）

- 新增 OpenAI 兼容后端（支持 Ollama /v1, DeepSeek, vLLM 等所有兼容 API）
- RAG 管线与 LLM 完全分离，切换模型只需改 config.yaml 一行
- CPU 优化预设 (`config_cpu.yaml`)：关查询改写，推荐 1.5b 模型
- 验证 1.5b 模型 CPU 方案端到端可用

### ◻️ Phase 3: 多文档评估（待做）

- 用第二份不同类型文档跑评估管线，拿到跨文档对比数据
- 判断当前优化的泛化能力

---

## 目录结构

```
Rag/
├── docqa/                     ← 新架构（主力）
│   ├── config.yaml            # 集中配置
│   ├── config.py              # YAML 加载器
│   ├── pipeline.py            # Pipeline 编排器
│   ├── ingestion/             # 摄入管线
│   │   ├── base.py            #   接口定义 (Page, Chunk, Parser, Chunker, Embedder, VectorStore)
│   │   ├── parser.py          #   PyMuPDF 解析器
│   │   ├── chunker.py         #   句子边界分块器
│   │   ├── embedder.py        #   BGE-small-zh 嵌入器
│   │   └── store.py           #   ChromaDB 向量存储
│   ├── retrieval/             # 检索管线
│   │   ├── base.py            #   接口定义 (Retriever, Reranker)
│   │   ├── dense.py           #   稠密向量检索
│   │   ├── bm25.py            #   BM25 关键词检索
│   │   ├── hybrid.py          #   混合检索 (BM25+向量, RRF融合)
│   │   ├── query_rewriter.py  #   查询改写 (LLM生成3-5变体)
│   │   ├── multi_query.py     #   多查询检索 (multi-query → RRF)
│   │   ├── chunk_expander.py  #   Chunk上下文扩展 (small-to-big)
│   │   └── reranker.py        #   Cross-encoder重排序
│   ├── generation/            # 生成管线
│   │   ├── base.py            #   接口定义 (PromptBuilder, LLM)
│   │   ├── prompt.py          #   RAG Prompt 构建器
│   │   └── llm.py             #   Ollama LLM 调用
│   └── evaluation/            # 评估模块
│       ├── base.py            #   接口定义
│       └── metrics.py         #   检索指标 (Recall/MRR/NDCG/HitRate)
│
├── main_part/                 ← 旧架构（保留兼容，不再维护）
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   └── generator.py
│
├── eval/                      ← 评估脚本和测试集
│   ├── config.py              #   评估配置
│   ├── metrics.py             #   评估指标
│   ├── testset.py             #   测试集管理
│   ├── oracle_test.py         #   Oracle 瓶颈分析
│   ├── auto_judge.py          #   LLM Judge
│   ├── compare_judge.py       #   LLM Judge vs 人工对比
│   ├── run_baseline.py        #   基线评估
│   ├── run_full_baseline.py   #   完整基线（检索+Oracle）
│   ├── run_optimization_eval.py # 优化前后对比
│   └── data/
│       ├── test_questions.jsonl    # 512chunk 测试集 (30题)
│       └── test_questions_256.jsonl # 256chunk 测试集 (30题)
│
├── app.py                     # Streamlit 前端（使用 main_part 旧架构）
├── models/                    # Embedding 模型（需自行下载）
├── chroma_db/                 # 向量数据库（运行时生成）
├── requirements.txt
└── README.md
```

---

## 当前架构详细说明

### 数据流

```
摄入管线:
  PDF → PyMuPDFParser → SentenceChunker(256,64) → BGEEmbedder → ChromaStore
                                                      ↓
                                                BM25 索引（并行）
查询管线:
  Query → [QueryRewriter → MultiQueryRetriever]
            ↓ 生成3-5个变体，多路检索+去重合并
            → HybridRetriever (BM25+向量, RRF融合)
            → [BGEReranker 重排序]
            → [ChunkExpander 上下文扩展(small-to-big)]
            → RAGPromptBuilder
            → OllamaLLM (qwen2.5:7b)
```

### 可切换的检索模式

修改 `docqa/config.yaml` 一行即可切换：

| mode | 说明 |
|------|------|
| `dense` | 纯稠密向量检索 |
| `bm25` | 纯 BM25 关键词检索 |
| `hybrid` | BM25 + 向量 RRF 融合（推荐） |

### 可选特性开关

```yaml
retrieval:
  query_rewrite:
    enabled: true      # 查询改写（多查询检索）
  reranker:
    enabled: true      # Cross-encoder 重排序

ingestion:
  chunk_expansion:
    enabled: true      # Chunk 上下文扩展 (small-to-big)
```

---

## 评估基线数据

### 检索指标演进（同一测试集 28 题，256chunk，新架构 SentenceChunker）

| 方案 | Recall@5 | Recall@10 | MRR | NDCG@10 |
|------|:------:|:------:|:------:|:------:|
| ①纯向量 (Baseline) | 64.29% | 76.79% | 0.587 | 0.597 |
| ②纯BM25 | 55.95% | 72.02% | 0.548 | 0.562 |
| ③混合检索 (RRF) | 62.50% | 68.45% | 0.519 | 0.537 |
| ④混合+重排序 | **83.33%** | **89.88%** | **0.758** | **0.766** |
| ⑤完整方案 (+查询改写) | 73.81% | **94.05%** | 0.703 | 0.740 |

> 注：方案⑤的查询改写对 Recall@5 有害(83→74%)，CPU 环境建议关闭。

### 效果标准

| 指标 | 目标(良好) | 当前④(新架构/Reranker) | 状态 |
|------|:---:|:---:|:---:|
| Recall@5 | ≥85% | 83.33% | ⚠️ 差 1.67% |
| Recall@10 | ≥90% | 89.88% | ⚠️ 差 0.12% |
| MRR | ≥0.75 | 0.758 | ✅ 达标 |

### Precision 说明

Precision≥80% 在本题集上数学上不可达（每题平均1.5个相关chunk，10个结果中最多1.5个相关=15%天花板）。已将 Precision 降级为参考指标，用 MRR/NDCG 作为主排序质量指标。

### 关键发现

- **检索是主瓶颈**：Oracle 准确率约89%，即7B模型拿到正确chunk基本能答对
- **混合检索单独不显著**：RRF 在47 chunks上区分度有限，需要配合重排序
- **重排序最有效**：chunk 30 从第11位拉到第1位，Recall@5 从63%飙升至89%

---

## 环境要求

### 必装

```bash
pip install -r requirements.txt
pip install ollama  # 如果要用 ollama Python 库
```

### Ollama 模型

```bash
ollama pull qwen2.5:7b           # 生成模型（当前默认）
ollama pull bge-reranker-v2-m3   # 不需要，这是 sentence-transformers 模型
```

### Embedding 模型

```bash
python download_model.py   # 下载 BGE-small-zh-v1.5 (~95MB) 到 models/
```

### 首次使用

```python
from docqa.pipeline import DocQAPipeline

pipeline = DocQAPipeline.from_config()  # 从 config.yaml 构建
pipeline.ingest('your_document.pdf')    # 摄入文档
answer = pipeline.ask('你的问题')         # 问答
chunks = pipeline.retrieve('你的问题')    # 只看检索结果
```

### 已知问题

1. **代理冲突**：如果系统配置了 HTTP_PROXY，localhost:11434 请求会被拦截。
   解决：generation/llm.py 已内置 `_clear_proxy()` 方法，自动清除代理环境变量。
   如果仍有问题，在终端运行 `set NO_PROXY=localhost,127.0.0.1`

2. **ChromaDB 首次运行**：会创建 `chroma_db/` 目录。多次 ingest 同一文档会覆盖旧数据。

3. **GPU vs CPU**：Reranker 在 CPU 上每对约 50-100ms，28题×50候选=1400对≈2-4分钟。
   GPU 上快 10-50 倍。建议在 GPU 机器上跑完整评估。

4. **256chunk 测试集**：需要重新映射 gold chunk id。当前已完成映射（test_questions_256.jsonl），
   但评估脚本尚需适配新 chunk_size。

---

## 待解决问题

1. **q009（旅行社全称）**：已通过查询改写修复。改写变体"深圳市报春国际旅游集团有限公司全名是什么？"命中 chunk 4 排名第1。
2. **q014（旅游景点，需3个chunk）**：小块+查询改写已覆盖2/3（chunk 69, 70），chunk 67 仍需解决。
3. **跨文档评估**：当前全部评估基于一份旅游合同PDF。需要第二份文档验证泛化能力。
4. **Precision 目标重新评估**：当前测试集数学天花板约15%，Precision≥80% 不适配。需要更多相关chunk的测试集或替换指标。
