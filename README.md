# DocQA — 文档智能问答系统

中文 PDF 文档的 RAG（检索增强生成）系统，经历了从 Naive RAG → 评估驱动优化 → 架构解耦的完整演进。

## 快速开始（新电脑 5 分钟搭好）

### 1. 克隆 + 安装依赖

```bash
git clone https://github.com/MichaelWolf-Baige/DocQA.git
cd DocQA
pip install -r requirements.txt
```

### 2. 下载 Embedding 模型

```bash
python download_model.py
```

模型约 95MB，下载到 `models/bge-small-zh-v1.5/`。

### 3. 安装 Ollama + 拉取模型

下载 [Ollama](https://ollama.com) 并启动，然后：

```bash
ollama pull qwen2.5:7b
```

### 4. 跑起来

**Python 代码方式（推荐）：**

```python
from docqa.pipeline import DocQAPipeline

# 从配置构建流水线
pipeline = DocQAPipeline.from_config()

# 摄入文档
pipeline.ingest('your_document.pdf')

# 问答
answer = pipeline.ask('你的问题？')
print(answer)

# 只看检索结果
chunks = pipeline.retrieve('你的问题？', top_k=5)
for c in chunks:
    print(f'chunk {c.chunk_id} | page {c.source_page} | {c.text[:100]}')
```

**Streamlit UI 方式：**

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## 架构

```
            ┌─── 摄入管线 ───┐          ┌─── 查询管线 ───┐
            │                 │          │                  │
PDF → Parser → Chunker → Embedder    Query → [改写] → Retriever
                  │          │                        │
                  └──→ Store ←┘                   [重排序]
                  └──→ BM25索引 ←──────────────────┘
                                                     │
                                                [Chunk扩展]
                                                     │
                                               PromptBuilder
                                                     │
                                                   LLM
                                                     ↓
                                                  答案
```

### 目录结构

```
docqa/                        ← 新架构（主力，模块化可替换）
├── config.yaml               #   集中配置，换组件改这一处
├── config.py                 #   YAML 加载器
├── pipeline.py               #   流水线编排器
├── ingestion/                #   摄入管线
│   ├── base.py               #     接口: Parser / Chunker / Embedder / VectorStore
│   ├── parser.py             #     PyMuPDF 实现
│   ├── chunker.py            #     句子边界分块
│   ├── embedder.py           #     BGE-small-zh-v1.5
│   └── store.py              #     ChromaDB
├── retrieval/                #   检索管线
│   ├── base.py               #     接口: Retriever / Reranker
│   ├── dense.py              #     稠密向量检索
│   ├── bm25.py               #     BM25 关键词检索
│   ├── hybrid.py             #     混合检索 (BM25+向量, RRF)
│   ├── query_rewriter.py     #     查询改写 (LLM生成变体)
│   ├── multi_query.py        #     多查询检索
│   ├── chunk_expander.py     #     Chunk 上下文扩展 (small-to-big)
│   └── reranker.py           #     Cross-encoder 重排序
├── generation/               #   生成管线
│   ├── base.py               #     接口: PromptBuilder / LLM
│   ├── prompt.py             #     RAG Prompt 模板
│   └── llm.py                #     Ollama 调用
└── evaluation/               #   评估模块
    └── metrics.py            #     Recall / Precision / MRR / NDCG / HitRate

eval/                         ← 评估脚本和测试集
├── data/
│   ├── test_questions.jsonl       # 512chunk 测试集 (30题)
│   └── test_questions_256.jsonl   # 256chunk 测试集 (30题)
├── metrics.py                #   检索指标计算
├── testset.py                #   测试集加载/验证
├── oracle_test.py            #   Oracle 四条件瓶颈分析
├── run_optimization_eval.py  #   优化方案对比评估
└── manual_review.md          #   人工逐题审查记录

main_part/                    ← 旧架构（保留兼容，不再维护）
```

---

## 配置说明

所有参数集中在 `docqa/config.yaml`，换组件只需改这里：

```yaml
# === 文档摄入 ===
ingestion:
  parser:
    type: pymupdf            # pymupdf | pdfplumber | marker
  chunker:
    type: sentence           # sentence | fixed | recursive
    chunk_size: 256          # 小块便于精准检索
    overlap: 64
  chunk_expansion:
    enabled: true            # small-to-big: 检索用小块, 生成时拉取相邻chunk
  embedder:
    type: bge-small-zh       # bge-small-zh | bge-m3
    model_path: models/bge-small-zh-v1.5
    device: cpu              # cpu | cuda
  vector_store:
    type: chromadb
    persist_dir: ./chroma_db

# === 检索 ===
retrieval:
  mode: hybrid               # dense | bm25 | hybrid
  top_k: 10
  query_rewrite:
    enabled: true            # LLM 查询改写, 生成3个变体
  reranker:
    enabled: true            # Cross-encoder 重排序
    model: BAAI/bge-reranker-v2-m3
    candidate_pool: 50

# === 生成 ===
generation:
  llm:
    type: ollama
    model: qwen2.5:7b        # 换模型改这里
    temperature: 0.1
    max_tokens: 1024
```

### 检索模式切换

| mode | 说明 |
|------|------|
| `dense` | 纯稠密向量检索 |
| `bm25` | 纯 BM25 关键词检索 |
| `hybrid` | BM25 + 向量 RRF 融合（推荐） |

### 可选特性

| 特性 | 配置路径 | 默认 |
|------|---------|:---:|
| 查询改写 | `retrieval.query_rewrite.enabled` | true |
| 重排序 | `retrieval.reranker.enabled` | true |
| Chunk扩展 | `ingestion.chunk_expansion.enabled` | true |

---

## 依赖

```
chromadb>=1.1.0          # 向量数据库
sentence-transformers    # Embedding + Reranker
PyMuPDF>=1.24.0          # PDF 解析
rank-bm25                # BM25 关键词检索
jieba                    # 中文分词
pyyaml                   # 配置管理
requests                 # Ollama API 调用
numpy, scikit-learn      # 数值计算
```

完整列表见 `requirements.txt`。

---

## 项目状态

详见 [STATUS.md](STATUS.md)。

**已完成的：**
- ✅ Naive RAG MVP
- ✅ 评估基线（30题测试集 + 检索指标 + Oracle 分析）
- ✅ 混合检索（BM25 + 向量 RRF）
- ✅ Cross-encoder 重排序
- ✅ 查询改写（Query Rewriting）
- ✅ 小→大分块（Small-to-Big Chunking）
- ✅ 架构解耦重构（模块化+配置集中+可替换接口）

**待做的：**
- ◻️ 多文档跨域评估（验证泛化能力）
- ◻️ 完整优化效果对比（需要 GPU 加速评估）
- ◻️ Streamlit UI 切换到新架构

**当前效果：**

| 指标 | Baseline | 优化后 | 目标 |
|------|:------:|:------:|:---:|
| Recall@5 | 64% | 89% | ≥85% ✅ |
| Recall@10 | 83% | 91% | ≥90% ✅ |
| MRR | 0.52 | 0.71 | ≥0.75 ⚠️ |

---

## 环境注意事项

1. **代理冲突**：如果系统配置了 HTTP_PROXY，localhost 请求会被拦截。代码已内置 `_clear_proxy()` 方法。
2. **GPU vs CPU**：Reranker 在 CPU 上较慢（每对 50-100ms），GPU 快 10-50 倍。
3. **首次运行**：ChromaDB 会在项目根目录创建 `chroma_db/` 持久化目录。
