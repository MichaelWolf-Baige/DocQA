# DocQA — 最小 RAG 文档问答系统

一个不到 200 行核心代码的 RAG（检索增强生成）MVP，上传 PDF 即可与文档对话。

## 架构

```
PDF 上传 → 文本解析 → 句子级分块 → BGE 向量化 → ChromaDB 索引
                                                    ↓
用户提问 → 问题向量化 → 语义检索(top-k) → Prompt 拼接 → Ollama 生成回答
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 Embedding 模型

```bash
python download_model.py
```

模型约 95MB，下载到 `models/bge-small-zh-v1.5/`。

### 3. 安装并启动 Ollama

下载 [Ollama](https://ollama.com)，然后拉取中文模型：

```bash
ollama pull qwen2.5:1.5b
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，上传 PDF 即可开始问答。

## 模块说明

```
├── app.py                    # Streamlit 前端
├── test.py                   # 端到端测试 + top_k 分析
├── download_model.py         # 模型下载脚本
├── requirements.txt
├── main_part/
│   ├── pdf_parser.py         # PDF → 逐页文本 (PyMuPDF)
│   ├── chunker.py            # 句子边界分块 + overlap
│   ├── embedder.py           # BGE-small-zh-v1.5 向量化
│   ├── retriever.py          # ChromaDB 语义检索
│   ├── prompt_builder.py     # RAG Prompt 构建
│   └── generator.py          # Ollama LLM 生成
├── models/                   # 本地模型文件（需自行下载）
└── chroma_db/                # 向量数据库（运行时生成）
```

## 配置

| 组件 | 默认值 | 说明 |
|------|--------|------|
| Embedding 模型 | `BAAI/bge-small-zh-v1.5` | 中文，512 维，CPU 运行 |
| 分块大小 | 512 字符 | `chunker.py` 中可调 |
| 重叠大小 | 128 字符 | 保证语义连续性 |
| 检索数量 | top-10 | `app.py` 中可调 |
| 生成模型 | `qwen2.5:1.5b` | `generator.py` 中可换其他 Ollama 模型 |

## 技术选型

- **Embedding**：BGE-small 中文模型，CPU 上单句编码 < 0.1s
- **向量存储**：ChromaDB 本地持久化，余弦距离检索
- **LLM**：通过 Ollama 调用本地模型，数据不出本机
