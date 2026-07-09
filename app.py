"""
DocQA Streamlit 前端
====================
基于新架构 docqa/，支持多文档、检索可视化、模型切换。
"""
import os
import sys
import tempfile
import streamlit as st
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from docqa.pipeline import DocQAPipeline

# ---- 页面配置 ----
st.set_page_config(
    page_title="DocQA - 文档智能问答",
    page_icon="📄",
    layout="wide",
)

# ---- 初始化 session_state ----
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_ingested" not in st.session_state:
    st.session_state.files_ingested = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "show_retrieval" not in st.session_state:
    st.session_state.show_retrieval = False
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []


def build_pipeline(device, model, top_k, chunk_size, use_rerank, use_rewrite):
    """从参数构建 pipeline"""
    from docqa.config import load_config
    cfg = load_config()
    cfg['ingestion']['embedder']['device'] = device
    cfg['ingestion']['chunker']['chunk_size'] = chunk_size
    cfg['ingestion']['chunker']['overlap'] = chunk_size // 4
    cfg['retrieval']['top_k'] = top_k
    cfg['retrieval']['reranker']['enabled'] = use_rerank
    cfg['retrieval']['query_rewrite']['enabled'] = use_rewrite
    cfg['generation']['llm']['model'] = model

    p = DocQAPipeline(
        parser=DocQAPipeline._build_parser(cfg),
        chunker=DocQAPipeline._build_chunker(cfg),
        embedder=DocQAPipeline._build_embedder(cfg),
        vector_store=DocQAPipeline._build_vector_store(cfg),
        retriever=None,
        reranker=DocQAPipeline._build_reranker(cfg),
        prompt_builder=DocQAPipeline._build_prompt_builder(cfg),
        llm=DocQAPipeline._build_llm(cfg),
        use_multi_query=use_rewrite,
        use_chunk_expansion=True,
    )
    return p


# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 配置")

    # ---- 文档管理 ----
    st.subheader("📁 文档管理")
    uploaded_files = st.file_uploader(
        "上传 PDF（可多选）",
        type="pdf",
        accept_multiple_files=True,
        help="支持同时上传多个文件",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重建索引", use_container_width=True):
            if uploaded_files:
                with st.spinner("正在摄入文档..."):
                    pipeline = st.session_state.pipeline
                    all_chunks = 0
                    for f in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        n = pipeline.ingest(tmp_path, clear=(all_chunks == 0))
                        all_chunks += n
                        os.unlink(tmp_path)
                    st.session_state.total_chunks = all_chunks
                    st.session_state.files_ingested = [f.name for f in uploaded_files]
                    st.session_state.messages = []
                st.success(f"✅ {len(uploaded_files)} 个文件，{all_chunks} 个 chunk")
            else:
                st.warning("请先上传文件")

    with col2:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_retrieval = []

    # 已摄入状态
    if st.session_state.files_ingested:
        st.caption(f"已索引: {', '.join(st.session_state.files_ingested)}")
        st.caption(f"共 {st.session_state.total_chunks} 个 chunk")

    st.divider()

    # ---- 模型设置 ----
    st.subheader("🤖 模型设置")

    model = st.selectbox(
        "LLM 模型",
        ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"],
        index=0,
        help="小模型更快，大模型更准",
    )

    device = st.radio("设备", ["auto", "cpu", "cuda"], horizontal=True,
                      help="auto=自动检测 GPU")

    st.divider()

    # ---- 检索设置 ----
    st.subheader("🔍 检索设置")

    chunk_size = st.slider("Chunk 大小", 128, 1024, 256, step=64,
                           help="越小检索越精准，越大上下文越完整")

    top_k = st.slider("Top-K", 3, 20, 10,
                      help="返回多少个相关片段给 LLM")

    use_rerank = st.checkbox("重排序 (Reranker)", value=True,
                             help="Cross-encoder 精排，效果提升最大")
    use_rewrite = st.checkbox("查询改写", value=False,
                              help="LLM 改写查询，多路检索融合（较慢）")

    st.divider()

    # ---- 检索可视化开关 ----
    st.subheader("🔎 调试")
    st.session_state.show_retrieval = st.checkbox(
        "显示检索结果", value=st.session_state.show_retrieval,
        help="查看每次检索命中了哪些 chunk"
    )

    # ---- 初始化/重建 Pipeline ----
    if device == "auto":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    if st.session_state.pipeline is None or st.button("🔧 应用配置"):
        with st.spinner("正在初始化..."):
            st.session_state.pipeline = build_pipeline(
                dev, model, top_k, chunk_size, use_rerank, use_rewrite
            )
        st.success(f"✅ Pipeline 就绪 (device={dev}, model={model})")


# ==================== 主界面 ====================
st.title("📄 DocQA — 文档智能问答")

# ---- 快速开始提示 ----
if not st.session_state.files_ingested:
    st.info("👈 先在左侧上传 PDF 文件，然后点击「重建索引」开始使用")

# ---- 对话历史 ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 如果该消息有检索结果且开启了显示
        if msg["role"] == "assistant" and "chunks" in msg:
            if st.session_state.show_retrieval:
                with st.expander("🔍 检索详情", expanded=False):
                    for i, c in enumerate(msg["chunks"]):
                        src = f"`{c.source_file}` " if c.source_file else ""
                        score = c.metadata.get("rerank_score") or c.metadata.get("score", 0)
                        st.markdown(
                            f"**#{i+1}** {src}第{c.source_page}页 "
                            f"| 相关度: `{score:.4f}`"
                        )
                        st.text(c.text[:300] + ("..." if len(c.text) > 300 else ""))
                        if i < len(msg["chunks"]) - 1:
                            st.divider()

# ---- 输入框 ----
if prompt := st.chat_input("输入你的问题..."):
    pipeline = st.session_state.pipeline

    if pipeline is None or not st.session_state.files_ingested:
        st.error("请先在左侧上传文档并点击「重建索引」")
    else:
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 检索 + 生成
        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                chunks = pipeline.retrieve(prompt, top_k=top_k)

            with st.spinner("生成回答..."):
                answer = pipeline.ask(prompt)

            st.markdown(answer)

            # 来源引用
            if chunks:
                sources = set()
                for c in chunks[:5]:
                    src = c.source_file or "文档"
                    sources.add(f"{src} 第{c.source_page}页")
                if sources:
                    st.caption("📖 参考: " + " | ".join(list(sources)[:5]))

            # 检索详情
            if st.session_state.show_retrieval:
                with st.expander("🔍 检索详情", expanded=False):
                    for i, c in enumerate(chunks[:top_k]):
                        src = f"`{c.source_file}` " if c.source_file else ""
                        score = c.metadata.get("rerank_score") or c.metadata.get("score", 0)
                        st.markdown(
                            f"**#{i+1}** {src}第{c.source_page}页 "
                            f"| 相关度: `{score:.4f}`"
                        )
                        st.text(c.text[:300] + ("..." if len(c.text) > 300 else ""))
                        if i < len(chunks[:top_k]) - 1:
                            st.divider()

        # 保存
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks[:top_k] if chunks else [],
        })
        st.session_state.last_retrieval = chunks[:top_k] if chunks else []
