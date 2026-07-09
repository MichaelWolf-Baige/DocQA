"""DocQA Streamlit 前端 — 上传 PDF → 问答"""
import os, sys, tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from docqa.pipeline import DocQAPipeline

st.set_page_config(page_title="DocQA", page_icon="📄", layout="wide")

# ── 初始化 ──
for key, val in [
    ("pipeline", None), ("messages", []), ("files_done", False),
    ("chunk_count", 0), ("show_chunks", False),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── 侧边栏 ──
with st.sidebar:
    st.header("📁 文档")
    uploaded = st.file_uploader("上传 PDF（可多选）", "pdf", accept_multiple_files=True)

    if uploaded and st.button("🔨 建立索引", use_container_width=True):
        # 保存临时文件
        tmp_paths = []
        for f in uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.getbuffer())
            tmp_paths.append(tmp.name)

        # 构建 pipeline
        st.session_state.pipeline = DocQAPipeline.from_config()
        p = st.session_state.pipeline

        with st.spinner("正在解析 + 向量化..."):
            total = 0
            for i, path in enumerate(tmp_paths):
                n = p.ingest(path, clear=(i == 0))
                total += n
                os.unlink(path)

        st.session_state.chunk_count = total
        st.session_state.files_done = True
        st.session_state.messages = []
        st.rerun()

    if st.session_state.files_done:
        st.success(f"已索引 {st.session_state.chunk_count} 个片段")
        for f in (uploaded or []):
            st.caption(f"📄 {f.name}")

    st.divider()

    dev = "GPU" if torch.cuda.is_available() else "CPU"
    st.caption(f"🔧 Embedding & Reranker: **{dev}**")
    st.caption(f"🤖 LLM: **qwen2.5:7b** (Ollama)")

    st.divider()
    st.session_state.show_chunks = st.checkbox("🔍 显示检索到的片段", value=st.session_state.show_chunks)

    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []

# ── 主界面 ──
st.title("📄 DocQA — 文档智能问答")

if not st.session_state.files_done:
    st.info("👈 上传 PDF → 点击「建立索引」开始")

# 对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入
if prompt := st.chat_input("输入问题..."):
    p = st.session_state.pipeline
    if p is None:
        st.error("请先上传文档并建立索引")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                chunks = p.retrieve(prompt, top_k=10)

            with st.spinner("生成回答..."):
                answer = p.ask(prompt)

            st.markdown(answer)

            # 来源
            if chunks:
                srcs = sorted(set(f"{c.source_file or 'PDF'} p{c.source_page}" for c in chunks[:5]))
                st.caption("📖 " + " | ".join(srcs))

            # 检索片段
            if st.session_state.show_chunks:
                with st.expander("🔍 命中的片段", expanded=False):
                    for i, c in enumerate(chunks[:10]):
                        s = c.metadata.get("rerank_score") or c.metadata.get("score", 0)
                        st.markdown(f"**#{i+1}** `{c.source_file}` p{c.source_page} — score `{s:.4f}`")
                        st.text(c.text[:250] + ("..." if len(c.text) > 250 else ""))
                        if i < min(len(chunks), 10) - 1:
                            st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
