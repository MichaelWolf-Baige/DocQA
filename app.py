"""DocQA"""
import os, sys, tempfile
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

st.set_page_config(page_title="DocQA", page_icon="📄")

for key, default in [("pipe", None), ("ready", False), ("msgs", []), ("_names", set())]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 侧边栏 ──
with st.sidebar:
    st.header("配置")
    model_name = st.selectbox("LLM 模型", ["qwen2.5:1.5b", "qwen2.5:7b"], index=0)
    device = st.selectbox("Embedding 设备", ["cuda", "cpu"],
        index=0 if torch.cuda.is_available() else 1)
    files = st.file_uploader("上传 PDF", "pdf", accept_multiple_files=True)

    if files:
        new_names = {f.name for f in files}
        if new_names != st.session_state._names:
            from docqa.config import load_config
            from docqa.pipeline import DocQAPipeline
            cfg = load_config()
            cfg["ingestion"]["embedder"]["device"] = device
            cfg["generation"]["llm"]["model"] = model_name
            with st.spinner("加载模型 + 处理文档..."):
                p = DocQAPipeline(
                    parser=DocQAPipeline._build_parser(cfg),
                    chunker=DocQAPipeline._build_chunker(cfg),
                    embedder=DocQAPipeline._build_embedder(cfg),
                    vector_store=DocQAPipeline._build_vector_store(cfg),
                    retriever=None,
                    reranker=DocQAPipeline._build_reranker(cfg),
                    prompt_builder=DocQAPipeline._build_prompt_builder(cfg),
                    llm=DocQAPipeline._build_llm(cfg),
                    use_multi_query=cfg["retrieval"]["query_rewrite"]["enabled"],
                    use_chunk_expansion=cfg["ingestion"]["chunk_expansion"]["enabled"],
                )
                st.session_state.pipe = p
                total = 0
                for i, f in enumerate(files):
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(f.getbuffer())
                    tmp.close()
                    total += p.ingest(tmp.name, clear=(i == 0))
                    os.unlink(tmp.name)
            st.session_state.ready = True
            st.session_state.msgs = []
            st.session_state._names = new_names

    if st.session_state.ready:
        cnt = st.session_state.pipe.vector_store.count() if st.session_state.pipe else 0
        st.success(f"就绪 — {cnt} 片段")

# ── 主界面 ──
st.title("📄 DocQA")

if st.session_state.ready:
    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.write(m["content"])
else:
    st.info("上传 PDF，自动建立索引")

if q := st.chat_input("输入问题" if st.session_state.ready else "请先上传 PDF"):
    if not st.session_state.ready:
        st.warning("请先在侧边栏上传 PDF")
    else:
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("..."):
                ans = st.session_state.pipe.ask(q)
            st.write(ans)
        st.session_state.msgs.append({"role": "assistant", "content": ans})
