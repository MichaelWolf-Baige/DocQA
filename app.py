"""
DocQA — 文档智能问答（Streamlit UI）
=====================================
本文件只负责交互层，所有 RAG 逻辑都委托给 docqa.pipeline.DocQAPipeline。

UI 与核心管线共用同一套组件（混合检索 / Rerank / HyDE / 多查询改写），
不会出现"演示用的是优化后的管线、UI 跑的是朴素管线"的脱节。

切换检索策略 / Reranker / LLM 后端，只需改 docqa/config.yaml，无需改本文件。
"""
import streamlit as st
from docqa.pipeline import DocQAPipeline
from docqa.config import load_config

st.set_page_config(page_title='DocQA', page_icon='📄')
st.title('DocQA - 文档智能问答')

CFG = load_config()
RCFG = CFG.get('retrieval', {})
MODE = RCFG.get('mode', 'hybrid')


def get_pipeline() -> DocQAPipeline:
    """构建（或复用缓存中的）DocQAPipeline 实例。"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = DocQAPipeline.from_config()
    return st.session_state.pipeline


def reset_pipeline():
    """配置或文档变化时，重建 pipeline（释放旧 embedder/reranker 等资源）。"""
    st.session_state.pop('pipeline', None)
    st.session_state.pop('ingested_file_id', None)


# --- 左侧栏：上传文档 ---
with st.sidebar:
    st.header('📁 上传文档')
    uploaded = st.file_uploader('选择 PDF 文件', type='pdf')

    st.caption(f'当前检索模式：**{MODE}**')
    st.caption(f'Reranker：{"开启" if RCFG.get("reranker",{}).get("enabled") else "关闭"}')

    if uploaded is not None:
        file_id = uploaded.file_id
        if st.session_state.get('ingested_file_id') != file_id:
            with st.spinner('正在解析文档并建索引...'):
                pipeline = get_pipeline()
                import tempfile, os as _os
                tmp_path = _os.path.join(tempfile.gettempdir(), f'docqa_{file_id}.pdf')
                with open(tmp_path, 'wb') as f:
                    f.write(uploaded.read())
                try:
                    n = pipeline.ingest(tmp_path, clear=True)
                finally:
                    try: _os.remove(tmp_path)
                    except OSError: pass
                st.session_state.ingested_file_id = file_id
                st.session_state.chunk_count = n
            st.success(f'已就绪，共 {st.session_state.chunk_count} 个段落')


# --- 对话历史 ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# --- 输入框 ---
if question := st.chat_input('输入你的问题...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    if st.session_state.get('ingested_file_id') is None:
        st.error('请先在左侧上传 PDF 文档')
    else:
        pipeline = get_pipeline()
        with st.spinner('检索中...'):
            chunks = pipeline.retrieve(question)
        with st.spinner('生成回答中...'):
            answer = pipeline.ask(question)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
            with st.expander('引用来源', expanded=False):
                for i, c in enumerate(chunks, 1):
                    src = f'{c.source_file} 第{c.source_page}页' if c.source_file else f'第{c.source_page}页'
                    st.caption(f'[{i}] {src}')
                    st.text(c.text[:200] + ('...' if len(c.text) > 200 else ''))