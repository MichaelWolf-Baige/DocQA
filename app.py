"""
DocQA Streamlit 前端
====================
上传多份 PDF → 选模型 → 问答，极简交互。
"""
import os
import tempfile
import streamlit as st
import requests

from docqa.pipeline import DocQAPipeline

# ── 页面设置 ──
st.set_page_config(page_title='DocQA', page_icon='📄', layout='wide')
st.title('📄 DocQA — 文档智能问答')

# ── 获取 Ollama 可用模型 ──
@st.cache_data(ttl=30)
def get_ollama_models():
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        return models if models else ['qwen2.5:7b']
    except Exception:
        return ['qwen2.5:7b']

# ── 侧边栏 ──
with st.sidebar:
    st.header('⚙️ 设置')

    # 模型选择
    available_models = get_ollama_models()
    default_idx = available_models.index('qwen2.5:7b') if 'qwen2.5:7b' in available_models else 0
    model = st.selectbox(
        '生成模型',
        available_models,
        index=default_idx,
        help='7b 质量好但慢，小的快但容易出错。'
    )

    st.divider()

    # 上传多个 PDF
    st.header('📁 上传文档')
    uploaded_files = st.file_uploader(
        '选择 PDF 文件（可多选）',
        type='pdf',
        accept_multiple_files=True,
        help='支持同时上传多份 PDF，自动合并分析。'
    )

    if uploaded_files:
        # 模型变了也重建
        need_rebuild = (
            'pipeline' not in st.session_state
            or st.session_state.get('last_model') != model
            or st.session_state.get('last_files') != tuple(f.name for f in uploaded_files)
        )

        if need_rebuild:
            with st.spinner('正在解析文档...'):
                # 保存所有上传的 PDF
                pdf_paths = []
                for uf in uploaded_files:
                    path = os.path.join(tempfile.gettempdir(), f'docqa_{uf.name}')
                    with open(path, 'wb') as f:
                        f.write(uf.read())
                    pdf_paths.append(path)

                # 构建流水线（ingest 支持多文件）
                pipeline = DocQAPipeline.from_config('docqa/config_cpu.yaml')
                pipeline.llm.model = model
                n = pipeline.ingest(pdf_paths)

                st.session_state.pipeline = pipeline
                st.session_state.last_model = model
                st.session_state.last_files = tuple(uf.name for uf in uploaded_files)
                st.session_state.n_chunks = n
                st.session_state.n_files = len(pdf_paths)

        st.success(
            f'已就绪 — {st.session_state.n_chunks} 个段落 '
            f'({st.session_state.n_files} 份文档) | 模型: {model}'
        )

    # 当前状态
    if 'pipeline' in st.session_state:
        st.caption(f'文档: {st.session_state.get("n_files", "?")} 份, '
                   f'{st.session_state.n_chunks} chunks')
        st.caption(f'模型: {st.session_state.last_model}')

# ── 对话区 ──
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── 输入框 ──
if question := st.chat_input('输入你的问题...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    if 'pipeline' not in st.session_state:
        st.error('请先在左侧上传 PDF 文档')
    else:
        pipeline = st.session_state.pipeline

        with st.spinner('检索中...'):
            chunks = pipeline.retrieve(question)

        with st.spinner('生成回答...'):
            answer = pipeline.ask(question)

        # 拼接引用来源
        sources = []
        seen = set()
        for c in chunks[:5]:
            key = f'{c.source_file}:{c.source_page}' if c.source_file else str(c.source_page)
            if key not in seen and c.source_page > 0:
                label = f'{c.source_file} 第{c.source_page}页' if c.source_file else f'第{c.source_page}页'
                sources.append(label)
                seen.add(key)

        if sources and answer and not answer.startswith('[ERROR'):
            answer += f'\n\n📖 参考: {" | ".join(sources[:5])}'

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
