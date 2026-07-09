"""
DocQA Streamlit 前端
====================
上传 PDF → 选模型 → 问答，极简交互。
"""
import os
import streamlit as st
import requests

from docqa.pipeline import DocQAPipeline

PDF_PATH = 'uploaded.pdf'

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
    model = st.selectbox(
        '生成模型',
        available_models,
        index=available_models.index('qwen2.5:7b') if 'qwen2.5:7b' in available_models else 0,
        help='选择 Ollama 中的模型。7b 质量好但慢，1.5b 快但容易出错。'
    )

    st.divider()

    # 上传 PDF
    st.header('📁 上传文档')
    uploaded = st.file_uploader('选择 PDF 文件', type='pdf',
                                help='支持中文 PDF，自动解析文本内容。')

    if uploaded:
        # 保存 PDF
        with open(PDF_PATH, 'wb') as f:
            f.write(uploaded.read())

        # 构建流水线
        with st.spinner('正在解析文档...'):
            if 'pipeline' not in st.session_state or st.session_state.get('last_model') != model:
                pipeline = DocQAPipeline.from_config('docqa/config_cpu.yaml')
                # 用选中的模型覆盖配置
                pipeline.llm.model = model
                n = pipeline.ingest(PDF_PATH)
                st.session_state.pipeline = pipeline
                st.session_state.last_model = model
                st.session_state.n_chunks = n

        st.success(f'已就绪 — {st.session_state.n_chunks} 个段落 | 模型: {model}')

    # 显示当前状态
    if 'pipeline' in st.session_state:
        st.caption(f'文档: {st.session_state.n_chunks} chunks')
        st.caption(f'模型: {st.session_state.last_model}')

# ── 对话区 ──
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── 输入框 ──
if question := st.chat_input('输入你的问题...'):
    # 显示用户消息
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    if 'pipeline' not in st.session_state:
        st.error('请先在左侧上传 PDF 文档')
    else:
        pipeline = st.session_state.pipeline

        # 检索 + 生成
        with st.spinner('检索中...'):
            chunks = pipeline.retrieve(question)

        with st.spinner('生成回答...'):
            answer = pipeline.ask(question)

        # 拼接引用来源
        sources = []
        seen_pages = set()
        for c in chunks[:5]:
            if c.source_page not in seen_pages and c.source_page > 0:
                sources.append(str(c.source_page))
                seen_pages.add(c.source_page)

        if sources and answer and not answer.startswith('[ERROR'):
            answer += f'\n\n📖 参考: 第 {", ".join(sources)} 页'

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
