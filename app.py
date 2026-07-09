"""
DocQA Streamlit 前端
====================
上传多份 PDF → 选模型 → 问答。
"""
import os
import tempfile
from typing import List, Optional

import streamlit as st
import requests

from docqa.pipeline import DocQAPipeline
from docqa.ingestion.base import Chunk

# ── 常量 ──
CONFIG_PATH = 'docqa/config_cpu.yaml'
FALLBACK_MODEL = 'qwen2.5:7b'
OLLAMA_TAGS_URL = 'http://localhost:11434/api/tags'


# ── 工具函数 ──
@st.cache_data(ttl=30)
def list_ollama_models() -> List[str]:
    """获取 Ollama 可用模型列表，失败返回 fallback。"""
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        return models if models else [FALLBACK_MODEL]
    except Exception:
        return [FALLBACK_MODEL]


def _safe_model_index(models: List[str], preferred: str) -> int:
    """安全取模型索引：找不到 preferred 时回退到第一个。"""
    try:
        return models.index(preferred)
    except ValueError:
        return 0


def _files_signature(uploaded_files) -> Optional[tuple]:
    """上传文件列表 → 可用于比较的签名，无文件时返回 None。"""
    return tuple(f.name for f in uploaded_files) if uploaded_files else None


def _chunk_source_label(c: Chunk) -> str:
    """chunk → 人类可读的来源标签。"""
    if c.source_file and c.source_page > 0:
        return f'{c.source_file} 第{c.source_page}页'
    if c.source_page > 0:
        return f'第{c.source_page}页'
    return c.source_file or '未知来源'


# ── 页面 ──
st.set_page_config(page_title='DocQA', page_icon='📄', layout='wide')
st.title('📄 DocQA — 文档智能问答')

# ════════════════════════════════════════════
# 侧边栏
# ════════════════════════════════════════════
with st.sidebar:
    st.header('⚙️ 设置')

    models = list_ollama_models()
    model = st.selectbox(
        '生成模型',
        models,
        index=_safe_model_index(models, FALLBACK_MODEL),
        help='7b 质量好但慢。'

    )

    st.divider()
    st.header('📁 上传文档')

    uploaded_files = st.file_uploader(
        '选择 PDF 文件（可多选）',
        type='pdf',
        accept_multiple_files=True,
    )

    current_sig = _files_signature(uploaded_files)

    if uploaded_files:
        need_rebuild = (
            'pipeline' not in st.session_state
            or st.session_state.get('last_model') != model
            or st.session_state.get('last_sig') != current_sig
        )

        if need_rebuild:
            with st.spinner('正在解析文档…'):
                pdf_paths = []
                for uf in uploaded_files:
                    path = os.path.join(tempfile.gettempdir(), f'docqa_{uf.name}')
                    with open(path, 'wb') as fh:
                        fh.write(uf.read())
                    pdf_paths.append(path)

                pipeline = DocQAPipeline.from_config(CONFIG_PATH)
                pipeline.llm.model = model
                n = pipeline.ingest(pdf_paths)

                st.session_state.pipeline = pipeline
                st.session_state.last_model = model
                st.session_state.last_sig = current_sig
                st.session_state.n_chunks = n
                st.session_state.n_files = len(pdf_paths)

        st.success(
            f'已就绪 — {st.session_state.n_chunks} 个段落 '
            f'({st.session_state.n_files} 份文档) | 模型: {model}'
        )

    if 'pipeline' in st.session_state:
        st.caption(
            f'文档: {st.session_state.get("n_files", "?")} 份, '
            f'{st.session_state.n_chunks} chunks  |  模型: {st.session_state.last_model}'
        )

# ════════════════════════════════════════════
# 对话区
# ════════════════════════════════════════════
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if question := st.chat_input('输入你的问题…'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    if 'pipeline' not in st.session_state:
        st.error('请先在左侧上传 PDF 文档')
    else:
        pipeline = st.session_state.pipeline

        with st.spinner('检索中…'):
            chunks = pipeline.retrieve(question)

        with st.spinner('生成回答…'):
            answer = pipeline.ask(question)

        # 去重、拼接来源引用
        seen = set()
        labels = []
        for c in chunks[:5]:
            key = (c.source_file, c.source_page)
            if key not in seen and c.source_page > 0:
                labels.append(_chunk_source_label(c))
                seen.add(key)

        if labels and answer and not answer.startswith('[ERROR'):
            answer += '\n\n📖 参考: ' + ' | '.join(labels)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
