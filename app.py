"""
DocQA Streamlit 前端
====================
上传 PDF → 建立知识库 → 问答。

设计原则:
  - 摄入只在上传新文件时触发，问答不会重新解析
  - 新文件追加到已有索引，不覆盖
  - 同名文件自动跳过（去重）
  - 文件持久化保存到 docqa_uploads/
"""
import os
from typing import List, Set

import streamlit as st
import requests

from docqa.pipeline import DocQAPipeline
from docqa.ingestion.base import Chunk

# ── 常量 ──
UPLOAD_DIR = 'docqa_uploads'
CONFIG_PATH = 'docqa/config_cpu.yaml'
FALLBACK_MODEL = 'qwen2.5:7b'
OLLAMA_URL = 'http://localhost:11434/api/tags'

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── 工具 ──
@st.cache_data(ttl=30)
def list_ollama_models() -> List[str]:
    try:
        r = requests.get(OLLAMA_URL, timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        return models if models else [FALLBACK_MODEL]
    except Exception:
        return [FALLBACK_MODEL]


def _safe_index(models: List[str], preferred: str) -> int:
    try:
        return models.index(preferred)
    except ValueError:
        return 0


def _source_label(c: Chunk) -> str:
    if c.source_file and c.source_page > 0:
        return f'{c.source_file} 第{c.source_page}页'
    if c.source_page > 0:
        return f'第{c.source_page}页'
    return c.source_file or '未知来源'


# ── 初始化 session ──
_defaults = {
    'messages': [],
    'ingested_files': {},   # filename → chunk_count
    'pipeline': None,
    'total_chunks': 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── 页面 ──
st.set_page_config(page_title='DocQA', page_icon='📄', layout='wide')
st.title('📄 DocQA — 文档智能问答')

# ═══ 侧边栏 ═══
with st.sidebar:
    st.header('⚙️ 设置')
    models = list_ollama_models()
    model = st.selectbox(
        '生成模型', models,
        index=_safe_index(models, FALLBACK_MODEL),
    )

    st.divider()
    st.header('📁 知识库')

    # 上传（支持 Ctrl+多选）
    uploaded = st.file_uploader(
        '上传 PDF 文件',
        type='pdf',
        accept_multiple_files=True,
        key='uploader',
        help='可按住 Ctrl / Cmd 多选文件。已存在的文件会自动跳过。',
    )

    # ── 检查新文件，仅在新文件出现时摄入 ──
    new_pdfs: List[str] = []
    skipped: List[str] = []

    if uploaded:
        for uf in uploaded:
            fname = uf.name
            if fname in st.session_state.ingested_files:
                skipped.append(fname)
                continue
            dst = os.path.join(UPLOAD_DIR, fname)
            with open(dst, 'wb') as fh:
                fh.write(uf.read())
            new_pdfs.append(dst)

    # ── 摄入新文件 ──
    if new_pdfs:
        with st.spinner(f'正在解析 {len(new_pdfs)} 个新文件…'):
            try:
                if st.session_state.pipeline is None:
                    pipeline = DocQAPipeline.from_config(CONFIG_PATH)
                    pipeline.llm.model = model
                    pipeline.ingest(new_pdfs, clear=True)
                else:
                    pipeline = st.session_state.pipeline
                    pipeline.llm.model = model
                    pipeline.ingest(new_pdfs, clear=False)

                # 成功后才更新状态，避免磁盘与 session 不一致
                for fp in new_pdfs:
                    fname = os.path.basename(fp)
                    count = len([c for c in pipeline._all_chunks
                                 if getattr(c, 'source_file', '') == fname])
                    st.session_state.ingested_files[fname] = count
                st.session_state.pipeline = pipeline
                st.session_state.total_chunks = pipeline.vector_store.count()
            except Exception as e:
                st.error(f'解析失败: {e}')
                # 删除已写入但未索引的文件
                for fp in new_pdfs:
                    if os.path.exists(fp):
                        os.remove(fp)
        st.rerun()

    if skipped:
        st.caption(f'已跳过 {len(skipped)} 个重复文件：{", ".join(skipped)}')

    # ── 已加载文档列表 ──
    if st.session_state.ingested_files:
        st.write('**已加载文档：**')
        for fname, count in sorted(st.session_state.ingested_files.items()):
            st.caption(f'📄 {fname} ({count} chunks)')

        st.caption(
            f'共 {len(st.session_state.ingested_files)} 份文档, '
            f'{st.session_state.total_chunks} 个段落  |  模型: {model}'
        )

        if st.button('🗑️ 清空知识库', use_container_width=True):
            st.session_state.pipeline = None
            st.session_state.ingested_files = {}
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption('尚未上传文档。')

    # ── 同步模型 ──
    if st.session_state.pipeline is not None:
        st.session_state.pipeline.llm.model = model

# ═══ 对话区 ═══
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if question := st.chat_input('输入你的问题…'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    if st.session_state.pipeline is None:
        st.error('请先在左侧上传 PDF 文档')
    else:
        pipeline = st.session_state.pipeline

        with st.spinner('检索中…'):
            chunks = pipeline.retrieve(question)

        with st.spinner('生成回答…'):
            answer = pipeline.ask(question)

        seen = set()
        labels = []
        for c in chunks[:5]:
            key = (c.source_file, c.source_page)
            if key not in seen and c.source_page > 0:
                labels.append(_source_label(c))
                seen.add(key)

        if labels and answer and not answer.startswith('[ERROR'):
            answer += '\n\n📖 参考: ' + ' | '.join(labels)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
