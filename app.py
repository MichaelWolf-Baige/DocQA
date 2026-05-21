import numpy as np
import streamlit as st
from main_part.pdf_parser import extract_text
from main_part.chunker import chunk_by_size
from main_part.embedder import Embedder
from main_part.prompt_builder import build_rag_prompt
from main_part.generator import generate_answer

PDF_PATH = 'uploaded.pdf'

def cosine_search(query, embedder, chunks, top_k=5):
    """向量化余弦检索，一次矩阵乘法搞定全部chunk"""
    q_vec = np.array(embedder.embed_query(query))
    chunk_matrix = np.array([c['embedding'] for c in chunks])
    # sentence-transformers输出已L2归一化，点积即余弦相似度
    sims = np.dot(chunk_matrix, q_vec)
    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    results = []
    for i in top_idx:
        c = chunks[i]
        results.append({
            'chunk_id': c['chunk_id'],
            'text': c['text'],
            'source_page': c['source_page'],
            'score': round(float(sims[i]), 4),
        })
    return results

st.set_page_config(page_title='DocQA', page_icon='📄')
st.title('DocQA - 文档智能问答')

# --- 左侧栏：上传文档 ---
with st.sidebar:
    st.header('📁 上传文档')
    uploaded = st.file_uploader('选择PDF文件', type='pdf')

    if uploaded:
        with open(PDF_PATH, 'wb') as f:
            f.write(uploaded.read())

        with st.spinner('正在解析文档...'):
            if 'embedder' not in st.session_state:
                st.session_state.embedder = Embedder()

            pages = extract_text(PDF_PATH)
            chunks = chunk_by_size(pages)
            st.session_state.chunks = st.session_state.embedder.embed_chunks(chunks)

        st.success(f'已就绪，{len(st.session_state.chunks)} 个段落')

    if st.button('清空对话'):
        st.session_state.messages = []

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

    if 'chunks' not in st.session_state:
        st.error('请先上传PDF文档')
    else:
        with st.spinner('检索中...'):
            results = cosine_search(
                question, st.session_state.embedder, st.session_state.chunks, top_k=5
            )
            prompt = build_rag_prompt(question, results)
        with st.spinner('生成回答中...'):
            answer = generate_answer(prompt)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
