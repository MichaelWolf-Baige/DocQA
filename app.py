import streamlit as st
from main_part.pdf_parser import extract_text
from main_part.chunker import chunk_by_size
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.prompt_builder import build_rag_prompt
from main_part.generator import generate_answer

PDF_PATH = 'uploaded.pdf'

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
            chunks = st.session_state.embedder.embed_chunks(chunks)

            st.session_state.retriever = Retriever()
            st.session_state.retriever.index_chunks(chunks)

        st.success(f'已就绪，{len(chunks)} 个段落')

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

    if 'retriever' not in st.session_state:
        st.error('请先上传PDF文档')
    else:
        with st.spinner('思考中...'):
            results = st.session_state.retriever.search(
                question, st.session_state.embedder, top_k=5
            )
            prompt = build_rag_prompt(question, results)
            answer = generate_answer(prompt)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)
