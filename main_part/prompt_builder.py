def build_rag_prompt(query, retrieved_chunks):
    '''把检索结果和用户问题拼成完整prompt，针对1.5B小模型优化'''
    context_parts = []
    for c in retrieved_chunks:
        part = f'[第{c["source_page"]}页] {c["text"]}'
        context_parts.append(part)

    context = '\n\n'.join(context_parts)

    prompt = (
        f'从以下文档片段中找出问题的答案，引用文档中的具体信息作答。'
        f'如果确实找不到相关信息，才回复”文档未提及”。\n'
        f'\n'
        f'文档片段：\n{context}\n'
        f'\n'
        f'问题：{query}\n'
        f'答案：'
    )

    return prompt

if __name__ == "__main__":
    from pdf_parser import extract_text
    from chunker import chunk_by_size
    from embedder import Embedder
    from retriever import Retriever

    embedder = Embedder()
    pages = extract_text(r'sample.pdf')
    chunks = chunk_by_size(pages)
    chunks = embedder.embed_chunks(chunks)

    retriever = Retriever()
    retriever.index_chunks(chunks)

    query = '怎么退款'
    results = retriever.search(query, embedder, top_k=3)
    prompt = build_rag_prompt(query, results)

    print(prompt)
    print(f'\n---prompt总长度：{len(prompt)}字符---')


















