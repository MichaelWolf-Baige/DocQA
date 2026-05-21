def build_rag_prompt(query, retrived_chunks):
    '''把检索结果和用户问题拼成完整prompt'''
    context_parts = []
    for i, c in enumerate(retrived_chunks):
        part = (
            f'[来源第{c["source_page"]}页] 相关度：{c["score"]}\n'
            f'{c["text"]}'
        )
        context_parts.append(part)

    context = '\n\n--\n\n'.join(context_parts)

    prompt = (
        '你是一个帮助用户理解文档内容的助手。'
        '请严格根据以下文档片段回答问题。\n\n'
        f'文档内容：\n{context}\n\n'
        f'用户问题：{query}\n\n'
        '回答要求：\n'
        '1.如果文档中有相关信息，请给出准确回答，并标注引用来源页码\n'
        '2.如果文档中没有相关信息，请明确说“文档中未提及”，不要编造\n'
        '3.回答尽量简洁'
    )

    return prompt

if __name__ == "__main__":
    from pdf_parser import extract_text
    from chunker import chunk_by_size
    from embedder import Embedder
    from retriever import Retriever

    embedder = Embedder()
    pages = extract_text(r'D:\桌面\I.pdf')
    chunks = chunk_by_size(pages)
    chunks = embedder.embed_chunks(chunks)

    retriever = Retriever()
    retriever.index_chunks(chunks)

    query = '怎么退款'
    results = retriever.search(query, embedder, top_k=3)
    prompt = build_rag_prompt(query, results)

    print(prompt)
    print(f'\n---prompt总长度：{len(prompt)}字符---')


















