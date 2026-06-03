from openai import OpenAI

OLLAMA_BASE = 'http://localhost:11434/v1'
MODEL = 'qwen2.5:1.5b'


def generate_answer(prompt, api_key='not-needed'):
    client = OpenAI(
        api_key=api_key,
        base_url=OLLAMA_BASE
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': '你是文档问答助手，只根据用户提供的文档内容作答，引用具体信息，用中文。'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from pdf_parser import extract_text
    from chunker import chunk_by_size
    from embedder import Embedder
    from retriever import Retriever
    from prompt_builder import build_rag_prompt

    print('加载模型...')
    embedder = Embedder()
    retriever = Retriever()
    if retriever.collection.count() == 0:
        print('首次运行，建索引...')
        pages = extract_text(r'sample.pdf')
        chunks = chunk_by_size(pages)
        chunks = embedder.embed_chunks(chunks)
        retriever.index_chunks(chunks)
    else:
        print(f'索引已存在，共{retriever.collection.count()}条，跳过建索引')

    print('\n---测试问答---')
    query = '如何退款'
    results = retriever.search(query, embedder, top_k=3)
    prompt = build_rag_prompt(query, results)

    print('调用本地Qwen2.5 1.5B...\n')
    answer = generate_answer(prompt)
    print(f'回答：\n{answer}')

    print('\n---引用来源---')
    for r in results:
        print(f"第{r['source_page']}页 score = {r['score']}")





