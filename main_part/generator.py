"""LLM 生成模块 —— 通过 Ollama 原生 API 调用本地模型"""
import os
import requests

OLLAMA_API = 'http://localhost:11434/api/chat'
MODEL = 'qwen2.5:7b'


def _clear_proxy():
    """清除代理环境变量 —— localhost 不需要走代理"""
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(key, None)
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'


def generate_answer(prompt, system_prompt=None, temperature=0.3, max_tokens=1024):
    """
    调用 Ollama 本地模型生成回答。

    参数
    ----
    prompt : str
        用户 prompt（包含上下文和问题）
    system_prompt : str, optional
        系统提示
    temperature : float
        生成温度，RAG 场景建议 0.1-0.3
    max_tokens : int
        最大输出 token 数

    返回
    ----
    str : 模型生成的回答文本
    """
    if system_prompt is None:
        system_prompt = '你是文档问答助手，只根据用户提供的文档内容作答，引用具体信息，用中文。'

    body = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        'stream': False,
        'options': {
            'temperature': temperature,
            'num_predict': max_tokens,
        }
    }

    try:
        _clear_proxy()
        r = requests.post(OLLAMA_API, json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        content = data.get('message', {}).get('content', '')
        return content.strip() if content else ''
    except requests.exceptions.Timeout:
        return '[ERROR: 模型响应超时，请检查 Ollama 是否正常运行]'
    except Exception as e:
        return f'[ERROR: 生成失败 - {e}]'


if __name__ == "__main__":
    from pdf_parser import extract_text
    from chunker import chunk_by_size
    from embedder import Embedder
    from retriever import Retriever
    from prompt_builder import build_rag_prompt

    print(f'模型: {MODEL}')
    embedder = Embedder()
    retriever = Retriever()
    if retriever.collection.count() == 0:
        print('首次运行，建索引...')
        pages = extract_text(r'uploaded.pdf')
        chunks = chunk_by_size(pages)
        chunks = embedder.embed_chunks(chunks)
        retriever.index_chunks(chunks)
    else:
        print(f'索引已存在，共{retriever.collection.count()}条，跳过建索引')

    print('\n---测试问答---')
    query = '旅游合同编号是什么？'
    results = retriever.search(query, embedder, top_k=3)
    prompt = build_rag_prompt(query, results)

    print(f'调用 {MODEL}...\n')
    answer = generate_answer(prompt)
    print(f'回答：\n{answer}')

    print('\n---引用来源---')
    for r in results:
        print(f"第{r['source_page']}页 score = {r['score']}")
