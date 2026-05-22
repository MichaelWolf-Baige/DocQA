"""端到端集成测试：PDF解析→分块→向量化→Chroma索引→检索→Prompt→LLM生成"""
from main_part.pdf_parser import extract_text
from main_part.chunker import chunk_by_size
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.prompt_builder import build_rag_prompt
from main_part.generator import generate_answer

PDF_PATH = r'D:\桌面\I.pdf'
TEST_QUESTIONS = ['怎么退款', '什么是深度学习']

print('=' * 50)
print('DocQA 端到端集成测试')
print('=' * 50)

# Step 1: PDF解析
print('\n[1/6] 解析PDF...')
pages = extract_text(PDF_PATH)
print(f'  提取 {len(pages)} 页')

# Step 2: 分块
print('[2/6] 文本分块...')
chunks = chunk_by_size(pages)
print(f'  生成 {len(chunks)} 个chunk')

# Step 3: 向量化
print('[3/6] 加载embedding模型并向量化...')
embedder = Embedder()
chunks = embedder.embed_chunks(chunks)
print(f'  向量维度: {embedder.dim}')

# Step 4: Chroma索引
print('[4/6] 构建Chroma索引...')
retriever = Retriever()
retriever.index_chunks(chunks)
print(f'  索引条目: {retriever.collection.count()}')

# Step 5: 检索
for query in TEST_QUESTIONS:
    print(f'\n[5/6] 检索: "{query}"')
    results = retriever.search(query, embedder, top_k=3)
    for r in results:
        print(f'  [{r["score"]:.3f}] 第{r["source_page"]}页: {r["text"][:80]}...')

    # Step 6: Prompt构建 + LLM生成
    print(f'[6/6] 生成回答...')
    prompt = build_rag_prompt(query, results)
    answer = generate_answer(prompt)
    print(f'  回答: {answer[:200]}')
    print('-' * 50)

print('\n全部测试通过!')
