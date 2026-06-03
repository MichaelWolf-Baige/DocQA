"""端到端集成测试 + top_k 评分分布分析"""
from main_part.pdf_parser import extract_text
from main_part.chunker import chunk_by_size
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.prompt_builder import build_rag_prompt
from main_part.generator import generate_answer

PDF_PATH = 'uploaded.pdf'
TEST_QUERIES = ['怎么退款', '什么是深度学习']

print('=' * 50)
print('DocQA 端到端测试')
print('=' * 50)

# Step 1-4: 建索引（只做一次）
print('\n[1/4] 解析PDF...')
pages = extract_text(PDF_PATH)
print(f'  提取 {len(pages)} 页')

print('[2/4] 文本分块...')
chunks = chunk_by_size(pages)
print(f'  生成 {len(chunks)} 个chunk')

print('[3/4] 加载embedding模型并向量化...')
embedder = Embedder()
chunks = embedder.embed_chunks(chunks)
print(f'  向量维度: {embedder.dim}')

print('[4/4] 构建Chroma索引...')
retriever = Retriever()
retriever.index_chunks(chunks)
print(f'  索引条目: {retriever.collection.count()}')

# Step 5: 评分分布分析 + 问答
for query in TEST_QUERIES:
    print(f'\n{"=" * 50}')
    print(f'查询: "{query}"')
    print('=' * 50)

    # 多取一些chunk看分数分布
    results = retriever.search(query, embedder, top_k=10)

    print('\n--- 评分分布（帮助判断最优top_k）---')
    for i, r in enumerate(results):
        bar = '█' * int(r['score'] * 20)
        print(f'  {i+1:2d}. [{r["score"]:.4f}] {bar} 第{r["source_page"]}页')
        print(f'      预览: {r["text"][:100]}...')

    # 取前5个有效chunk做问答
    print(f'\n--- LLM生成（使用top-5）---')
    prompt = build_rag_prompt(query, results[:5])
    answer = generate_answer(prompt)
    print(f'  回答: {answer}')
    print()

print('=' * 50)
print('top_k 判断方法:')
print('  - 观察分数曲线，找到自然下跌点（例如从0.5跌到0.3的位置）')
print('  - 分数 > 0.5 = 强相关，0.3-0.5 = 弱相关，< 0.3 = 噪音')
print('  - 选强相关到弱相关的拐点作为 top_k')
print('  - 本模型47个chunk，top_k=10约占21%，合理')
print('=' * 50)
