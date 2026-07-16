"""端到端集成测试 + top_k 评分分布分析

新架构下用 DocQAPipeline 一站式跑：摄入 → 检索 → 生成。
所有组件通过 docqa/config.yaml 配置，不直接依赖老 main_part。
"""
import os
from docqa.pipeline import DocQAPipeline

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploaded.pdf')
TEST_QUERIES = ['怎么退款', '什么是深度学习']

print('=' * 50)
print('DocQA 端到端测试（新架构 docqa.pipeline）')
print('=' * 50)

pipeline = DocQAPipeline.from_config()

# 摄入文档（已有索引则复用）
if pipeline.vector_store.count() == 0:
    print(f'\n[摄入] {PDF_PATH}')
    n = pipeline.ingest(PDF_PATH, clear=True)
    print(f'  索引 {n} 个 chunk')
else:
    print(f'\n[复用已有索引] {pipeline.vector_store.count()} 条')

for query in TEST_QUERIES:
    print(f'\n{"=" * 50}')
    print(f'查询: "{query}"')
    print('=' * 50)

    chunks = pipeline.retrieve(query, top_k=10)

    print('\n--- 评分分布（帮助判断最优top_k）---')
    for i, c in enumerate(chunks):
        score = c.metadata.get('rrf_score', c.metadata.get('rerank_score', c.metadata.get('score', 0.0)))
        bar = '█' * int(score * 20)
        print(f'  {i+1:2d}. [{score:.4f}] {bar} 第{c.source_page}页')
        print(f'      预览: {c.text[:100]}...')

    print(f'\n--- LLM生成（使用 top-5）---')
    answer = pipeline.ask(query)
    print(f'  回答: {answer}')
    print()

print('=' * 50)
print('top_k 判断方法:')
print('  - 观察分数曲线，找到自然下跌点（例如从0.5跌到0.3的位置）')
print('  - 分数 > 0.5 = 强相关，0.3-0.5 = 弱相关，< 0.3 = 噪音')
print('=' * 50)