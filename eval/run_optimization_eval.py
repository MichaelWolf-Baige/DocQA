"""
优化前后对比评估
================
对比四种检索方案的指标：
  1. 纯向量 (baseline)     → 当前系统
  2. 纯 BM25               → 关键词检索
  3. 混合检索 (BM25+向量)   → RRF 融合
  4. 混合检索 + 重排序      → 完整优化方案
"""

import os, sys, time
from collections import defaultdict
import numpy as np

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.testset import load_testset
from eval.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate
from main_part.pdf_parser import extract_text
from main_part.chunker import chunk_by_size
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.hybrid_retriever import HybridRetriever
from main_part.reranker import Reranker

TOP_K_VALUES = [3, 5, 10, 20]


def evaluate_method(name, questions, search_fn, k_values):
    """评估单种检索方案。search_fn(query, top_k) -> [{chunk_id, ...}, ...]"""
    metrics = {f'recall@{k}': [] for k in k_values}
    metrics.update({f'precision@{k}': [] for k in k_values})
    metrics.update({f'hit_rate@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics['mrr'] = []

    max_k = max(k_values)

    for q in questions:
        relevant = set(q['relevant_chunk_ids'])
        results = search_fn(q['question'], max_k)
        retrieved_ids = [r['chunk_id'] for r in results]

        for k in k_values:
            metrics[f'recall@{k}'].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[f'precision@{k}'].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[f'hit_rate@{k}'].append(hit_rate(retrieved_ids, relevant, k))
            metrics[f'ndcg@{k}'].append(ndcg_at_k(retrieved_ids, relevant, k))

        metrics['mrr'].append(mrr(retrieved_ids, relevant))

    # 汇总
    summary = {}
    for key, vals in metrics.items():
        summary[key] = round(float(np.mean(vals)), 4)

    # 打印
    print(f'\n{"="*50}')
    print(f'【{name}】')
    print(f'{"="*50}')
    for k in k_values:
        print(f'Recall@{k:>2}:  {summary[f"recall@{k}"]:.2%}  |  '
              f'Precision@{k:>2}: {summary[f"precision@{k}"]:.2%}  |  '
              f'Hit@{k:>2}: {summary[f"hit_rate@{k}"]:.2%}  |  '
              f'NDCG@{k:>2}: {summary[f"ndcg@{k}"]:.4f}')
    print(f'MRR:      {summary["mrr"]:.4f}')

    # 低分题
    low = [q for i, q in enumerate(questions)
           if metrics['recall@10'][i] < 0.5]
    if low:
        print(f'Recall@10 < 50%: {len(low)} 题')
        for q in low:
            print(f'  {q["id"]}: {q["question"][:50]}...')

    return summary


def main():
    start = time.time()

    # 加载测试集
    questions = load_testset()
    retrievable = [q for q in questions if len(q['relevant_chunk_ids']) > 0]
    print(f'可检索题: {len(retrievable)}')

    # 构建索引
    print('构建索引...')
    pages = extract_text('uploaded.pdf')
    chunks = chunk_by_size(pages)
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    retriever = Retriever()
    retriever.index_chunks(chunks)

    # BM25 索引
    hr = HybridRetriever()
    hr.build_bm25_index(chunks)

    # Reranker
    reranker = Reranker()

    # ====== 方案 1: 纯向量 (baseline) ======
    def vector_search(query, top_k):
        return retriever.search(query, embedder, top_k=top_k)

    vec_metrics = evaluate_method('1-纯向量(Baseline)', retrievable, vector_search, TOP_K_VALUES)

    # ====== 方案 2: 纯 BM25 ======
    def bm25_search(query, top_k):
        results = hr.search_bm25(query, top_k=top_k)
        # 回填 text 和 source_page
        all_data = retriever.collection.get()
        id_to_data = {}
        for i, cid in enumerate(all_data['ids']):
            id_to_data[int(cid)] = {
                'text': all_data['documents'][i],
                'source_page': all_data['metadatas'][i].get('source_page', -1),
                'score': 0,
            }
        filled = []
        for r in results:
            data = id_to_data.get(r['chunk_id'], {})
            filled.append({
                'chunk_id': r['chunk_id'],
                'text': data.get('text', ''),
                'source_page': data.get('source_page', -1),
                'score': r['score'],
            })
        return filled

    bm25_metrics = evaluate_method('2-纯BM25', retrievable, bm25_search, TOP_K_VALUES)

    # ====== 方案 3: 混合检索 ======
    def hybrid_search(query, top_k):
        return hr.search_hybrid(query, retriever, embedder, top_k=top_k)

    hybrid_metrics = evaluate_method('3-混合检索(BM25+向量)', retrievable, hybrid_search, TOP_K_VALUES)

    # ====== 方案 4: 混合 + 重排序 ======
    def hybrid_rerank_search(query, top_k):
        # 混合检索取 top-50
        candidates = hr.search_hybrid(query, retriever, embedder, top_k=50)
        # 重排序取 top_k
        return reranker.rerank(query, candidates, top_k=top_k)

    rerank_metrics = evaluate_method('4-混合检索+重排序', retrievable, hybrid_rerank_search, TOP_K_VALUES)

    # ====== 汇总对比表 ======
    all_methods = [
        ('1-纯向量(Baseline)', vec_metrics),
        ('2-纯BM25', bm25_metrics),
        ('3-混合检索', hybrid_metrics),
        ('4-混合+重排序', rerank_metrics),
    ]

    print(f'\n\n{"="*80}')
    print('对比汇总')
    print(f'{"="*80}')

    metrics_to_show = [f'recall@{k}' for k in TOP_K_VALUES] + [f'precision@{k}' for k in TOP_K_VALUES] + ['mrr']
    headers = ['指标'] + [m[0][:20] for m in all_methods]
    header_line = f'{"指标":<20}' + ''.join(f'{m[0][:20]:>20}' for m in all_methods)
    print(header_line)
    print('-' * len(header_line))

    for metric in metrics_to_show:
        vals = [m[1].get(metric, 0) for m in all_methods]
        best = max(vals) if 'recall' in metric or 'precision' in metric or 'mrr' in metric else max(vals)
        row = f'{metric:<20}'
        for v in vals:
            if 'mrr' in metric or 'ndcg' in metric:
                marker = ' ★' if v == best else ''
                row += f'{v:.4f}{marker:>15}'
            else:
                marker = ' ★' if v == best else ''
                row += f'{v:.2%}{marker:>15}'
        print(row)

    # ====== 逐题对比 (Recall@10) ======
    print(f'\n\n--- 逐题 Recall@10 对比 ---')
    print(f'{"ID":<6} {"问题":<35} {"向量":>6} {"BM25":>6} {"混合":>6} {"混合+重排":>8}')
    print('-' * 70)

    for i, q in enumerate(retrievable):
        relevant = set(q['relevant_chunk_ids'])
        recs = []
        for name, search_fn in [
            ('vector', vector_search),
            ('bm25', bm25_search),
            ('hybrid', hybrid_search),
            ('rerank', hybrid_rerank_search),
        ]:
            results = search_fn(q['question'], 10)
            ids = [r['chunk_id'] for r in results]
            recs.append(recall_at_k(ids, relevant, 10))

        # 标记改进的题
        vec_r = recs[0]
        rerank_r = recs[3]
        flag = ''
        if vec_r == 0 and rerank_r > 0:
            flag = ' ← 从0找回!'
        elif rerank_r > vec_r + 0.2:
            flag = ' ↑'

        print(f'{q["id"]:<6} {q["question"][:35]:<35} {recs[0]:>6.0%} {recs[1]:>6.0%} {recs[2]:>6.0%} {recs[3]:>8.0%}{flag}')

    elapsed = time.time() - start
    print(f'\n总耗时: {elapsed/60:.1f} 分钟')


if __name__ == '__main__':
    main()
