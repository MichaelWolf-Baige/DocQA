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
import numpy as np

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.testset import load_testset
from eval.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate
from eval.adapters import build_eval_pipeline, _chunk_to_dict

from docqa.retrieval.dense import DenseRetriever
from docqa.retrieval.bm25 import BM25Retriever
from docqa.retrieval.hybrid import HybridRetriever
from docqa.retrieval.reranker import BGEReranker

TOP_K_VALUES = [3, 5, 10, 20]


def _ids(chunks):
    """从 Chunk 列表取 chunk_id。"""
    return [c.chunk_id for c in chunks]


def evaluate_method(name, questions, search_fn, k_values):
    """评估单种检索方案。search_fn(query, top_k) -> List[Chunk]"""
    metrics = {f'recall@{k}': [] for k in k_values}
    metrics.update({f'precision@{k}': [] for k in k_values})
    metrics.update({f'hit_rate@{k}': [] for k in k_values})
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    metrics['mrr'] = []
    max_k = max(k_values)

    for q in questions:
        relevant = set(q['relevant_chunk_ids'])
        results = search_fn(q['question'], max_k)
        retrieved_ids = _ids(results)
        for k in k_values:
            metrics[f'recall@{k}'].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[f'precision@{k}'].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[f'hit_rate@{k}'].append(hit_rate(retrieved_ids, relevant, k))
            metrics[f'ndcg@{k}'].append(ndcg_at_k(retrieved_ids, relevant, k))
        metrics['mrr'].append(mrr(retrieved_ids, relevant))

    summary = {}
    for key, vals in metrics.items():
        summary[key] = round(float(np.mean(vals)), 4)

    print(f'\n{"="*50}')
    print(f'【{name}】')
    print(f'{"="*50}')
    for k in k_values:
        print(f'Recall@{k:>2}:  {summary[f"recall@{k}"]:.2%}  |  '
              f'Precision@{k:>2}: {summary[f"precision@{k}"]:.2%}  |  '
              f'Hit@{k:>2}: {summary[f"hit_rate@{k}"]:.2%}  |  '
              f'NDCG@{k:>2}: {summary[f"ndcg@{k}"]:.4f}')
    print(f'MRR:      {summary["mrr"]:.4f}')

    low = [q for i, q in enumerate(questions)
           if metrics['recall@10'][i] < 0.5]
    if low:
        print(f'Recall@10 < 50%: {len(low)} 题')
        for q in low:
            print(f'  {q["id"]}: {q["question"][:50]}...')

    return summary


def main():
    start = time.time()
    questions = load_testset()
    retrievable = [q for q in questions if len(q['relevant_chunk_ids']) > 0]
    print(f'可检索题: {len(retrievable)}')

    # 用新架构构建摄入好的 pipeline（含 ingest）
    pipeline = build_eval_pipeline(rebuild_index=False)
    vector_store = pipeline.vector_store
    embedder = pipeline.embedder
    all_chunks = pipeline.vector_store.get_all_chunks()
    print(f'索引就绪: {len(all_chunks)} chunks')

    # 各方案独立构建 retriever
    dense = DenseRetriever(vector_store, embedder)
    bm25 = BM25Retriever()
    bm25.build_index(all_chunks)
    hybrid = HybridRetriever(vector_store, embedder, rrf_k=60)
    hybrid.build_bm25_index(all_chunks)
    reranker = BGEReranker()

    # ====== 方案 1: 纯向量 ======
    def vector_search(query, top_k):
        return dense.search(query, top_k=top_k)
    vec_metrics = evaluate_method('1-纯向量(Baseline)', retrievable, vector_search, TOP_K_VALUES)

    # ====== 方案 2: 纯 BM25 ======
    def bm25_search(query, top_k):
        return bm25.search(query, top_k=top_k)
    bm25_metrics = evaluate_method('2-纯BM25', retrievable, bm25_search, TOP_K_VALUES)

    # ====== 方案 3: 混合检索 ======
    def hybrid_search(query, top_k):
        return hybrid.search(query, top_k=top_k)
    hybrid_metrics = evaluate_method('3-混合检索(BM25+向量)', retrievable, hybrid_search, TOP_K_VALUES)

    # ====== 方案 4: 混合 + 重排序 ======
    def hybrid_rerank_search(query, top_k):
        candidates = hybrid.search(query, top_k=50)
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
    header_line = f'{"指标":<20}' + ''.join(f'{m[0][:20]:>20}' for m in all_methods)
    print(header_line)
    print('-' * len(header_line))
    for metric in metrics_to_show:
        vals = [m[1].get(metric, 0) for m in all_methods]
        best = max(vals)
        row = f'{metric:<20}'
        for v in vals:
            marker = ' ★' if v == best else ''
            if 'mrr' in metric or 'ndcg' in metric:
                row += f'{v:.4f}{marker:>15}'
            else:
                row += f'{v:.2%}{marker:>15}'
        print(row)

    # ====== 逐题对比 (Recall@10) ======
    print(f'\n\n--- 逐题 Recall@10 对比 ---')
    print(f'{"ID":<6} {"问题":<35} {"向量":>6} {"BM25":>6} {"混合":>6} {"混合+重排":>8}')
    print('-' * 70)
    for q in retrievable:
        relevant = set(q['relevant_chunk_ids'])
        recs = []
        for fn in [vector_search, bm25_search, hybrid_search, hybrid_rerank_search]:
            ids = _ids(fn(q['question'], 10))
            recs.append(recall_at_k(ids, relevant, 10))
        vec_r, _, _, rerank_r = recs
        flag = ''
        if vec_r == 0 and rerank_r > 0:
            flag = ' ← 从0找回!'
        elif rerank_r > vec_r + 0.2:
            flag = ' ↑'
        print(f'{q["id"]:<6} {q["question"][:35]:<35} {recs[0]:>6.0%} {recs[1]:>6.0%} {recs[2]:>6.0%} {recs[3]:>8.0%}{flag}')

    print(f'\n总耗时: {(time.time() - start)/60:.1f} 分钟')


if __name__ == '__main__':
    main()