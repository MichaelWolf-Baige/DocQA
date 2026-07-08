"""
检索评估指标
============
纯数学计算，不依赖 LLM。
"""
import numpy as np
from typing import List, Dict, Set


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if len(relevant_ids) == 0:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / k


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if len(relevant_ids) == 0:
        return 1.0
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, cid in enumerate(retrieved_ids[:k])
        if cid in relevant_ids
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def hit_rate(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def evaluate_retrieval(
    questions: List[Dict],
    search_fn,
    k_values: List[int] = None,
) -> Dict:
    """对一个检索函数进行全面评估"""
    if k_values is None:
        k_values = [3, 5, 10, 20]

    results = {f'recall@{k}': [] for k in k_values}
    results.update({f'precision@{k}': [] for k in k_values})
    results.update({f'hit_rate@{k}': [] for k in k_values})
    results.update({f'ndcg@{k}': [] for k in k_values})
    results['mrr'] = []
    results['per_question'] = []

    for q in questions:
        relevant = set(q['relevant_chunk_ids'])
        chunks = search_fn(q['question'], max(k_values))
        retrieved_ids = [c.chunk_id for c in chunks]

        per_q = {'question': q['question'], 'relevant_count': len(relevant)}
        for k in k_values:
            per_q[f'recall@{k}'] = round(recall_at_k(retrieved_ids, relevant, k), 4)
            per_q[f'precision@{k}'] = round(precision_at_k(retrieved_ids, relevant, k), 4)
            per_q[f'hit_rate@{k}'] = round(hit_rate(retrieved_ids, relevant, k), 4)
            per_q[f'ndcg@{k}'] = round(ndcg_at_k(retrieved_ids, relevant, k), 4)

            results[f'recall@{k}'].append(per_q[f'recall@{k}'])
            results[f'precision@{k}'].append(per_q[f'precision@{k}'])
            results[f'hit_rate@{k}'].append(per_q[f'hit_rate@{k}'])
            results[f'ndcg@{k}'].append(per_q[f'ndcg@{k}'])

        m = mrr(retrieved_ids, relevant)
        results['mrr'].append(m)
        per_q['mrr'] = round(m, 4)
        per_q['retrieved_ids'] = retrieved_ids[:10]
        results['per_question'].append(per_q)

    summary = {}
    for key, vals in results.items():
        if key != 'per_question':
            summary[key] = round(float(np.mean(vals)), 4)
    summary['per_question'] = results['per_question']
    return summary
