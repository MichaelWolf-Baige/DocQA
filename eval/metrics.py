"""
检索层评估指标
==============
数学定义委托给 docqa.evaluation.metrics（单一真源），避免两处实现漂移。

本模块额外保留 compute_all_metrics：
  - 接收一个宽松签名的 search_fn（兼容 eval.adapters.dict_search_fn 的三参历史调用）
  - 返回项既可以是旧式 dict，也可以是新架构 Chunk 对象
"""
import numpy as np
from typing import List, Dict

# 单一真源：所有指标数学都来自 docqa 包，本文件不再重复实现
from docqa.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    hit_rate,
)


def compute_all_metrics(
    questions: List[Dict],
    search_fn,
    k_values: List[int] = None,
) -> Dict:
    """
    对一个测试集计算所有检索指标。

    参数
    ----
    questions : List[Dict]
        测试用例列表，每个元素包含:
        - question: str
        - relevant_chunk_ids: List[int]
    search_fn : Callable
        检索函数，签名宽松：
          search_fn(question, top_k)            -> 旧式 dict 列表（eval.adapters.dict_search_fn 包出来的）
          search_fn(question, embedder, top_k)  -> 兼容历史三参调用
        返回项需是带 chunk_id 与 score 字段的对象/dict。
    k_values : List[int]
        要评估的 k 值列表，默认 [3, 5, 10, 20]

    返回
    ----
    Dict : {recall@k, precision@k, hit_rate@k, ndcg@k, mrr, per_question}
    """
    if k_values is None:
        k_values = [3, 5, 10, 20]

    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"precision@{k}": [] for k in k_values})
    results.update({f"hit_rate@{k}": [] for k in k_values})
    results.update({f"ndcg@{k}": [] for k in k_values})
    results["mrr"] = []
    results["per_question"] = []

    for q in questions:
        question = q["question"]
        relevant = set(q["relevant_chunk_ids"])

        retrieved = search_fn(question, top_k=max(k_values))
        retrieved_ids = [_get(chunk, "chunk_id") for chunk in retrieved]

        per_q = {"question": question, "relevant_count": len(relevant)}
        for k in k_values:
            r = recall_at_k(retrieved_ids, relevant, k)
            p = precision_at_k(retrieved_ids, relevant, k)
            h = hit_rate(retrieved_ids, relevant, k)
            n = ndcg_at_k(retrieved_ids, relevant, k)
            results[f"recall@{k}"].append(r)
            results[f"precision@{k}"].append(p)
            results[f"hit_rate@{k}"].append(h)
            results[f"ndcg@{k}"].append(n)
            per_q[f"recall@{k}"] = round(r, 4)
            per_q[f"precision@{k}"] = round(p, 4)
            per_q[f"hit_rate@{k}"] = round(h, 4)

        m = mrr(retrieved_ids, relevant)
        results["mrr"].append(m)
        per_q["mrr"] = round(m, 4)

        per_q["retrieved_ids"] = retrieved_ids[:10]
        per_q["retrieved_scores"] = [
            round(_get(chunk, "score"), 4) for chunk in retrieved[:10]
        ]
        results["per_question"].append(per_q)

    summary = {}
    for metric_name, values in results.items():
        if metric_name == "per_question":
            continue
        summary[metric_name] = round(np.mean(values), 4)
    summary["per_question"] = results["per_question"]
    return summary


def _get(chunk, key, default=0):
    """兼容 dict（旧式）与对象（Chunk）两种取值方式。"""
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default if key == "score" else None)