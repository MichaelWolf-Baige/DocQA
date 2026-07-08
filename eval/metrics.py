"""
检索层评估指标
==============
不需要 LLM judge，纯数值计算。
每个指标都有清晰的公式注释，方便理解原理。
"""

import numpy as np
from typing import List, Dict, Set


def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Recall@k = |top-k中相关的| / |所有相关的|

    衡量：检索是否"找全了"？
    示例：文档有 5 个相关 chunk，top-5 搜到了 3 个 → Recall@5 = 0.6
    """
    if len(relevant_ids) == 0:
        return 1.0  # 没有相关chunk则视为完美（避免除零）
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Precision@k = |top-k中相关的| / k

    衡量：检索是否"找对了"？（信噪比）
    示例：top-5 中有 3 个相关 → Precision@5 = 0.6
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / k


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """
    MRR (Mean Reciprocal Rank) = 1 / (第一个相关结果的排名)

    衡量：第一个正确答案排在第几位？
    示例：第一个相关结果排第 2 → RR = 1/2 = 0.5
          第一个相关结果排第 1 → RR = 1/1 = 1.0
          没找到任何相关 → RR = 0
    """
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    NDCG@k (Normalized Discounted Cumulative Gain)

    衡量：相关结果是否排在前面？（考虑排名位置的质量分）
    比 MRR 更精细：在 top-k 中有多个相关结果时，排前面的权重更高。
    """
    if len(relevant_ids) == 0:
        return 1.0

    # DCG: 每个相关结果贡献 1/log2(rank+1)
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved_ids[:k]):
        if chunk_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 因为 i 从 0 开始

    # IDCG: 理想排序下的 DCG（所有相关结果排在最前面）
    ideal_rel_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_rel_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """
    Hit Rate@k = top-k 中是否至少有一个相关结果？

    二元指标：找到了就是 1，没找到就是 0。
    适合衡量"用户至少能看到一个正确答案"的概率。
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def compute_all_metrics(
    questions: List[Dict],
    retriever,
    embedder,
    k_values: List[int] = None
) -> Dict:
    """
    对一个测试集计算所有检索指标。

    参数
    ----
    questions : List[Dict]
        测试用例列表，每个元素包含:
        - question: str
        - relevant_chunk_ids: List[int]
    retriever : Retriever
        已建好索引的检索器
    embedder : Embedder
        已加载的嵌入模型
    k_values : List[int]
        要评估的 k 值列表，默认 [3, 5, 10, 20]

    返回
    ----
    Dict : {
        "recall@3": float, "recall@5": float, ...,
        "precision@3": float, ...,
        "mrr": float,
        "ndcg@10": float,
        "hit_rate@5": float, ...,
        "per_question": [...]  每个题的详细结果
    }
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

        # 执行检索
        retrieved = retriever.search(question, embedder, top_k=max(k_values))
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved]

        # 计算各指标
        per_q = {"question": question, "relevant_count": len(relevant)}

        # Recall & Precision @k
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

        # MRR
        m = mrr(retrieved_ids, relevant)
        results["mrr"].append(m)
        per_q["mrr"] = round(m, 4)

        # 记录检索到的 chunk id 列表（方便调试）
        per_q["retrieved_ids"] = retrieved_ids[:10]
        per_q["retrieved_scores"] = [
            round(chunk["score"], 4) for chunk in retrieved[:10]
        ]

        results["per_question"].append(per_q)

    # 汇总：取平均
    summary = {}
    for metric_name, values in results.items():
        if metric_name == "per_question":
            continue
        summary[metric_name] = round(np.mean(values), 4)
    summary["per_question"] = results["per_question"]

    return summary
