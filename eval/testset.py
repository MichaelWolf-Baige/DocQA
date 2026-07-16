"""
测试集管理
==========
定义测试集的 JSONL 格式，提供加载、验证、统计、示例生成功能。

数据格式：
{
    "id": "q001",
    "question": "退团需要提前几天通知？",
    "question_type": "fact_lookup",       // fact_lookup | summary | exact_match | boundary
    "ground_truth": "提前7天通知旅行社。",
    "relevant_chunk_ids": [4, 5],         // 包含答案的 chunk id 列表
    "relevant_page_numbers": [2, 3],      // 可选：方便人工对照
    "difficulty": "easy",                 // easy | medium | hard
    "note": "出自退团条款部分"              // 可选：标注来源，帮助后续维护
}
"""

import json
import os
from typing import List, Dict, Optional

from .config import TEST_DATA_DIR, QUESTION_TYPE_DIST


# --- 测试集加载 ---

def load_testset(filepath: Optional[str] = None) -> List[Dict]:
    """
    从 JSONL 文件加载测试集。

    参数
    ----
    filepath : str, optional
        JSONL 文件路径，默认 eval/data/test_questions.jsonl

    返回
    ----
    List[Dict] : 测试用例列表，按 id 排序
    """
    if filepath is None:
        filepath = os.path.join(TEST_DATA_DIR, "test_questions.jsonl")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"测试集文件不存在: {filepath}\n"
            f"请先运行 create_sample_testset() 创建示例测试集，"
            f"或手动编写测试用例。"
        )

    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            q = json.loads(line)
            _validate_question(q)
            questions.append(q)

    return sorted(questions, key=lambda x: x["id"])


def save_testset(questions: List[Dict], filepath: Optional[str] = None) -> None:
    """保存测试集到 JSONL 文件。"""
    if filepath is None:
        filepath = os.path.join(TEST_DATA_DIR, "test_questions.jsonl")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"测试集已保存: {filepath} ({len(questions)} 题)")


# --- 验证 ---

def _validate_question(q: Dict) -> None:
    """验证单个测试用例的字段完整性。"""
    required = ["id", "question", "question_type", "relevant_chunk_ids"]
    for field in required:
        if field not in q:
            raise ValueError(f"测试用例缺少必填字段: {field} (id={q.get('id', '?')})")

    valid_types = {"fact_lookup", "summary", "exact_match", "boundary"}
    if q["question_type"] not in valid_types:
        raise ValueError(
            f"无效的 question_type: {q['question_type']} (id={q['id']}), "
            f"必须是 {valid_types} 之一"
        )

    if not isinstance(q["relevant_chunk_ids"], list):
        raise ValueError(
            f"relevant_chunk_ids 必须是列表 (id={q['id']})"
        )


# --- 统计 ---

def compute_stats(questions: List[Dict]) -> Dict:
    """计算测试集的统计信息。"""
    stats = {
        "total": len(questions),
        "by_type": {},
        "by_difficulty": {},
        "avg_relevant_chunks": 0.0,
        "boundary_count": 0,  # 拒答题数量
    }

    total_chunks = 0
    for q in questions:
        # 题型分布
        qtype = q.get("question_type", "unknown")
        stats["by_type"][qtype] = stats["by_type"].get(qtype, 0) + 1

        # 难度分布
        diff = q.get("difficulty", "unknown")
        stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        # 相关 chunk 数
        total_chunks += len(q["relevant_chunk_ids"])

        # 拒答题
        if q.get("is_unanswerable", False):
            stats["boundary_count"] += 1

    if stats["total"] > 0:
        stats["avg_relevant_chunks"] = round(total_chunks / stats["total"], 2)

    return stats


def print_stats(questions: List[Dict]) -> None:
    """打印测试集统计报告。"""
    stats = compute_stats(questions)
    print("\n" + "=" * 50)
    print("测试集统计")
    print("=" * 50)
    print(f"总题数: {stats['total']}")

    print("\n按题型分布:")
    for qtype, count in sorted(stats["by_type"].items()):
        pct = count / stats["total"] * 100
        print(f"  {qtype}: {count} ({pct:.1f}%)")

    print("\n按难度分布:")
    for diff, count in sorted(stats["by_difficulty"].items()):
        pct = count / stats["total"] * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")

    print(f"\n平均每题相关chunk数: {stats['avg_relevant_chunks']}")
    print(f"拒答/边界题: {stats['boundary_count']}")
    if stats["boundary_count"] < stats["total"] * 0.1:
        print("  [WARN] 拒答题占比 < 10%，建议增加到 10-20%")
    print("=" * 50 + "\n")


# --- 辅助：根据检索结果辅助标注 chunk id ---

def search_and_show_chunks(
    question: str, pipeline, top_k: int = 15
) -> List[Dict]:
    """
    辅助工具：给定一个问题，返回检索结果，帮助人工判断哪些 chunk 是相关的。

    用法（在 Python 交互环境中）:
        >>> from eval.testset import search_and_show_chunks
        >>> from eval.adapters import build_eval_pipeline
        >>> pipeline = build_eval_pipeline()
        >>> results = search_and_show_chunks("退团怎么退？", pipeline)
        >>> # 人工浏览后，记录 relevant_chunk_ids = [3, 5, 7]

    参数
    ----
    question : 问句
    pipeline : DocQAPipeline
        已 ingest 好的 pipeline（新架构）
    top_k : 返回结果数

    返回
    ----
    List[Dict] : 旧式 dict 列表（含 chunk_id / source_page / score / text）
    """
    from eval.adapters import dict_search_fn
    search = dict_search_fn(pipeline)
    results = search(question, top_k)
    print(f"\n问题: {question}")
    print(f"检索到 {len(results)} 个结果:\n")
    for i, chunk in enumerate(results):
        marker = f"[{i+1}]"
        print(f"{marker} chunk_id={chunk['chunk_id']} | "
              f"第{chunk['source_page']}页 | score={chunk['score']:.4f}")
        text = chunk["text"][:200].replace("\n", " ")
        print(f"    {text}...")
        print()
    return results
