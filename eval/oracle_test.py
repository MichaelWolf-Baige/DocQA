"""
Oracle 瓶颈分析
================
通过四个对照条件精确定位 RAG 系统的瓶颈在检索层还是生成层。

实验设计（四条件）：
  A (actual):  正常检索 → 生成       （当前系统表现）
  B (oracle):  完美检索 → 生成       （gold chunk 直接给模型，绕过检索）
  C (bare):    无上下文 → 生成       （裸模型，测模型本身的知识）
  D (noise):   随机chunk → 生成      （错误上下文对模型的干扰程度）

瓶颈诊断公式：
  检索瓶颈 =  B_correct - A_correct   （完美检索能提升多少？越大→检索层是瓶颈）
  生成瓶颈 =  1 - B_correct          （即使完美检索也无法答对的比例，越大→生成层是瓶颈）
  知识盲区 =  1 - C_correct          （模型本身不知道答案的比例）
  检索干扰 =  C_correct - D_correct  （错误检索对模型已有知识的破坏程度）
  检索增益 =  A_correct - C_correct  （添加检索后比裸模型提升了多少）
"""

import random
from typing import List, Dict, Optional, Callable

from .config import RANDOM_SEED, DEFAULT_TOP_K


def run_oracle_analysis(
    questions: List[Dict],
    pipeline,
    generator: Callable[[str], str],
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = True,
) -> Dict:
    """
    运行四条件 Oracle 瓶颈分析。

    参数
    ----
    questions : List[Dict]
        测试用例，每个需要 question, relevant_chunk_ids, 可选 ground_truth
    pipeline : DocQAPipeline
        已 ingest 好的 pipeline（提供 retrieve / vector_store）
    generator : Callable
        生成函数，签名为 generator(prompt: str) -> str
    top_k : int
        检索 top_k
    verbose : bool
        是否打印每题细节

    返回
    ----
    Dict : 包含每个条件的详细结果和聚合诊断
    """
    random.seed(RANDOM_SEED)
    from eval.adapters import build_rag_prompt, get_chunk_by_id, get_all_chunks

    all_chunks = get_all_chunks(pipeline)
    all_chunk_ids = [c['chunk_id'] for c in all_chunks]
    cid_to_chunk = {c['chunk_id']: c for c in all_chunks}

    if not all_chunk_ids:
        raise RuntimeError("向量库为空，请先建好索引")

    results = {
        "conditions": {"actual": [], "oracle": [], "bare": [], "noise": []},
        "per_question": [],
    }

    for i, q in enumerate(questions):
        question = q["question"]
        relevant_ids = set(q["relevant_chunk_ids"])
        ground_truth = q.get("ground_truth", "")

        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {question[:60]}...")

        # --- 条件 A：正常检索 ---
        retrieved = pipeline.retrieve(question, top_k=top_k)
        prompt_a = build_rag_prompt(question, retrieved)
        answer_a = generator(prompt_a)
        results["conditions"]["actual"].append(answer_a)

        # --- 条件 B：Oracle 检索（直接注入 gold chunks）---
        oracle_chunks = [cid_to_chunk[cid] for cid in relevant_ids if cid in cid_to_chunk]
        if oracle_chunks:
            prompt_b = build_rag_prompt(question, oracle_chunks)
            answer_b = generator(prompt_b)
        else:
            prompt_b, answer_b = None, "[ERROR: gold chunks not found]"
        results["conditions"]["oracle"].append(answer_b)

        # --- 条件 C：裸模型 ---
        prompt_c = _build_bare_prompt(question)
        answer_c = generator(prompt_c)
        results["conditions"]["bare"].append(answer_c)

        # --- 条件 D：随机干扰 ---
        noise_chunks = _pick_noise(relevant_ids, all_chunk_ids, cid_to_chunk, top_k)
        if noise_chunks:
            prompt_d = build_rag_prompt(question, noise_chunks)
        else:
            prompt_d = prompt_c   # fallback：没有可干扰 chunk 就降级为裸模型
        answer_d = generator(prompt_d)
        results["conditions"]["noise"].append(answer_d)

        # 单题记录
        per_q = {
            "id": q["id"],
            "question": question,
            "ground_truth": ground_truth,
            "relevant_chunk_ids": list(relevant_ids),
            "answers": {
                "actual": answer_a,
                "oracle": answer_b,
                "bare": answer_c,
                "noise": answer_d,
            },
            "actual_retrieved_ids": [c.chunk_id for c in retrieved],
        }
        results["per_question"].append(per_q)

        if verbose:
            print(f"  actual: {answer_a[:80]}...")
            print(f"  oracle: {answer_b[:80]}...")
            print(f"  bare:   {answer_c[:80]}...")
            print(f"  noise:  {answer_d[:80]}...")

    results["diagnosis"] = _summarize_conditions(results)
    return results


def _pick_noise(relevant_ids, all_chunk_ids, cid_to_chunk, top_k):
    """条件 D：从非相关 chunk 中随机采样 top_k 个作为干扰上下文。"""
    irrelevant = [cid for cid in all_chunk_ids if cid not in relevant_ids]
    if not irrelevant:
        return []
    if len(irrelevant) < top_k:
        selected = irrelevant
    else:
        selected = random.sample(irrelevant, top_k)
    return [cid_to_chunk[cid] for cid in selected if cid in cid_to_chunk]


def _build_bare_prompt(question):
    """条件 C：裸模型——不给任何文档上下文。"""
    return (
        '请回答以下问题。如果不知道答案，请直接说"不知道"。\n\n'
        f"问题：{question}\n"
        f"答案："
    )


def _summarize_conditions(results: Dict) -> Dict:
    """生成诊断摘要（不含正确率判断——正确率需要人工或 judge 模型评估）。"""
    n = len(results["per_question"])
    avg_len = {}
    for cond in ["actual", "oracle", "bare", "noise"]:
        lengths = [len(a) for a in results["conditions"][cond] if a]
        avg_len[cond] = round(sum(lengths) / max(len(lengths), 1), 1)

    return {
        "total_questions": n,
        "avg_answer_length": avg_len,
        "interpretation": {
            "actual_vs_oracle": (
                "如果 Oracle 答案明显优于 Actual，说明检索是瓶颈。\n"
                "如果两者差不多，说明即使给正确答案模型也用不好——瓶颈在生成层。"
            ),
            "actual_vs_bare": (
                "如果 Actual 明显优于 Bare，说明检索确实提供了有用信息。\n"
                "如果两者差不多，说明检索基本没用——需要优化检索器。"
            ),
            "bare_vs_noise": (
                "如果 Noise 比 Bare 差很多，说明模型容易被错误上下文干扰。\n"
                "这是小模型的常见问题，可以通过 Prompt 优化（强调'不确定就说不知道'）缓解。"
            ),
            "avg_length_comparison": (
                f"各条件平均答案长度: {avg_len}。\n"
                "Oracle 长度明显大于 Actual → 检索召回不足，模型缺少信息。\n"
                "Noise 长度明显大于 Bare → 模型在错误信息上过度发挥（幻觉风险）。"
            ),
        },
    }


def print_oracle_report(results: Dict) -> None:
    """打印 Oracle 分析的格式化报告。"""
    diag = results["diagnosis"]
    print("\n" + "=" * 60)
    print("Oracle 瓶颈分析报告")
    print("=" * 60)
    print(f"测试题目数: {diag['total_questions']}")
    print(f"\n平均答案长度:")
    for cond, length in diag["avg_answer_length"].items():
        label = {"actual": "A-实际检索", "oracle": "B-Oracle", "bare": "C-裸模型", "noise": "D-随机干扰"}
        print(f"  {label.get(cond, cond)}: {length} 字符")

    print(f"\n📊 诊断解读:")
    for key, text in diag["interpretation"].items():
        if key == "avg_length_comparison":
            continue
        print(f"\n  [{key}]")
        print(f"  {text}")

    print("\n" + "=" * 60)
    print("⚠️ 以上为机器自动分析。精确的正确率判断需要人工评估或 LLM judge。")
    print("建议：对比 Actual vs Oracle 的前 5 题答案，人眼判断差距。")
    print("=" * 60 + "\n")