"""
RAGAS 评估模块
==============
使用 RAGAS 框架对 DocQA 系统进行四维评估：
  - Faithfulness: 答案是否有上下文依据（检测幻觉）
  - Answer Relevancy: 答案是否切题
  - Context Precision: 检索结果的信噪比
  - Context Recall: 检索是否覆盖了 ground truth 所需信息

Judge 模型要求：必须比被测模型更强，推荐 7B+。
当前可用：qwen3.5:4b（最小可用）或 qwen3.5:9b（推荐）。

注意：对 1.5B 生成模型的评估结果只是基线参考值，
      因为 judge（4B/9B）和被测模型（1.5B）不是一个体系。
"""

import os
import sys
import json
from typing import List, Dict, Optional

from .config import (
    OLLAMA_BASE, JUDGE_MODEL, GENERATOR_MODEL
)


def check_ragas_installed() -> bool:
    """检查 RAGAS 是否已安装。"""
    try:
        import ragas
        return True
    except ImportError:
        return False


def install_ragas_instructions():
    """RAGAS 安装说明。"""
    return (
        "RAGAS 未安装。请运行:\n"
        "  pip install ragas>=0.2.4 datasets langchain-openai\n"
        "注意: RAGAS v0.3+ 移除了 context_relevancy，"
        "已替换为 context_precision + context_recall。"
    )


def build_eval_dataset(
    questions: List[Dict],
    retriever,
    embedder,
    generator,
    top_k: int = 10,
) -> List[Dict]:
    """
    构建 RAGAS 评估所需的 Dataset。

    对每个测试用例：
    1. 用实际检索获取 contexts
    2. 用生成模型获取 answer
    3. 组装为 {question, answer, contexts, ground_truth}

    参数
    ----
    questions : List[Dict]
        测试用例（需包含 ground_truth 字段）
    retriever, embedder, generator : 同前
    top_k : int
        检索 top_k

    返回
    ----
    List[Dict] : 可直接转为 HuggingFace Dataset 的字典列表
    """
    from main_part.prompt_builder import build_rag_prompt

    eval_samples = []

    for q in questions:
        question = q["question"]
        ground_truth = q.get("ground_truth", "")

        # 执行实际检索
        retrieved = retriever.search(question, embedder, top_k=top_k)
        contexts = [chunk["text"] for chunk in retrieved]

        # 生成答案
        prompt = build_rag_prompt(question, retrieved)
        answer = generator(prompt)

        eval_samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    return eval_samples


def run_ragas_evaluation(
    eval_samples: List[Dict],
    judge_model: Optional[str] = None,
    metrics: Optional[List] = None,
) -> Dict:
    """
    运行 RAGAS 评估。

    参数
    ----
    eval_samples : List[Dict]
        build_eval_dataset 的输出
    judge_model : str
        Ollama 模型名（用作 judge），默认 JUDGE_MODEL
    metrics : List
        RAGAS 指标列表，默认四项全开

    返回
    ----
    Dict : {metric_name: score, ..., per_sample: [...]}
    """
    if not check_ragas_installed():
        return {"error": install_ragas_instructions()}

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.llms import llm_factory
    from langchain_openai import ChatOpenAI

    if judge_model is None:
        judge_model = JUDGE_MODEL

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # 构建 HuggingFace Dataset
    dataset = Dataset.from_list(eval_samples)

    # 配置 judge LLM（通过 Ollama）
    judge_llm = ChatOpenAI(
        model=judge_model,
        base_url=OLLAMA_BASE,
        api_key="ollama",  # Ollama 不需要真实 key
        temperature=0,     # Judge 必须用 0，确保一致性
    )

    # 配置 embeddings（用于 Answer Relevancy 等需 embedding 的指标）
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
        )
    except ImportError:
        print("[WARN] langchain_huggingface 未安装，Answer Relevancy 可能无法计算")
        embeddings = None

    print(f"Judge 模型: {judge_model}")
    print(f"评估样本数: {len(eval_samples)}")
    print(f"指标: {[m.name for m in metrics]}")
    print("运行中...")

    # 执行评估
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=embeddings,
        )
    except Exception as e:
        return {
            "error": f"RAGAS 评估失败: {e}",
            "hint": (
                "常见原因:\n"
                "1. Ollama 未运行 → 运行 ollama serve\n"
                f"2. judge 模型 {judge_model} 未拉取 → ollama pull {judge_model}\n"
                "3. RAGAS 版本不兼容 → pip install ragas==0.2.10 (稳定版)\n"
                "4. 小模型结构化输出不稳定 → 尝试更大的 judge 模型"
            ),
        }

    # 提取分数
    scores = {}
    for key in result:
        if key not in ("question", "answer", "contexts", "ground_truth"):
            scores[key] = round(float(result[key]), 4) if result[key] is not None else None

    # 逐样本分数
    per_sample = []
    for i in range(len(eval_samples)):
        sample = {"id": i}
        for key in result:
            if key not in ("question", "answer", "contexts", "ground_truth"):
                val = result[key][i] if result[key] is not None else None
                if isinstance(val, (int, float)):
                    sample[key] = round(float(val), 4)
        per_sample.append(sample)

    scores["per_sample"] = per_sample
    return scores


def print_ragas_report(scores: Dict) -> None:
    """打印 RAGAS 评估报告。"""
    if "error" in scores:
        print(f"\n❌ RAGAS 评估出错: {scores['error']}")
        if "hint" in scores:
            print(f"\n💡 提示:\n{scores['hint']}")
        return

    print("\n" + "=" * 50)
    print("RAGAS 评估报告")
    print("=" * 50)

    metric_labels = {
        "faithfulness": "Faithfulness (忠实度)",
        "answer_relevancy": "Answer Relevancy (答案相关性)",
        "context_precision": "Context Precision (上下文精度)",
        "context_recall": "Context Recall (上下文召回)",
    }

    for key, label in metric_labels.items():
        if key in scores and scores[key] is not None:
            val = scores[key]
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            status = "✅" if val >= 0.80 else ("⚠️" if val >= 0.60 else "❌")
            print(f"\n{status} {label}")
            print(f"   {bar} {val:.2%}")

    print("\n" + "=" * 50)
    print("📊 解读:")
    print("  Faithfulness < 0.7  → 严重幻觉，生成模型需要升级或 Prompt 优化")
    print("  Context Precision < 0.6 → 检索噪声大，考虑混合检索或重排序")
    print("  Context Recall < 0.6 → 检索遗漏多，考虑增加 top_k 或查询改写")
    print("  Answer Relevancy < 0.6 → 模型没有正面回答问题")
    print("=" * 50 + "\n")


def quick_faithfulness_check(
    questions: List[Dict],
    retriever,
    embedder,
    generator,
    n_samples: int = 3,
) -> None:
    """
    快速人工抽检——不需要 LLM judge。
    打印 n 个样本的 {question, contexts, answer}，让人眼判断 Faithfulness。

    用法：在完成基线运行后，快速看看实际输出质量。
    """
    from main_part.prompt_builder import build_rag_prompt

    print("\n" + "=" * 60)
    print(f"快速抽检 ({n_samples} 题) —— 请人眼判断答案质量")
    print("=" * 60)

    import random
    random.seed(42)
    samples = random.sample(questions, min(n_samples, len(questions)))

    for i, q in enumerate(samples):
        retrieved = retriever.search(q["question"], embedder, top_k=10)
        prompt = build_rag_prompt(q["question"], retrieved)
        answer = generator(prompt)

        print(f"\n{'─' * 50}")
        print(f"📝 问题 [{i+1}]: {q['question']}")
        if q.get("ground_truth"):
            print(f"🎯 标准答案: {q['ground_truth']}")
        print(f"\n📄 检索到的 Chunks (top-3):")
        for j, chunk in enumerate(retrieved[:3]):
            text = chunk["text"][:150].replace("\n", " ")
            print(f"  [{j+1}] chunk_id={chunk['chunk_id']} | 第{chunk['source_page']}页")
            print(f"      {text}...")
        print(f"\n🤖 模型回答: {answer[:300]}")

        # 简单自动检测
        issues = []
        if len(answer) < 10:
            issues.append("答案过短（可能拒答）")
        if "文档未提及" in answer:
            issues.append("模型报告'文档未提及'")
        if issues:
            print(f"⚠️ 自动检测: {', '.join(issues)}")

    print("\n" + "=" * 60)
    print("请判断: 答案是否基于检索到的上下文？是否有幻觉？")
    print("如果大部分回答都有问题 → 生成模型是瓶颈。")
    print("如果检索到的 chunk 不相关 → 检索是瓶颈。")
    print("=" * 60 + "\n")
