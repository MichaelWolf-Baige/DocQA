"""
DocQA 评估基线 — 主运行入口
==============================
按照 Phase 0 的步骤，系统性地评估当前 RAG 系统的各项指标。

用法：
    # 完整评估（推荐）
    python -m eval.run_baseline

    # 仅检索指标（不需要 LLM，不需要 judge）
    python -m eval.run_baseline --retrieval-only

    # 仅 Oracle 分析
    python -m eval.run_baseline --oracle-only

    # 跳过 RAGAS（如果没有合适的 judge 模型）
    python -m eval.run_baseline --skip-ragas

评估流程：
    Step 1: 环境检查（模型是否就绪、向量库是否有数据）
    Step 2: 检索层指标（Recall@k, Precision@k, MRR, NDCG, Hit Rate）
    Step 3: Oracle 瓶颈分析（四条件对照实验）
    Step 4: RAGAS 评估（需 judge 模型，可选）
    Step 5: 综合诊断报告
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict

# 将项目根目录加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.config import (
    PDF_PATH, CHROMA_DIR, MODEL_DIR, TEST_DATA_DIR,
    TOP_K_VALUES, DEFAULT_TOP_K, OLLAMA_BASE,
    GENERATOR_MODEL, JUDGE_MODEL,
)
from eval.metrics import compute_all_metrics
from eval.testset import load_testset, print_stats
from eval.oracle_test import run_oracle_analysis, print_oracle_report


def check_environment() -> Dict:
    """Step 1: 环境检查。"""
    print("\n" + "=" * 60)
    print("Step 1: 环境检查")
    print("=" * 60)

    status = {"ok": True, "warnings": []}

    # 1. 检查 PDF
    if os.path.exists(PDF_PATH):
        print(f"✅ PDF 文件: {PDF_PATH}")
    else:
        print(f"❌ PDF 文件不存在: {PDF_PATH}")
        status["ok"] = False

    # 2. 检查 Embedding 模型
    if os.path.exists(MODEL_DIR):
        print(f"✅ Embedding 模型: {MODEL_DIR}")
    else:
        print(f"❌ Embedding 模型不存在: {MODEL_DIR}")
        status["ok"] = False

    # 3. 检查 Ollama 连接
    try:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE)
        models = client.models.list()
        available = [m.id for m in models]
        print(f"✅ Ollama 可用模型: {', '.join(available)}")

        # 检查生成模型
        if GENERATOR_MODEL in available:
            print(f"✅ 生成模型 {GENERATOR_MODEL} 已就绪")
        else:
            print(f"⚠️ 生成模型 {GENERATOR_MODEL} 不在 Ollama 中！")
            print(f"   可用: {', '.join(available)}")
            print(f"   建议: ollama pull {GENERATOR_MODEL}")
            # 尝试找到替代
            for alt in available:
                if "qwen" in alt.lower():
                    print(f"   💡 可临时使用 {alt} 替代（需修改 eval/config.py）")
                    break
            status["warnings"].append(f"生成模型 {GENERATOR_MODEL} 不可用")

        # 检查 judge 模型
        if JUDGE_MODEL in available:
            print(f"✅ Judge 模型 {JUDGE_MODEL} 已就绪")
        else:
            print(f"⚠️ Judge 模型 {JUDGE_MODEL} 不在 Ollama 中")
            status["warnings"].append(f"Judge 模型 {JUDGE_MODEL} 不可用")

    except Exception as e:
        print(f"❌ Ollama 连接失败: {e}")
        print("   请确保 Ollama 正在运行: ollama serve")
        status["ok"] = False

    # 4. 检查 RAGAS
    try:
        import ragas
        print(f"✅ RAGAS 已安装")
    except ImportError:
        print(f"⚠️ RAGAS 未安装 (pip install ragas)")
        status["warnings"].append("RAGAS 未安装，将跳过 RAGAS 评估")

    return status


def build_system():
    """构建完整的 RAG 系统（Pipeline + 检索/生成）。"""
    from eval.adapters import build_eval_pipeline, dict_search_fn, make_generator

    print("\n构建 RAG pipeline（新架构 docqa）...")
    pipeline = build_eval_pipeline(pdf_path=PDF_PATH, rebuild_index=False)
    print(f"向量库: {pipeline.vector_store.count()} 条记录")

    search_fn = dict_search_fn(pipeline)      # 旧式 dict 接口
    generate = make_generator(pipeline)        # generate(prompt) -> str

    return pipeline, search_fn, generate


def step2_retrieval_metrics(questions, search_fn) -> Dict:
    """Step 2: 检索层指标评估。"""
    print("\n" + "=" * 60)
    print("Step 2: 检索层指标")
    print("=" * 60)

    # 过滤掉拒答题（没有 relevant_chunk_ids）
    retrievable = [q for q in questions if len(q["relevant_chunk_ids"]) > 0]
    unanswerable = [q for q in questions if len(q["relevant_chunk_ids"]) == 0]

    print(f"可检索题: {len(retrievable)}，拒答题: {len(unanswerable)}")

    results = compute_all_metrics(
        retrievable, search_fn, k_values=TOP_K_VALUES
    )

    # 打印指标
    print("\n检索指标汇总:")
    print("-" * 40)
    for k in TOP_K_VALUES:
        recall = results.get(f"recall@{k}", "N/A")
        precision = results.get(f"precision@{k}", "N/A")
        hit = results.get(f"hit_rate@{k}", "N/A")
        ndcg = results.get(f"ndcg@{k}", "N/A")
        print(f"  Recall@{k:>2}:    {recall:.2%}" if isinstance(recall, float) else f"  Recall@{k:>2}:    N/A")
        print(f"  Precision@{k:>2}: {precision:.2%}" if isinstance(precision, float) else f"  Precision@{k:>2}: N/A")
        print(f"  Hit Rate@{k:>2}: {hit:.2%}" if isinstance(hit, float) else f"  Hit Rate@{k:>2}: N/A")
        print(f"  NDCG@{k:>2}:     {ndcg:.4f}" if isinstance(ndcg, float) else f"  NDCG@{k:>2}:     N/A")
        print()

    mrr = results.get("mrr", "N/A")
    print(f"  MRR:        {mrr:.4f}" if isinstance(mrr, float) else f"  MRR:        N/A")

    # 低分题目分析
    low_recall = [
        r for r in results["per_question"]
        if r.get("recall@10", 0) < 0.5
    ]
    if low_recall:
        print(f"\n⚠️ Recall@10 < 50% 的题目 ({len(low_recall)} 题):")
        for r in low_recall:
            print(f"  [{r.get('question', '?')[:60]}...] "
                  f"Recall@10={r.get('recall@10', 0):.0%}")

    return results


def step3_oracle_analysis(questions, pipeline, generate) -> Dict:
    """Step 3: Oracle 瓶颈分析。"""
    print("\n" + "=" * 60)
    print("Step 3: Oracle 瓶颈分析 (四条件对照实验)")
    print("=" * 60)

    # 只对可检索题做 Oracle 分析
    retrievable = [q for q in questions if len(q["relevant_chunk_ids"]) > 0]

    results = run_oracle_analysis(
        retrievable, pipeline, generate,
        top_k=DEFAULT_TOP_K,
        verbose=True,
    )

    print_oracle_report(results)
    return results


def generate_report(
    env_status, retrieval_results, oracle_results, ragas_results, questions
) -> str:
    """Step 5: 生成综合评估报告。"""
    report = []
    report.append("=" * 70)
    report.append("DocQA 评估基线报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    # 环境
    report.append("\n## 环境")
    report.append(f"PDF: {PDF_PATH}")
    report.append(f"Embedding: {MODEL_DIR}")
    report.append(f"生成模型: {GENERATOR_MODEL}")
    report.append(f"Judge 模型: {JUDGE_MODEL}")

    # 测试集
    stats = __import__('eval.testset', fromlist=['compute_stats']).compute_stats(questions)
    report.append(f"\n## 测试集")
    report.append(f"总题数: {stats['total']}")
    report.append(f"题型分布: {stats['by_type']}")
    report.append(f"难度分布: {stats['by_difficulty']}")
    report.append(f"拒答题: {stats['boundary_count']}")

    # 检索指标
    if retrieval_results:
        report.append(f"\n## 检索层指标")
        for k in TOP_K_VALUES:
            r = retrieval_results.get(f"recall@{k}")
            p = retrieval_results.get(f"precision@{k}")
            h = retrieval_results.get(f"hit_rate@{k}")
            if isinstance(r, float):
                report.append(f"Recall@{k}: {r:.2%}")
                report.append(f"Precision@{k}: {p:.2%}" if isinstance(p, float) else "")
                report.append(f"Hit Rate@{k}: {h:.2%}" if isinstance(h, float) else "")
        m = retrieval_results.get("mrr")
        if isinstance(m, float):
            report.append(f"MRR: {m:.4f}")

    # Oracle
    if oracle_results:
        report.append(f"\n## Oracle 瓶颈分析")
        diag = oracle_results["diagnosis"]
        report.append(f"各条件平均答案长度:")
        for cond, length in diag["avg_answer_length"].items():
            report.append(f"  {cond}: {length} 字符")

    # RAGAS
    if ragas_results and "error" not in ragas_results:
        report.append(f"\n## RAGAS 评估")
        for key, val in ragas_results.items():
            if key != "per_sample" and val is not None:
                report.append(f"{key}: {val:.4f}" if isinstance(val, float) else f"{key}: {val}")

    report.append("\n" + "=" * 70)
    report.append("报告结束。请根据数据判断瓶颈位置并制定优化计划。")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="DocQA 评估基线")
    parser.add_argument("--retrieval-only", action="store_true", help="仅运行检索指标")
    parser.add_argument("--oracle-only", action="store_true", help="仅运行 Oracle 分析")
    parser.add_argument("--skip-ragas", action="store_true", help="跳过 RAGAS 评估")
    parser.add_argument("--testset", type=str, default=None, help="测试集路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    # Step 1: 环境检查
    env = check_environment()
    if not env["ok"]:
        print("\n❌ 环境检查未通过，请修复后重试。")
        return

    # 加载测试集
    questions = load_testset(args.testset)
    print_stats(questions)
    print(f"✅ 加载 {len(questions)} 道测试题")

    # 构建系统
    pipeline, search_fn, generate = build_system()

    retrieval_results = None
    oracle_results = None
    ragas_results = None

    # Step 2: 检索指标
    if not args.oracle_only:
        retrieval_results = step2_retrieval_metrics(
            questions, search_fn
        )

    # Step 3: Oracle 分析
    if not args.retrieval_only:
        oracle_results = step3_oracle_analysis(
            questions, pipeline, generate
        )

    # Step 4: RAGAS（可选，需要 judge 模型）
    if not args.skip_ragas and not args.retrieval_only and not args.oracle_only:
        print("\n" + "=" * 60)
        print("Step 4: RAGAS 评估")
        print("=" * 60)
        try:
            from eval.ragas_eval import (
                build_eval_dataset, run_ragas_evaluation, print_ragas_report
            )
            eval_samples = build_eval_dataset(
                questions, search_fn, generate,
                top_k=DEFAULT_TOP_K,
            )
            ragas_results = run_ragas_evaluation(eval_samples)
            print_ragas_report(ragas_results)
        except Exception as e:
            print(f"RAGAS 评估失败: {e}")
            print("这通常是 judge 模型能力不足导致的，可以跳过。")

    # Step 5: 综合报告
    report = generate_report(
        env, retrieval_results, oracle_results, ragas_results, questions
    )
    print(report)

    # 保存报告
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            f"baseline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
