"""
DocQA 完整基线评估 —— 一键运行
===============================
运行检索指标 + Oracle 四条件分析，输出完整评估报告。
"""
import os, sys, json, time, random
from datetime import datetime
from collections import defaultdict

# 清除代理（localhost 不需要）
for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.testset import load_testset, print_stats, compute_stats
from eval.metrics import compute_all_metrics, recall_at_k, precision_at_k, mrr, hit_rate
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.generator import generate_answer
from main_part.prompt_builder import build_rag_prompt

TOP_K = 10
TOP_K_VALUES = [3, 5, 10, 20]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def main():
    start_time = time.time()
    report = []

    # ========== 加载 ==========
    print("加载测试集...")
    questions = load_testset()
    stats = compute_stats(questions)

    retrievable = [q for q in questions if len(q['relevant_chunk_ids']) > 0]
    unanswerable = [q for q in questions if len(q['relevant_chunk_ids']) == 0]
    print(f"可检索: {len(retrievable)}, 拒答: {len(unanswerable)}")

    print("加载系统...")
    embedder = Embedder()
    retriever = Retriever()
    print(f"向量库: {retriever.collection.count()} 条 (cosine 距离)")

    # ========== Step 1: 检索指标 ==========
    print("\n" + "="*60)
    print("Step 1: 检索层指标")
    print("="*60)

    metrics = compute_all_metrics(retrievable, retriever, embedder, k_values=TOP_K_VALUES)

    report.append("## Step 1: 检索层指标\n")
    for k in TOP_K_VALUES:
        r = metrics.get(f'recall@{k}')
        p = metrics.get(f'precision@{k}')
        h = metrics.get(f'hit_rate@{k}')
        n = metrics.get(f'ndcg@{k}')
        print(f"Recall@{k:>2}:    {r:.2%}" if isinstance(r, float) else f"Recall@{k:>2}:    N/A")
        print(f"Precision@{k:>2}: {p:.2%}" if isinstance(p, float) else "")
        print(f"Hit Rate@{k:>2}: {h:.2%}" if isinstance(h, float) else "")
        print(f"NDCG@{k:>2}:     {n:.4f}" if isinstance(n, float) else "")
        print()
        if isinstance(r, float):
            report.append(f"- Recall@{k}: {r:.2%} | Precision@{k}: {p:.2%} | Hit Rate@{k}: {h:.2%} | NDCG@{k}: {n:.4f}")

    m = metrics.get('mrr')
    print(f"MRR:        {m:.4f}" if isinstance(m, float) else "")
    if isinstance(m, float):
        report.append(f"\n- MRR: {m:.4f}")

    # 按题型分析召回
    print("\n按题型 Recall@10:")
    report.append("\n### 按题型 Recall@10")
    by_type = defaultdict(list)
    for i, r in enumerate(metrics['per_question']):
        qtype = retrievable[i].get('question_type', 'unknown')
        by_type[qtype].append(r.get('recall@10', 0))
    for qtype, scores in sorted(by_type.items()):
        avg = sum(scores) / len(scores)
        print(f"  {qtype}: {avg:.2%} ({len(scores)}题)")
        report.append(f"- {qtype}: {avg:.2%} ({len(scores)}题)")

    # ========== Step 2: Oracle 四条件分析 ==========
    print("\n" + "="*60)
    print("Step 2: Oracle 瓶颈分析 (四条件)")
    print("="*60)

    oracle_results = []
    all_data = retriever.collection.get()
    all_chunk_ids = [int(i) for i in all_data['ids']]

    for i, q in enumerate(retrievable):
        question = q['question']
        relevant_ids = set(q['relevant_chunk_ids'])
        gt = q.get('ground_truth', '')
        print(f"[{i+1}/{len(retrievable)}] {q['id']}: {question[:50]}...", end=' ', flush=True)

        # A: 实际检索
        actual_results = retriever.search(question, embedder, top_k=TOP_K)
        prompt_a = build_rag_prompt(question, actual_results)
        answer_a = generate_answer(prompt_a)

        # B: Oracle
        oracle_chunks = []
        for j, cid in enumerate(all_data['ids']):
            if int(cid) in relevant_ids:
                oracle_chunks.append({
                    'text': all_data['documents'][j],
                    'source_page': all_data['metadatas'][j].get('source_page', '?'),
                    'score': 1.0,
                    'chunk_id': int(cid),
                })
        prompt_b = build_rag_prompt(question, oracle_chunks) if oracle_chunks else None
        answer_b = generate_answer(prompt_b) if prompt_b else '[N/A]'

        # C: 裸模型
        prompt_c = '请回答以下问题。如果不知道答案，请直接说不知道。\n\n问题：' + question + '\n答案：'
        answer_c = generate_answer(prompt_c)

        # D: 随机干扰
        irrelevant = [cid for cid in all_chunk_ids if cid not in relevant_ids]
        selected = random.sample(irrelevant, min(TOP_K, len(irrelevant)))
        noise_chunks = []
        for j, cid in enumerate(all_data['ids']):
            if int(cid) in selected:
                noise_chunks.append({
                    'text': all_data['documents'][j],
                    'source_page': all_data['metadatas'][j].get('source_page', '?'),
                    'score': 0.0,
                    'chunk_id': int(cid),
                })
        prompt_d = build_rag_prompt(question, noise_chunks)
        answer_d = generate_answer(prompt_d)

        # 检索命中情况
        top10_ids = [c['chunk_id'] for c in actual_results[:TOP_K]]
        retrieval_hit = len(set(top10_ids) & relevant_ids)

        oracle_results.append({
            'id': q['id'], 'question': question, 'ground_truth': gt,
            'relevant_count': len(relevant_ids),
            'retrieval_hit': retrieval_hit,
            'answers': {'actual': answer_a, 'oracle': answer_b, 'bare': answer_c, 'noise': answer_d},
            'actual_top10_ids': top10_ids,
            'oracle_chunk_ids': list(relevant_ids),
        })

        # 简易判断
        gt_kw = gt[:8] if gt else ''
        a_ok = gt_kw in answer_a if answer_a and gt_kw else '?'
        b_ok = gt_kw in answer_b if answer_b and gt_kw else '?'
        print(f"检索:{retrieval_hit}/{len(relevant_ids)} | 实际:{a_ok} Oracle:{b_ok}")

    # Oracle 汇总
    total = len(oracle_results)
    actual_hits = sum(1 for r in oracle_results
                      if r['ground_truth'][:8] in r['answers']['actual']
                      if r['answers']['actual'] and r['ground_truth'])
    oracle_hits = sum(1 for r in oracle_results
                      if r['ground_truth'][:8] in r['answers']['oracle']
                      if r['answers']['oracle'] and r['ground_truth'])
    bare_refuse = sum(1 for r in oracle_results
                      if '不知道' in r['answers']['bare'] or '未提及' in r['answers']['bare'])
    noise_confused = sum(1 for r in oracle_results
                         if r['ground_truth'][:8] in r['answers']['noise']
                         if r['answers']['noise'] and r['ground_truth'])

    # 按检索是否命中拆分 Oracle 准确率
    hit_cases = [r for r in oracle_results if r['retrieval_hit'] > 0]
    miss_cases = [r for r in oracle_results if r['retrieval_hit'] == 0]
    hit_oracle_ok = sum(1 for r in hit_cases
                        if r['ground_truth'][:8] in r['answers']['oracle']
                        if r['answers']['oracle'] and r['ground_truth'])
    miss_oracle_ok = sum(1 for r in miss_cases
                         if r['ground_truth'][:8] in r['answers']['oracle']
                         if r['answers']['oracle'] and r['ground_truth'])

    print("\n" + "=" * 60)
    print("Oracle 分析汇总")
    print("=" * 60)
    print(f"总题数: {total}")
    print(f"\n简易准确率 (keyword match, 仅供参考):")
    print(f"  A-实际检索: {actual_hits}/{total} ({actual_hits/total:.0%})")
    print(f"  B-Oracle:   {oracle_hits}/{total} ({oracle_hits/total:.0%})")
    print(f"  C-裸模型拒绝率: {bare_refuse}/{total} ({bare_refuse/total:.0%})")
    print(f"  D-噪声干扰误答: {noise_confused}/{total} ({noise_confused/total:.0%})")
    print(f"\n检索命中的题 ({len(hit_cases)}题) Oracle准确: {hit_oracle_ok}/{len(hit_cases)} ({hit_oracle_ok/len(hit_cases):.0%})" if hit_cases else "")
    print(f"检索未命中的题 ({len(miss_cases)}题) Oracle准确: {miss_oracle_ok}/{len(miss_cases)} ({miss_oracle_ok/len(miss_cases):.0%})" if miss_cases else "")

    # 检索瓶颈题（Oracle答对了但实际答错了）
    retrieval_bottleneck = [
        r for r in oracle_results
        if r['ground_truth'] and r['ground_truth'][:8] in r['answers']['oracle']
        and r['ground_truth'][:8] not in r['answers']['actual']
        and r['answers']['actual']
    ]
    gen_bottleneck = [
        r for r in oracle_results
        if r['ground_truth'] and r['ground_truth'][:8] not in r['answers']['oracle']
        and r['answers']['oracle']
    ]

    print(f"\n检索瓶颈题 (Oracle对但实际错): {len(retrieval_bottleneck)}题")
    for r in retrieval_bottleneck:
        print(f"  {r['id']}: {r['question'][:50]}")

    print(f"生成瓶颈题 (Oracle也错): {len(gen_bottleneck)}题")
    for r in gen_bottleneck:
        print(f"  {r['id']}: {r['question'][:50]}")

    # ========== 生成报告 ==========
    elapsed = time.time() - start_time

    report_full = [
        "=" * 70,
        "DocQA 评估基线报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"耗时: {elapsed/60:.1f} 分钟",
        "=" * 70,
        "",
        "## 系统配置",
        f"- Embedding: BAAI/bge-small-zh-v1.5 (512维, CPU)",
        f"- 向量库: ChromaDB (cosine 距离, 47 chunks)",
        f"- 生成模型: qwen2.5:7b (Ollama)",
        f"- PDF: uploaded.pdf (29页旅游合同, 47 chunks)",
        "",
        "## 测试集",
        f"- 总题数: {stats['total']}",
        f"- 题型分布: {stats['by_type']}",
        f"- 难度分布: {stats['by_difficulty']}",
        f"- 拒答题: {stats['boundary_count']}",
        "",
    ] + report + [
        "",
        "## Step 2: Oracle 瓶颈分析",
        "",
        f"### 四条件简易准确率",
        f"| 条件 | 准确数 | 准确率 |",
        f"|------|--------|--------|",
        f"| A-实际检索 | {actual_hits}/{total} | {actual_hits/total:.0%} |",
        f"| B-Oracle | {oracle_hits}/{total} | {oracle_hits/total:.0%} |",
        f"| C-裸模型拒绝率 | {bare_refuse}/{total} | {bare_refuse/total:.0%} |",
        f"| D-噪声干扰误答 | {noise_confused}/{total} | {noise_confused/total:.0%} |",
        "",
        f"### 瓶颈分类",
        f"- 检索瓶颈 (Oracle对但实际错): {len(retrieval_bottleneck)} 题",
        f"- 生成瓶颈 (Oracle也错): {len(gen_bottleneck)} 题",
        f"- 检索命中时 Oracle 准确率: {hit_oracle_ok}/{len(hit_cases)} ({hit_oracle_ok/len(hit_cases):.0%})" if hit_cases else "",
        f"- 检索未命中时 Oracle 准确率: {miss_oracle_ok}/{len(miss_cases)} ({miss_oracle_ok/len(miss_cases):.0%})" if miss_cases else "",
        "",
        "### 诊断结论",
    ]

    # 智能诊断
    retrieval_gap = (oracle_hits - actual_hits) / max(total, 1)
    if retrieval_gap > 0.1:
        report_full.append(f"- [检索层] 检索瓶颈显著 (Oracle比实际高 {retrieval_gap:.0%})，建议优先优化: 混合检索、重排序")
    else:
        report_full.append(f"- [检索层] Oracle与实际差距不大 ({retrieval_gap:.0%})，检索不是主要瓶颈")

    if oracle_hits / max(total, 1) < 0.7:
        report_full.append(f"- [生成层] Oracle准确率仅 {oracle_hits/total:.0%}，即使给正确答案也有 {total-oracle_hits} 题答不对，生成模型是瓶颈")
    else:
        report_full.append(f"- [生成层] Oracle准确率 {oracle_hits/total:.0%}，模型信息提取能力尚可")

    report_full.extend([
        "",
        "## 附录: 逐题详细结果",
        "",
        "| ID | 问题 | 检索命中 | 实际 | Oracle | 裸模型 |",
        "|----|------|----------|------|--------|--------|",
    ])
    for r in oracle_results:
        a_short = r['answers']['actual'][:40] if r['answers']['actual'] else '(空)'
        b_short = r['answers']['oracle'][:40] if r['answers']['oracle'] else '(空)'
        c_short = r['answers']['bare'][:30] if r['answers']['bare'] else '(空)'
        report_full.append(f"| {r['id']} | {r['question'][:30]} | {r['retrieval_hit']}/{r['relevant_count']} | {a_short} | {b_short} | {c_short} |")

    report_full.extend([
        "",
        "=" * 70,
        "报告结束",
        "=" * 70,
    ])

    report_text = "\n".join(report_full)

    # 保存
    report_dir = os.path.join(PROJECT_ROOT, 'eval', 'data')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f'baseline_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n\n完整报告已保存: {report_path}")
    print(report_text)


if __name__ == '__main__':
    main()
