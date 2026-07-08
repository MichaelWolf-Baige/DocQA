"""
自动化评估 vs 人工标注对比
==========================
1. LLM judge 评估 28 题的 Actual 和 Oracle 答案
2. 与人工标注对比，计算一致率
3. 输出对比报告
"""

import os, sys, json, time
from datetime import datetime
from collections import defaultdict

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.testset import load_testset
from eval.auto_judge import judge_answer
from main_part.embedder import Embedder
from main_part.retriever import Retriever
from main_part.prompt_builder import build_rag_prompt
from main_part.generator import generate_answer
from eval.config import DEFAULT_TOP_K


# ====== 人工标注（来自 manual_review.md） ======
MANUAL_ACTUAL = {
    'q001': 'correct', 'q002': 'correct', 'q003': 'wrong', 'q004': 'correct',
    'q005': 'correct', 'q006': 'correct', 'q007': 'partial', 'q008': 'correct',
    'q009': 'wrong', 'q010': 'correct', 'q011': 'correct', 'q012': 'correct',
    'q013': 'correct', 'q014': 'correct', 'q015': 'correct', 'q016': 'correct',
    'q017': 'correct', 'q018': 'correct', 'q019': 'correct', 'q020': 'correct',
    'q021': 'wrong', 'q022': 'correct', 'q023': 'partial', 'q024': 'correct',
    'q025': 'correct', 'q027': 'partial', 'q028': 'correct', 'q029': 'wrong',
}

MANUAL_ORACLE = {
    'q001': 'correct', 'q002': 'correct', 'q003': 'correct', 'q004': 'correct',
    'q005': 'correct', 'q006': 'partial', 'q007': 'wrong', 'q008': 'correct',
    'q009': 'correct', 'q010': 'correct', 'q011': 'correct', 'q012': 'correct',
    'q013': 'correct', 'q014': 'correct', 'q015': 'correct', 'q016': 'correct',
    'q017': 'correct', 'q018': 'correct', 'q019': 'correct', 'q020': 'correct',
    'q021': 'wrong', 'q022': 'correct', 'q023': 'partial', 'q024': 'correct',
    'q025': 'wrong', 'q027': 'partial', 'q028': 'correct', 'q029': 'correct',
}


def main():
    print("加载系统...")
    questions = load_testset()
    retrievable = [q for q in questions if len(q['relevant_chunk_ids']) > 0]
    embedder = Embedder()
    retriever = Retriever()
    all_data = retriever.collection.get()
    print(f"测试集: {len(retrievable)} 题, 向量库: {retriever.collection.count()} 条")

    # ====== 生成 Actual + Oracle 答案 ======
    print("\n" + "="*60)
    print("生成 Actual 和 Oracle 答案 (28x2 = 56次调用)")
    print("="*60)

    samples = []
    for i, q in enumerate(retrievable):
        question = q['question']
        relevant_ids = set(q['relevant_chunk_ids'])
        gt = q.get('ground_truth', '')

        # Actual
        actual_results = retriever.search(question, embedder, top_k=DEFAULT_TOP_K)
        prompt_a = build_rag_prompt(question, actual_results)
        answer_a = generate_answer(prompt_a)

        # Oracle
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

        samples.append({
            'id': q['id'],
            'question': question,
            'ground_truth': gt,
            'answer_actual': answer_a,
            'answer_oracle': answer_b,
            'manual_actual': MANUAL_ACTUAL.get(q['id'], '?'),
            'manual_oracle': MANUAL_ORACLE.get(q['id'], '?'),
        })
        print(f"[{i+1}/{len(retrievable)}] {q['id']} done")

    # ====== LLM Judge 评估 ======
    print("\n" + "="*60)
    print("LLM Judge 评估 Actual 答案")
    print("="*60)

    auto_actual = {'correct': 0, 'partial': 0, 'wrong': 0, 'error': 0}
    auto_oracle = {'correct': 0, 'partial': 0, 'wrong': 0, 'error': 0}
    agree_actual = 0
    agree_oracle = 0
    total = 0

    for i, s in enumerate(samples):
        gt = s['ground_truth']
        manual_a = s['manual_actual']
        manual_o = s['manual_oracle']

        # Judge actual
        verdict_a, raw_a = judge_answer(s['question'], gt, s['answer_actual'])
        s['auto_actual'] = verdict_a
        s['raw_actual'] = raw_a
        auto_actual[verdict_a] = auto_actual.get(verdict_a, 0) + 1
        a_agree = '✅' if verdict_a == manual_a else '❌'
        if verdict_a == manual_a:
            agree_actual += 1

        # Judge oracle
        verdict_o, raw_o = judge_answer(s['question'], gt, s['answer_oracle'])
        s['auto_oracle'] = verdict_o
        s['raw_oracle'] = raw_o
        auto_oracle[verdict_o] = auto_oracle.get(verdict_o, 0) + 1
        o_agree = '✅' if verdict_o == manual_o else '❌'
        if verdict_o == manual_o:
            agree_oracle += 1

        total += 1
        print(f"[{i+1}/{len(samples)}] {s['id']} "
              f"A:auto={verdict_a:7s} vs manual={manual_a:7s} {a_agree} | "
              f"O:auto={verdict_o:7s} vs manual={manual_o:7s} {o_agree}")

    # ====== 汇总 ======
    n = len(samples)
    print("\n" + "="*60)
    print("对比结果汇总")
    print("="*60)

    print(f"\n### LLM Judge 自动评估")
    print(f"| 条件 | correct | partial | wrong | 严格准确率 | 宽松准确率 |")
    print(f"|------|---------|---------|-------|-----------|-----------|")

    for label, counts in [('Actual', auto_actual), ('Oracle', auto_oracle)]:
        strict = counts['correct'] / n
        fuzzy = (counts['correct'] + 0.5 * counts['partial']) / n
        print(f"| {label} | {counts['correct']} | {counts['partial']} | {counts['wrong']} | {strict:.0%} | {fuzzy:.0%} |")

    print(f"\n### LLM Judge vs 人工标注一致率")
    print(f"- Actual: {agree_actual}/{n} ({agree_actual/n:.0%})")
    print(f"- Oracle: {agree_oracle}/{n} ({agree_oracle/n:.0%})")
    print(f"- 综合: {(agree_actual+agree_oracle)}/{2*n} ({(agree_actual+agree_oracle)/(2*n):.0%})")

    # 不一致的题
    disagree_cases = []
    for s in samples:
        if s['auto_actual'] != s['manual_actual']:
            disagree_cases.append({
                'id': s['id'], 'condition': 'actual', 'question': s['question'][:40],
                'auto': s['auto_actual'], 'manual': s['manual_actual'],
                'answer': s['answer_actual'][:100],
            })
        if s['auto_oracle'] != s['manual_oracle']:
            disagree_cases.append({
                'id': s['id'], 'condition': 'oracle', 'question': s['question'][:40],
                'auto': s['auto_oracle'], 'manual': s['manual_oracle'],
                'answer': s['answer_oracle'][:100],
            })

    if disagree_cases:
        print(f"\n### 不一致的 {len(disagree_cases)} 题")
        for d in disagree_cases:
            print(f"  {d['id']} [{d['condition']}] auto={d['auto']} manual={d['manual']} | {d['question']}...")

    # ====== keyword 对比 ======
    print(f"\n### 三方案对比")
    print(f"| 评估方案 | Actual准确率 | Oracle准确率 | 与人工一致率 |")
    print(f"|----------|:-----------:|:-----------:|:-----------:|")
    kw_a = sum(1 for s in samples if s['ground_truth'][:8] in s['answer_actual'] and s['ground_truth']) / n
    kw_o = sum(1 for s in samples if s['ground_truth'][:8] in s['answer_oracle'] and s['ground_truth']) / n
    print(f"| keyword匹配 | {kw_a:.0%} | {kw_o:.0%} | ~{40:.0%}% |")

    a_strict = auto_actual['correct'] / n
    o_strict = auto_oracle['correct'] / n
    print(f"| LLM Judge | {a_strict:.0%} | {o_strict:.0%} | {agree_actual/n:.0%} |")

    manual_a = sum(1 for s in samples if s['manual_actual'] == 'correct') / n
    manual_pa = sum(1 for s in samples if s['manual_actual'] in ('correct', 'partial')) / n
    manual_o = sum(1 for s in samples if s['manual_oracle'] == 'correct') / n
    manual_po = sum(1 for s in samples if s['manual_oracle'] in ('correct', 'partial')) / n
    print(f"| 人工标注 | {manual_a:.0%}({manual_pa:.0%}含partial) | {manual_o:.0%}({manual_po:.0%}含partial) | 基准 |")

    # ====== 保存 ======
    output = {
        'timestamp': datetime.now().isoformat(),
        'judge_model': 'qwen2.5:7b',
        'samples': samples,
        'auto_actual_counts': auto_actual,
        'auto_oracle_counts': auto_oracle,
        'agreement_actual': agree_actual / n,
        'agreement_oracle': agree_oracle / n,
    }
    outpath = os.path.join(PROJECT_ROOT, 'eval', 'data',
                           f'judge_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        # 只保存关键字段
        slim = {
            'timestamp': output['timestamp'],
            'samples': [{
                'id': s['id'], 'question': s['question'],
                'ground_truth': s['ground_truth'],
                'manual_actual': s['manual_actual'],
                'manual_oracle': s['manual_oracle'],
                'auto_actual': s['auto_actual'],
                'auto_oracle': s['auto_oracle'],
                'answer_actual': s['answer_actual'][:200],
                'answer_oracle': s['answer_oracle'][:200],
            } for s in samples],
            'agreement_actual': output['agreement_actual'],
            'agreement_oracle': output['agreement_oracle'],
        }
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {outpath}")


if __name__ == '__main__':
    main()
