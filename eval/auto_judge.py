"""
LLM Judge — 自动化答案正确性评估
=================================
用 LLM 判断模型回答是否覆盖了 ground truth 的核心信息，
替代不可靠的 keyword 匹配。

输出三分类：correct / partial / wrong

设计原则：
- 只做信息比对，不做知识判断
- ground_truth 为"空"时跳过（无法判断）
- 结构化输出，便于统计
"""

import os
import json
import requests
from typing import List, Dict, Tuple

OLLAMA_API = 'http://localhost:11434/api/chat'


def _clear_proxy():
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(key, None)
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'


JUDGE_PROMPT = """你是一个严格的评估者。你的任务是：判断一个AI助手的回答是否正确回答了用户的问题。

评估规则：
1. 将AI回答与"参考答案"对比，判断核心信息是否一致
2. 不要求逐字相同——语义等价即可。例如"246元/人"和"单价246元"是同义的
3. 对于列举类问题，AI回答覆盖了参考答案中60%以上的关键点即算 correct
4. 如果AI回答的核心信息正确但遗漏了部分细节，评为 partial
5. 如果AI回答的核心信息错误、完全无关、或说"不知道/文档未提及"但参考答案有明确信息，评为 wrong

请严格输出以下三个词之一（不要输出其他内容）：
- correct
- partial
- wrong

---
问题：{question}
参考答案：{ground_truth}
AI回答：{answer}
---
判断结果："""


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    judge_model: str = 'qwen2.5:7b',
    timeout: int = 120,
) -> Tuple[str, str]:
    """
    用 LLM 判断答案正确性。

    返回
    ----
    (verdict, raw_judge_output)
        verdict: 'correct' | 'partial' | 'wrong'
        raw_judge_output: judge 模型的完整回答（用于调试）
    """
    if not ground_truth or not answer:
        return ('wrong', 'EMPTY_GT_OR_ANSWER')

    if answer.startswith('[ERROR'):
        return ('wrong', 'GENERATION_ERROR')

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    body = {
        'model': judge_model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'options': {'temperature': 0, 'num_predict': 50},
    }

    try:
        _clear_proxy()
        r = requests.post(OLLAMA_API, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        raw = data.get('message', {}).get('content', '').strip().lower()

        # 解析 verdict
        if 'correct' in raw and 'partial' not in raw:
            verdict = 'correct'
        elif 'partial' in raw:
            verdict = 'partial'
        elif 'wrong' in raw:
            verdict = 'wrong'
        else:
            # fallback: 如果模型没按要求输出，从内容推断
            if any(w in raw for w in ['正确', '准确', '一致', 'correct']):
                verdict = 'correct'
            elif any(w in raw for w in ['部分', 'partial', '遗漏']):
                verdict = 'partial'
            elif any(w in raw for w in ['错误', '不正确', 'wrong', '未提及']):
                verdict = 'wrong'
            else:
                verdict = 'unknown'

        return (verdict, raw)

    except Exception as e:
        return ('error', str(e))


def batch_judge(
    test_cases: List[Dict],
    answer_key: str = 'answer',
    judge_model: str = 'qwen2.5:7b',
    verbose: bool = True,
) -> Dict:
    """
    批量评估答案正确性。

    参数
    ----
    test_cases : List[Dict]
        每个元素包含: question, ground_truth, {answer_key}
    answer_key : str
        答案字段名，如 'answers.actual' 或 'answers.oracle'
    judge_model : str
        评估模型
    verbose : bool
        是否打印逐题结果

    返回
    ----
    Dict : {
        'correct': int, 'partial': int, 'wrong': int, 'error': int,
        'accuracy': float,  # correct / total
        'fuzzy_accuracy': float,  # (correct + 0.5*partial) / total
        'per_question': [{verdict, raw_judge, ...}, ...],
        'compared_with_manual': Dict  # 如果提供了 manual_labels
    }
    """
    counts = {'correct': 0, 'partial': 0, 'wrong': 0, 'error': 0, 'unknown': 0}
    per_question = []

    for i, tc in enumerate(test_cases):
        question = tc.get('question', '')
        ground_truth = tc.get('ground_truth', '')
        answer = tc.get(answer_key, tc.get('answer', ''))

        # 支持嵌套 key 如 'answers.actual'
        if '.' in answer_key:
            parts = answer_key.split('.')
            obj = tc
            for p in parts:
                obj = obj.get(p, {}) if isinstance(obj, dict) else {}
            answer = obj if isinstance(obj, str) else ''

        verdict, raw = judge_answer(question, ground_truth, answer, judge_model)

        counts[verdict] = counts.get(verdict, 0) + 1

        per_q = {
            'id': tc.get('id', f'q{i}'),
            'question': question[:60],
            'ground_truth': ground_truth[:80] if ground_truth else '',
            'answer_preview': answer[:80] if answer else '',
            'verdict': verdict,
            'raw_judge': raw[:200],
        }
        per_question.append(per_q)

        if verbose:
            emoji = {'correct': '✅', 'partial': '⚠️', 'wrong': '❌', 'error': '💥', 'unknown': '?'}
            print(f"[{i+1}/{len(test_cases)}] {tc.get('id','?'):5s} {emoji.get(verdict,'?')} {verdict:7s} | {question[:40]}...")

    total = len(test_cases)
    accuracy = counts['correct'] / total if total > 0 else 0
    fuzzy_accuracy = (counts['correct'] + 0.5 * counts['partial']) / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"LLM Judge 评估结果 (模型: {judge_model}, 答案来源: {answer_key})")
    print(f"{'='*50}")
    print(f"总题数:   {total}")
    print(f"correct:  {counts['correct']} ({counts['correct']/total:.0%})")
    print(f"partial:  {counts['partial']} ({counts['partial']/total:.0%})")
    print(f"wrong:    {counts['wrong']} ({counts['wrong']/total:.0%})")
    print(f"error:    {counts['error']}")
    print(f"---")
    print(f"准确率(严格):  {accuracy:.0%}")
    print(f"准确率(宽松):  {fuzzy_accuracy:.0%}")

    result = {
        'counts': counts,
        'total': total,
        'accuracy': round(accuracy, 4),
        'fuzzy_accuracy': round(fuzzy_accuracy, 4),
        'per_question': per_question,
    }
    return result


def compare_with_manual(auto_results: Dict, manual_labels: Dict[str, str]) -> Dict:
    """
    对比 LLM judge 自动评估 vs 人工标注。

    manual_labels: {question_id: 'correct'|'partial'|'wrong'}
    """
    agree = 0
    disagree = []
    total = 0

    for pq in auto_results['per_question']:
        qid = pq['id']
        if qid not in manual_labels:
            continue
        total += 1
        auto = pq['verdict']
        manual = manual_labels[qid]
        if auto == manual:
            agree += 1
        else:
            disagree.append({
                'id': qid,
                'auto': auto,
                'manual': manual,
                'question': pq['question'],
            })

    agreement = agree / total if total > 0 else 0

    print(f"\n--- LLM Judge vs 人工标注 ---")
    print(f"一致率: {agree}/{total} ({agreement:.0%})")
    if disagree:
        print(f"不一致的 {len(disagree)} 题:")
        for d in disagree:
            print(f"  {d['id']}: auto={d['auto']} manual={d['manual']} | {d['question'][:40]}...")

    return {
        'agreement': round(agreement, 4),
        'agree_count': agree,
        'total': total,
        'disagree': disagree,
    }
