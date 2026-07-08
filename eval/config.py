"""
DocQA 评估基线配置
====================
所有路径、模型名、评估参数集中管理，方便后续调整。
"""

import os

# --- 项目路径 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PART = os.path.join(PROJECT_ROOT, "main_part")
PDF_PATH = os.path.join(PROJECT_ROOT, "uploaded.pdf")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "bge-small-zh-v1.5")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
TEST_DATA_DIR = os.path.join(EVAL_DIR, "data")

# --- 检索参数 ---
TOP_K_VALUES = [3, 5, 10, 20]  # 评估时测试的 k 值
DEFAULT_TOP_K = 10

# --- Embedding 模型 ---
EMBEDDING_MODEL_PATH = MODEL_DIR
EMBEDDING_DIM = 512

# --- 生成模型（被测系统）---
# 注意：generator.py 里硬编码了 qwen2.5:1.5b，如果 Ollama 实际用的是其他模型，改这里
GENERATOR_MODEL = "qwen2.5:7b"  # 非 thinking 模型，RAG 生成稳定可靠
OLLAMA_BASE = "http://localhost:11434/v1"

# --- Judge 模型（用于 RAGAS 评估）---
# 必须比被测模型更强，推荐 7B+。当前 Ollama 可用 qwen3.5:4b 或 qwen3.5:9b
JUDGE_MODEL = "qwen2.5:7b"  # 与被测模型同款做 judge（本地只有两个模型，同模型 judge 优于 thinking 模型的不可靠输出）

# --- RAGAS 指标阈值（"良好"标准）---
# 这些是初始参考值，应根据实际基线数据和业务需求调整
THRESHOLDS = {
    "recall_at_5": 0.85,
    "recall_at_10": 0.90,
    "mrr": 0.75,
    "context_recall": 0.85,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.80,
}

# --- Oracle 测试 ---
ORACLE_TOP_K = 10
RANDOM_SEED = 42

# --- 测试集 ---
# 题型分布参考
QUESTION_TYPE_DIST = {
    "fact_lookup": 0.40,       # 事实查询（单点检索）
    "summary": 0.30,           # 总结/归纳（多点综合）
    "exact_match": 0.20,       # 精确匹配（数字/编号/日期）
    "boundary": 0.10,          # 拒答/边界测试
}
