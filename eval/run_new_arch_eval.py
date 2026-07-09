"""
新架构检索效果评估
==================
对比多种检索方案，纯检索层指标（不依赖 LLM）。

方案：
  1. 纯向量 (dense)
  2. 纯 BM25
  3. 混合检索 (hybrid = BM25 + dense, RRF)
  4. 混合 + 重排序 (hybrid + reranker + query_rewrite)
"""
import os, sys, time, json
import numpy as np

for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(key, None)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from docqa.ingestion.base import Chunk
from docqa.ingestion.parser import PyMuPDFParser
from docqa.ingestion.chunker import SentenceChunker
from docqa.ingestion.embedder import BGEEmbedder
from docqa.ingestion.store import ChromaStore
from docqa.retrieval.dense import DenseRetriever
from docqa.retrieval.bm25 import BM25Retriever
from docqa.retrieval.hybrid import HybridRetriever
from docqa.retrieval.reranker import BGEReranker
from docqa.retrieval.query_rewriter import QueryRewriter
from docqa.retrieval.multi_query import MultiQueryRetriever
from docqa.evaluation.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate

import torch

# ----- 配置 -----
PDF_PATH = "D:/桌面/I.pdf"
TESTSET_PATH = os.path.join(PROJECT_ROOT, "eval", "data", "test_questions_256.jsonl")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db_eval")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bge-small-zh-v1.5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_VALUES = [3, 5, 10, 20]

# ----- 加载测试集 -----
def load_testset(filepath):
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            q = json.loads(line)
            if q.get("relevant_chunk_ids"):
                questions.append(q)
    return questions


# ----- 评估函数 -----
def evaluate_method(name, questions, search_fn, k_values):
    metrics = {}
    for k in k_values:
        metrics[f'recall@{k}'] = []
        metrics[f'precision@{k}'] = []
        metrics[f'hit_rate@{k}'] = []
        metrics[f'ndcg@{k}'] = []
    metrics['mrr'] = []

    max_k = max(k_values)

    for q in questions:
        relevant = set(q['relevant_chunk_ids'])
        chunks = search_fn(q['question'], max_k)
        retrieved_ids = [c.chunk_id for c in chunks]

        for k in k_values:
            metrics[f'recall@{k}'].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[f'precision@{k}'].append(precision_at_k(retrieved_ids, relevant, k))
            metrics[f'hit_rate@{k}'].append(hit_rate(retrieved_ids, relevant, k))
            metrics[f'ndcg@{k}'].append(ndcg_at_k(retrieved_ids, relevant, k))
        metrics['mrr'].append(mrr(retrieved_ids, relevant))

    summary = {}
    for key, vals in metrics.items():
        summary[key] = round(float(np.mean(vals)), 4)

    print(f'\n{"="*55}')
    print(f'【{name}】')
    print(f'{"="*55}')
    for k in k_values:
        print(f'Recall@{k:>2}: {summary[f"recall@{k}"]:.2%}  |  '
              f'Precision@{k:>2}: {summary[f"precision@{k}"]:.2%}  |  '
              f'Hit@{k:>2}: {summary[f"hit_rate@{k}"]:.2%}  |  '
              f'NDCG@{k:>2}: {summary[f"ndcg@{k}"]:.4f}')
    print(f'MRR:      {summary["mrr"]:.4f}')

    # 低分题
    low = [q for i, q in enumerate(questions) if metrics['recall@10'][i] < 0.3]
    if low:
        print(f'\nRecall@10 < 30%: {len(low)} 题')
        for q in low:
            print(f'  {q["id"]}: {q["question"]}')

    return summary, metrics


def main():
    total_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"PDF: {PDF_PATH}")
    print(f"Testset: {TESTSET_PATH}")

    # Stage 1: 摄入文档
    print("\n[1/4] Ingesting PDF...")
    t0 = time.time()

    # 解析
    parser = PyMuPDFParser()
    pages = parser.parse(PDF_PATH)

    # 分块 (chunk_size=256, 与测试集 gold mapping 一致)
    chunker = SentenceChunker(chunk_size=256, overlap=64)
    chunks = chunker.chunk(pages)
    print(f"  Parser: {len(pages)} pages, Chunker: {len(chunks)} chunks")

    # 嵌入 (GPU)
    embedder = BGEEmbedder(model_path=MODEL_PATH, device=DEVICE)
    chunks = embedder.embed_chunks(chunks)
    print(f"  Embedder: {embedder.dim}-dim, device={DEVICE}")

    # 存储
    store = ChromaStore(persist_dir=CHROMA_DIR)
    store.index(chunks)
    print(f"  Store: {store.count()} vectors in ChromaDB")

    print(f"  Ingestion done in {time.time()-t0:.1f}s")

    # Stage 2: 构建检索器
    print("\n[2/4] Building retrievers...")

    dense_retriever = DenseRetriever(store, embedder)
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(chunks)
    hybrid_retriever = HybridRetriever(store, embedder, rrf_k=60)
    hybrid_retriever.build_bm25_index(chunks)

    # 重排序器 (GPU)
    print("  Loading reranker...")
    t0 = time.time()
    reranker = BGEReranker(model_name="BAAI/bge-reranker-v2-m3", device=DEVICE)
    print(f"  Reranker loaded in {time.time()-t0:.1f}s")

    # 查询改写器 (需要 LLM)
    print("  Setting up query rewriter...")
    from docqa.generation.llm import OllamaLLM
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0.1, max_tokens=200, timeout=60)
    rewriter = QueryRewriter(llm.generate)
    mq_retriever = MultiQueryRetriever(hybrid_retriever, rewriter)

    # Stage 3: 加载测试集
    print("\n[3/4] Loading testset...")
    questions = load_testset(TESTSET_PATH)
    print(f"  {len(questions)} answerable questions (with relevant_chunk_ids)")

    # Stage 4: 评估
    print("\n[4/4] Evaluating...")

    # 方案 1: 纯向量
    def dense_search(query, top_k):
        return dense_retriever.search(query, top_k)

    dense_summary, _ = evaluate_method("1-Dense(纯向量/Baseline)", questions, dense_search, TOP_K_VALUES)

    # 方案 2: 纯 BM25
    def bm25_search(query, top_k):
        return bm25_retriever.search(query, top_k)

    bm25_summary, bm25_metrics = evaluate_method("2-BM25(关键词)", questions, bm25_search, TOP_K_VALUES)

    # 方案 3: 混合检索
    def hybrid_search(query, top_k):
        return hybrid_retriever.search(query, top_k)

    hybrid_summary, _ = evaluate_method("3-Hybrid(BM25+Dense RRF)", questions, hybrid_search, TOP_K_VALUES)

    # 方案 4: 混合 + 重排序
    def hybrid_rerank_search(query, top_k):
        candidates = hybrid_retriever.search(query, 50)
        return reranker.rerank(query, candidates, top_k)

    rerank_summary, _ = evaluate_method("4-Hybrid+Reranker(GPU)", questions, hybrid_rerank_search, TOP_K_VALUES)

    # 方案 5: 混合 + 多查询改写 + 重排序 (完整方案)
    def mq_rerank_search(query, top_k):
        # 多查询检索
        queries = rewriter.rewrite(query, n_variants=3)
        # Hybrid on each variant, collect unique
        seen = {}
        for q in queries:
            for c in hybrid_retriever.search(q, 50):
                if c.chunk_id not in seen:
                    seen[c.chunk_id] = c
        candidates = list(seen.values())
        # Rerank each variant, keep best score per chunk
        best_score = {}
        for q in queries:
            reranked = reranker.rerank(q, candidates, len(candidates))
            for c in reranked:
                s = c.metadata.get('rerank_score', 0)
                if c.chunk_id not in best_score or s > best_score[c.chunk_id][0]:
                    best_score[c.chunk_id] = (s, c)
        return sorted(
            [c for _, c in best_score.values()],
            key=lambda x: x.metadata.get('rerank_score', 0), reverse=True
        )[:top_k]

    mq_summary, _ = evaluate_method("5-Hybrid+MultiQuery+Rerank(完整方案)", questions, mq_rerank_search, TOP_K_VALUES)

    # ====== 汇总对比 ======
    all_methods = [
        ("1-Dense(Baseline)", dense_summary),
        ("2-BM25", bm25_summary),
        ("3-Hybrid", hybrid_summary),
        ("4-Hybrid+Rerank", rerank_summary),
        ("5-Full(+MultiQuery)", mq_summary),
    ]

    print(f'\n\n{"="*85}')
    print("对比汇总")
    print(f'{"="*85}')
    print(f'{"Method":<30} {"Recall@5":>9} {"Recall@10":>9} {"MRR":>7} {"NDCG@10":>8}')
    print("-" * 65)
    for name, summary in all_methods:
        r5, r10, mr, nd = summary['recall@5'], summary['recall@10'], summary['mrr'], summary['ndcg@10']
        # Best marker
        best_r5 = max(m[1]['recall@5'] for m in all_methods)
        best_r10 = max(m[1]['recall@10'] for m in all_methods)
        best_mrr = max(m[1]['mrr'] for m in all_methods)
        print(f'{name:<30} {r5:>8.2%}{" *" if r5==best_r5 else ""}'
              f'  {r10:>8.2%}{" *" if r10==best_r10 else ""}'
              f'  {mr:>6.4f}{" *" if mr==best_mrr else ""}'
              f'  {nd:>7.4f}')

    # 关键数据对比
    print(f'\n{"="*85}')
    print("关键指标 vs 目标")
    print(f'{"="*85}')
    targets = {"recall@5": 0.85, "recall@10": 0.90, "mrr": 0.75}
    for metric, target in targets.items():
        val = mq_summary[metric]
        status = "PASS" if val >= target else "MISS"
        print(f'{metric}: {val:.2%} (target >={target:.0%}) -> {status}')

    elapsed = time.time() - total_start
    print(f'\n总耗时: {elapsed/60:.1f} 分钟')


if __name__ == "__main__":
    main()
