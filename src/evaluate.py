import math
import time


def precision_at_k(ranked, relevant, k=10):
    return sum(1 for d in ranked[:k] if d in relevant) / k


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(ranked, relevant, k=10):
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked, relevant, k=100):
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def evaluate(search_fn, queries, qrels):
    """search_fn(query_text, top_k) -> list of (doc_id, score)."""
    totals = {"P@10": 0.0, "nDCG@10": 0.0, "MRR": 0.0, "Recall@100": 0.0}
    per_query = {}
    start = time.time()
    for q_id, text in queries.items():
        ranked = [d for d, _ in search_fn(text, 100)]
        rel = qrels[q_id]
        m = {
            "P@10": precision_at_k(ranked, rel),
            "nDCG@10": ndcg_at_k(ranked, rel),
            "MRR": reciprocal_rank(ranked, rel),
            "Recall@100": recall_at_k(ranked, rel),
        }
        per_query[q_id] = m
        for k, v in m.items():
            totals[k] += v
    n = len(queries)
    results = {k: v / n for k, v in totals.items()}
    results["ms/query"] = (time.time() - start) / n * 1000
    return results, per_query


def print_results(name, results):
    print(f"{name:<12} " + "  ".join(f"{k}={v:.4f}" for k, v in results.items()))


if __name__ == "__main__":
    from bm25 import BM25
    from data import load_scifact
    from index import InvertedIndex

    corpus, queries, qrels = load_scifact()
    index = InvertedIndex()
    index.build(corpus)
    bm25 = BM25(index)

    results, _ = evaluate(bm25.search, queries, qrels)
    print_results("BM25", results)
