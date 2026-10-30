import random

from bm25 import BM25
from data import load_scifact
from evaluate import evaluate, print_results
from index import InvertedIndex

corpus, queries, qrels = load_scifact()
index = InvertedIndex()
index.build(corpus)

ids = sorted(queries)
random.Random(42).shuffle(ids)
tune_ids, test_ids = ids[:100], ids[100:]
tune_q = {i: queries[i] for i in tune_ids}
test_q = {i: queries[i] for i in test_ids}

best = None
print("Tuning on 100 queries (nDCG@10):")
for k1 in [0.6, 0.9, 1.2, 1.5, 1.8, 2.1]:
    for b in [0.3, 0.5, 0.75, 0.9]:
        res, _ = evaluate(BM25(index, k1, b).search, tune_q, qrels)
        print(f"  k1={k1} b={b}: {res['nDCG@10']:.4f}")
        if best is None or res["nDCG@10"] > best[0]:
            best = (res["nDCG@10"], k1, b)

_, k1, b = best
print(f"\nBest on tuning set: k1={k1}, b={b}")
print("\nHeld-out 200 test queries:")
res, _ = evaluate(BM25(index).search, test_q, qrels)
print_results("default", res)
res, _ = evaluate(BM25(index, k1, b).search, test_q, qrels)
print_results("tuned", res)
