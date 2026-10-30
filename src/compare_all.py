import json

from baseline import TfIdf
from bm25 import BM25
from data import load_scifact
from dense import DenseRetriever
from evaluate import evaluate, print_results
from hybrid import RRFHybrid, WeightedHybrid
from index import InvertedIndex

corpus, queries, qrels = load_scifact()
index = InvertedIndex()
index.build(corpus)
bm25 = BM25(index)
dense = DenseRetriever(corpus)

engines = [
    ("TF-IDF", TfIdf(index)),
    ("BM25", bm25),
    ("Dense", dense),
    ("Hybrid-RRF", RRFHybrid(bm25, dense)),
    ("Hybrid-W0.5", WeightedHybrid(bm25, dense, alpha=0.5)),
]

all_results, all_per_query = {}, {}
for name, engine in engines:
    results, per_query = evaluate(engine.search, queries, qrels)
    print_results(name, results)
    all_results[name] = results
    all_per_query[name] = per_query

with open("results/metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)
with open("results/per_query.json", "w") as f:
    json.dump(all_per_query, f)
print("\nSaved results/metrics.json and results/per_query.json")
