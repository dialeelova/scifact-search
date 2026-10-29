from baseline import TfIdf
from bm25 import BM25
from data import load_scifact
from evaluate import evaluate, print_results
from index import InvertedIndex

corpus, queries, qrels = load_scifact()
index = InvertedIndex()
index.build(corpus)

for name, engine in [("TF-IDF", TfIdf(index)), ("BM25", BM25(index))]:
    results, _ = evaluate(engine.search, queries, qrels)
    print_results(name, results)
