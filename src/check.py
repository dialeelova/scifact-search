from rank_bm25 import BM25Okapi

from bm25 import BM25
from data import load_scifact
from index import InvertedIndex
from tokenizer import tokenize

corpus, queries, qrels = load_scifact()
index = InvertedIndex()
index.build(corpus)
mine = BM25(index, k1=1.5, b=0.75)

doc_ids = list(corpus)
lib = BM25Okapi([tokenize(corpus[d]) for d in doc_ids], k1=1.5, b=0.75)

same_top1, overlap = 0, 0.0
sample = list(queries.items())[:50]
for q_id, text in sample:
    my_top = [d for d, _ in mine.search(text, 10)]
    scores = lib.get_scores(tokenize(text))
    order = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)[:10]
    lib_top = [doc_ids[i] for i in order]
    same_top1 += my_top[0] == lib_top[0]
    overlap += len(set(my_top) & set(lib_top)) / 10

print(f"Same top-1 result: {same_top1}/{len(sample)}")
print(f"Average top-10 overlap: {overlap / len(sample):.2%}")
