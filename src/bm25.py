import math
from collections import defaultdict

from tokenizer import tokenize


class BM25:
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term):
        n = self.index.doc_freq(term)
        N = self.index.num_docs
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def search(self, query, top_k=10):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for doc_id, tf in self.index.get_postings(term).items():
                dl = self.index.doc_len[doc_id]
                norm = self.k1 * (1 - self.b + self.b * dl / self.index.avg_doc_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    from data import load_scifact
    from index import InvertedIndex

    corpus, queries, qrels = load_scifact()
    index = InvertedIndex()
    index.build(corpus)
    bm25 = BM25(index)

    for q_id in list(queries)[:3]:
        print("\nQUERY:", queries[q_id])
        print("CORRECT DOCS:", list(qrels[q_id]))
        for rank, (doc_id, score) in enumerate(bm25.search(queries[q_id], 5), 1):
            mark = "  <-- correct" if doc_id in qrels[q_id] else ""
            print(f"  {rank}. {doc_id} ({score:.2f}) {corpus[doc_id][:70]}{mark}")
