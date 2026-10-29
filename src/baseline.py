import math
from collections import defaultdict

from tokenizer import tokenize


class TfIdf:
    """Classic TF-IDF with cosine similarity, built on the same inverted index."""

    def __init__(self, index):
        self.index = index
        self.idf = {
            t: math.log(index.num_docs / len(p)) for t, p in index.postings.items()
        }
        norms = defaultdict(float)
        for term, plist in index.postings.items():
            for doc_id, tf in plist.items():
                w = (1 + math.log(tf)) * self.idf[term]
                norms[doc_id] += w * w
        self.norms = {d: math.sqrt(v) for d, v in norms.items()}

    def search(self, query, top_k=10):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            qw = self.idf[term]
            for doc_id, tf in self.index.get_postings(term).items():
                scores[doc_id] += qw * (1 + math.log(tf)) * self.idf[term]
        for d in scores:
            scores[d] /= self.norms[d]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
