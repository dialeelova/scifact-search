from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path("data/dense_embeddings.npy")


class DenseRetriever:
    def __init__(self, corpus):
        self.doc_ids = list(corpus)
        self.model = SentenceTransformer(MODEL)
        if CACHE.exists():
            self.emb = np.load(CACHE)
        else:
            texts = [corpus[d] for d in self.doc_ids]
            self.emb = self.model.encode(
                texts, batch_size=64, show_progress_bar=True,
                normalize_embeddings=True,
            )
            np.save(CACHE, self.emb)

    def search(self, query, top_k=10):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    from data import load_scifact
    from evaluate import evaluate, print_results

    corpus, queries, qrels = load_scifact()
    dense = DenseRetriever(corpus)
    results, _ = evaluate(dense.search, queries, qrels)
    print_results("Dense", results)
