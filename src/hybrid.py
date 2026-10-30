from collections import defaultdict


class RRFHybrid:
    """Reciprocal Rank Fusion of BM25 and dense rankings."""

    def __init__(self, bm25, dense, k=60, depth=100):
        self.bm25, self.dense, self.k, self.depth = bm25, dense, k, depth

    def search(self, query, top_k=10):
        scores = defaultdict(float)
        for engine in (self.bm25, self.dense):
            for rank, (doc_id, _) in enumerate(engine.search(query, self.depth), 1):
                scores[doc_id] += 1.0 / (self.k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class WeightedHybrid:
    """Min-max normalised score blend: alpha * BM25 + (1 - alpha) * dense."""

    def __init__(self, bm25, dense, alpha=0.5, depth=100):
        self.bm25, self.dense, self.alpha, self.depth = bm25, dense, alpha, depth

    @staticmethod
    def _norm(results):
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        return {d: (s - lo) / span for d, s in results}

    def search(self, query, top_k=10):
        b = self._norm(self.bm25.search(query, self.depth))
        d = self._norm(self.dense.search(query, self.depth))
        combined = {
            doc: self.alpha * b.get(doc, 0.0) + (1 - self.alpha) * d.get(doc, 0.0)
            for doc in set(b) | set(d)
        }
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
