import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from tokenizer import tokenize


class InvertedIndex:
    def __init__(self):
        self.postings = {}    # term -> {doc_id: term frequency}
        self.doc_len = {}     # doc_id -> number of tokens
        self.num_docs = 0
        self.avg_doc_len = 0.0

    def build(self, corpus):
        postings = defaultdict(dict)
        for doc_id, text in corpus.items():
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term][doc_id] = tf
        self.postings = dict(postings)
        self.num_docs = len(self.doc_len)
        self.avg_doc_len = sum(self.doc_len.values()) / self.num_docs

    def get_postings(self, term):
        return self.postings.get(term, {})

    def doc_freq(self, term):
        return len(self.postings.get(term, {}))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        index = cls()
        with open(path, "rb") as f:
            index.__dict__.update(pickle.load(f))
        return index


if __name__ == "__main__":
    from data import load_scifact

    corpus, queries, qrels = load_scifact()

    start = time.time()
    index = InvertedIndex()
    index.build(corpus)
    print(f"Built in {time.time() - start:.1f}s")
    print(f"{index.num_docs} docs, {len(index.postings)} unique terms, "
          f"avg length {index.avg_doc_len:.1f} tokens")

    index.save("data/index.pkl")
    index = InvertedIndex.load("data/index.pkl")
    print("Saved and reloaded OK")

    term = tokenize("insulin")[0]
    print(f"'{term}' appears in {index.doc_freq(term)} docs, e.g.:",
          dict(list(index.get_postings(term).items())[:3]))
