import math

from bm25 import BM25
from evaluate import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from index import InvertedIndex
from tokenizer import tokenize

CORPUS = {
    "d1": "insulin controls blood sugar",
    "d2": "cats sleep all day",
    "d3": "blood pressure and sugar",
}


def make_index():
    index = InvertedIndex()
    index.build(CORPUS)
    return index


def test_tokenizer_stems_and_drops_stopwords():
    assert tokenize("Running cats") == ["run", "cat"]
    assert "the" not in tokenize("the cat")


def test_index_counts():
    index = make_index()
    assert index.num_docs == 3
    assert index.doc_freq("blood") == 2
    assert index.get_postings("insulin") == {"d1": 1}
    assert index.avg_doc_len == (4 + 3 + 3) / 3


def test_index_save_load(tmp_path):
    index = make_index()
    index.save(tmp_path / "idx.pkl")
    loaded = InvertedIndex.load(tmp_path / "idx.pkl")
    assert loaded.postings == index.postings


def test_bm25_ranks_matching_doc_first():
    bm25 = BM25(make_index())
    results = bm25.search("insulin", 3)
    assert results[0][0] == "d1"
    assert len(results) == 1  # only d1 contains the term


def test_bm25_rare_term_beats_common_term():
    bm25 = BM25(make_index())
    assert bm25.idf("insulin") > bm25.idf("blood")


def test_metrics():
    assert ndcg_at_k(["a", "b"], {"a": 1}) == 1.0
    assert math.isclose(ndcg_at_k(["b", "a"], {"a": 1}), 1 / math.log2(3))
    assert reciprocal_rank(["b", "a"], {"a": 1}) == 0.5
    assert precision_at_k(["a", "b"], {"a": 1}, k=2) == 0.5
    assert recall_at_k(["a", "b"], {"a": 1, "c": 1}, k=2) == 0.5
