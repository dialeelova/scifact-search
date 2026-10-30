# SciFact Search: BM25, dense and hybrid retrieval from scratch

**Takeaway:** A BM25 engine built from scratch reaches nDCG@10 = 0.695 on BEIR SciFact, beating TF-IDF (0.650) and a MiniLM dense retriever (0.645); a hybrid of BM25 and dense reaches 0.729 because the two methods fail on different queries.

| Method | P@10 | nDCG@10 | MRR | Recall@100 | ms/query |
|---|---|---|---|---|---|
| TF-IDF (cosine) | 0.0907 | 0.6501 | 0.6054 | 0.9349 | 2.5 |
| BM25 (from scratch) | 0.0923 | 0.6954 | 0.6567 | 0.9309 | 2.1 |
| Dense (all-MiniLM-L6-v2) | 0.0883 | 0.6451 | 0.6047 | 0.9250 | 14.0 |
| Hybrid (RRF) | 0.0943 | 0.7168 | 0.6847 | 0.9550 | 15.8 |
| Hybrid (weighted, alpha=0.5) | 0.0950 | 0.7285 | 0.6975 | 0.9550 | 15.9 |

All numbers are averages over the 300 SciFact test queries (CPU only). The weighted hybrid uses a fixed alpha = 0.5 that was not tuned. Raw results: `results/metrics.json`.

## What this is

A small search engine with no search library: tokenizer (lowercase, stopwords, Porter stemming), inverted index with save/load, BM25 ranking, evaluation metrics, a TF-IDF baseline, a dense retriever, and two hybrid fusion methods. It is evaluated on [BEIR SciFact](https://github.com/beir-cellar/beir) (5,183 abstracts, 300 claims with human relevance labels).

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/scifact-search.git
cd scifact-search
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python src/compare_all.py       # prints the table above
python -m pytest -q             # 6 tests
```

The first run of `compare_all.py` embeds the 5,183 documents with MiniLM, which takes about 5 minutes on a laptop CPU; the embeddings are cached in `data/` afterwards.

## Checks

- **Correctness:** against `rank_bm25` (same parameters) my BM25 returns the same top-1 document on 48 of 50 queries, with 93% average top-10 overlap (`python src/check.py`). The gap comes from a slightly different IDF formula.
- **Tuning:** k1 and b were tuned on 100 queries and tested on the other 200. The tuned values (k1=2.1, b=0.9) scored 0.7084 nDCG@10 versus 0.7117 for the defaults (k1=1.5, b=0.75), so the defaults were kept (`python src/tune.py`).
- **Index:** 5,183 documents, 26,078 unique terms, built in about 1 second.

## Failure analysis

BM25 scores zero nDCG@10 on 43 of 300 queries and the dense model on 62; only 33 fail on both. BM25 rescues 29 dense failures and dense rescues 10 BM25 failures, so exact terms (gene and drug names) matter more than meaning on scientific claims. Hybrid still fails on 44 queries: its gain comes mainly from better ranking. Failures group into vocabulary mismatch, spelling and tokenization (e.g. `Bcl2` vs `BCL-2`), rare exact terms, negation, right topic but wrong paper, and details missing from the abstract.

Full write-up with examples: [results/failure_analysis.md](results/failure_analysis.md).

## Repo layout
src/tokenizer.py tokenizer and stemming
src/index.py inverted index, save/load
src/bm25.py BM25 ranking
src/baseline.py TF-IDF baseline
src/dense.py MiniLM dense retriever
src/hybrid.py RRF and weighted fusion
src/evaluate.py P@10, nDCG@10, MRR, Recall@100
src/compare_all.py runs every method, writes results/
src/tune.py k1/b tuning on a held-out split
src/check.py cross-check against rank_bm25
src/failures.py failure analysis
tests/ unit tests

## Limitations

- One dataset (SciFact); results may not transfer to other collections.
- Failure categories come from 12 hand-read examples, not all failing queries.
- The tokenizer is general-purpose, not biomedical-aware.
- No query expansion or cross-encoder reranking. Both are natural next steps.
