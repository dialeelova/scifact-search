# SciFact Search

A small search engine built from scratch in Python, tested on scientific claims.

**Result:** BM25 beats a simple baseline (TF-IDF) and a small AI model. Mixing BM25 with the AI model gives the best score.

| Method | nDCG@10 | MRR | Recall@100 | ms/query |
|---|---|---|---|---|
| TF-IDF (baseline) | 0.650 | 0.605 | 0.935 | 2.5 |
| BM25 (from scratch) | 0.695 | 0.657 | 0.931 | 2.1 |
| Dense (MiniLM AI model) | 0.645 | 0.605 | 0.925 | 14.0 |
| Hybrid (rank fusion) | 0.717 | 0.685 | 0.955 | 15.8 |
| Hybrid (score mix, 50/50) | **0.729** | **0.698** | **0.955** | 15.9 |

Tested on 300 questions from [BEIR SciFact](https://github.com/beir-cellar/beir) (5,183 paper abstracts). Higher is better. The 50/50 mix was not tuned. All numbers: `results/metrics.json`.

## What is in this project

- A tokenizer (lowercase, remove stopwords, stemming)
- An inverted index (word -> documents), saved to disk
- BM25 ranking and a TF-IDF baseline
- A dense retriever (an AI model that matches meaning, not just words)
- Two ways to combine BM25 and dense
- Evaluation code (nDCG@10, MRR, Recall@100, P@10)

## Run it

```bash
git clone https://github.com/dialeelova/scifact-search.git
cd scifact-search
python -m venv .venv
.venv\Scripts\activate        # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python src/compare_all.py     # prints the table
python -m pytest -q           # runs the tests
```

The first run takes about 5 minutes (it computes AI embeddings for all documents). Later runs are fast.

## Checks

- **Is the BM25 correct?** It gives the same top result as the `rank_bm25` library on 48 of 50 queries.
- **Does tuning help?** No. k1 and b were tuned on 100 queries and tested on the other 200. The tuned values did not beat the defaults, so the defaults were kept.

## Where it fails

BM25 gets 43 of 300 questions completely wrong, the AI model 62, and only 33 are missed by both. The two methods make different mistakes, which is why mixing them helps.

- **BM25 is better** when the question has exact names (genes, proteins, drugs).
- **The AI model is better** when the question uses different words than the paper.
- **Both struggle** with negation ("is not predictive") and with papers that do not state the detail.

Examples and details: [results/failure_analysis.md](results/failure_analysis.md)

## Limits

- Only one dataset.
- Failure types come from 12 hand-read examples.
- Ideas for later: a tokenizer for gene names (`Bcl2` vs `BCL-2`), and a reranker.
