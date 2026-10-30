import json

from bm25 import BM25
from data import load_scifact
from dense import DenseRetriever
from index import InvertedIndex
from tokenizer import tokenize

corpus, queries, qrels = load_scifact()
pq = json.load(open("results/per_query.json"))

def failed(name):
    return {q for q, m in pq[name].items() if m["nDCG@10"] == 0}

bm_f, de_f, hy_f = failed("BM25"), failed("Dense"), failed("Hybrid-W0.5")
both = bm_f & de_f
only_bm = bm_f - de_f
only_de = de_f - bm_f

print(f"Queries with nDCG@10 = 0 (out of {len(queries)}):")
print(f"  BM25 fails:    {len(bm_f)}")
print(f"  Dense fails:   {len(de_f)}")
print(f"  Both fail:     {len(both)}")
print(f"  Only BM25:     {len(only_bm)}  (dense rescues these)")
print(f"  Only dense:    {len(only_de)}  (BM25 rescues these)")
print(f"  Hybrid fails:  {len(hy_f)}")

index = InvertedIndex()
index.build(corpus)
bm25 = BM25(index)
dense = DenseRetriever(corpus)

def rank_of(engine, text, rel):
    for i, (d, _) in enumerate(engine.search(text, 100), 1):
        if d in rel:
            return i
    return None

def block(title, ids, n):
    out = [f"## {title}\n"]
    for q in sorted(ids)[:n]:
        text, rel = queries[q], qrels[q]
        target = next(iter(rel))
        top_b = bm25.search(text, 1)[0][0]
        top_d = dense.search(text, 1)[0][0]
        out.append(f"**Query {q}:** {text}\n")
        out.append(f"- Correct doc {target}: {corpus[target][:200]}...")
        out.append(f"- Query tokens after stemming: {tokenize(text)}")
        out.append(f"- BM25 rank of correct doc: {rank_of(bm25, text, rel)}; top result: {corpus[top_b][:120]}...")
        out.append(f"- Dense rank of correct doc: {rank_of(dense, text, rel)}; top result: {corpus[top_d][:120]}...\n")
    return "\n".join(out)

lines = [
    "# Failure analysis (BEIR SciFact, 300 test queries)\n",
    f"- BM25 scores zero on {len(bm_f)} queries, dense on {len(de_f)}, both on {len(both)}.",
    f"- Dense rescues {len(only_bm)} BM25 failures; BM25 rescues {len(only_de)} dense failures.",
    f"- Hybrid (weighted) still fails on {len(hy_f)}.\n",
    "Categories to assign by hand after reading the examples below: "
    "vocabulary mismatch, numbers or negation, very short or vague claim, "
    "claim needs reasoning, other.\n",
    block("Both methods fail", both, 4),
    block("BM25 fails, dense succeeds", only_bm, 4),
    block("Dense fails, BM25 succeeds", only_de, 4),
]
open("results/failure_analysis.md", "w", encoding="utf-8").write("\n".join(lines))
print("\nWrote results/failure_analysis.md")
