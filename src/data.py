import csv
import json
from pathlib import Path

def load_scifact(path="data/scifact"):
    path = Path(path)

    corpus = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d["text"]).strip()

    qrels = {}
    with open(path / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])

    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["_id"] in qrels:
                queries[d["_id"]] = d["text"]

    return corpus, queries, qrels

if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    print(len(corpus), "documents,", len(queries), "queries")
    for doc_id in list(corpus)[:5]:
        print("DOC", doc_id, corpus[doc_id][:100])
    for q_id in list(queries)[:5]:
        print("QUERY", q_id, queries[q_id], "->", qrels[q_id])
