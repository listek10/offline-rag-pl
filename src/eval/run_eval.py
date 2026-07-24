"""
src/eval/run_eval.py
Etap 4/5: ewaluacja na gold secie. Liczy BASELINE (dense) i — dla porównania „przed/po" —
wariant z RERANKINGIEM (checkpoint #6).

Co liczymy:
- RETRIEVAL (obiektywnie, z chunk_id): recall@k (k=1,3,5,10) i MRR — na pytaniach z korpusu,
  osobno dla dense i dense+rerank, z różnicą Δ. Plus rozbicie recall@5 wg typu pytania.
- ODMOWA (anty-halucynacja): dla pytań spoza korpusu — czy system poprawnie odmawia.

Uruchomienie (wymaga Qdrant + Ollama; pierwszy rerank pobierze model ~2,3 GB):
    python src/eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.eval.metrics import hit_at_k, mean, reciprocal_rank
from src.retrieve.search import search

K_VALUES = [1, 3, 5, 10]
RETRIEVE_N = max(K_VALUES)


def load_gold() -> list[dict]:
    return [json.loads(line) for line in config.GOLD_PATH.open(encoding="utf-8")]


def eval_retrieval(in_corpus: list[dict], rerank: bool) -> list[dict]:
    rows = []
    for g in in_corpus:
        hits = search(g["question"], top_k=RETRIEVE_N, rerank=rerank)
        ids = [h["chunk_id"] for h in hits]
        row = {"id": g["id"], "type": g["type"], "rr": reciprocal_rank(ids, g["relevant_chunk_ids"])}
        for k in K_VALUES:
            row[f"h@{k}"] = hit_at_k(ids, g["relevant_chunk_ids"], k)
        rows.append(row)
    return rows


def agg(rows: list[dict]) -> dict:
    m = {f"recall@{k}": mean([r[f"h@{k}"] for r in rows]) for k in K_VALUES}
    m["MRR"] = mean([r["rr"] for r in rows])
    return m


def eval_refusal(ooc: list[dict]) -> list[dict]:
    from src.generate.answer import answer
    rows = []
    for g in ooc:
        resp = answer(g["question"])["answer"]
        rows.append({"question": g["question"],
                     "refused": "nie znajduję odpowiedzi w źródłach" in resp.lower(),
                     "resp": resp})
    return rows


def main() -> None:
    gold = load_gold()
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]
    print(f"Gold set: {len(gold)} par ({len(in_corpus)} z korpusu, {len(ooc)} spoza).\n")

    dense = eval_retrieval(in_corpus, rerank=False)
    reranked = eval_retrieval(in_corpus, rerank=True)
    md, mr = agg(dense), agg(reranked)

    # --- Tabela przed/po ---
    print("=== RETRIEVAL: dense (baseline)  vs  dense+rerank ===")
    print(f"  {'metryka':10} {'dense':>8} {'rerank':>8} {'Δ':>8}")
    for key in [f"recall@{k}" for k in K_VALUES] + ["MRR"]:
        d, r = md[key], mr[key]
        print(f"  {key:10} {d:8.3f} {r:8.3f} {r-d:+8.3f}")

    # --- Rozbicie recall@5 wg typu ---
    print("\n  recall@5 wg typu (dense -> rerank):")
    for t in sorted({r["type"] for r in dense}):
        d = mean([r["h@5"] for r in dense if r["type"] == t])
        r = mean([r["h@5"] for r in reranked if r["type"] == t])
        n = sum(1 for x in dense if x["type"] == t)
        print(f"    {t:16} {d:.2f} -> {r:.2f}  (n={n})")

    # --- Miss (poza top-10) ---
    for label, rows in [("dense", dense), ("rerank", reranked)]:
        miss = [r["id"] for r in rows if r["rr"] == 0.0]
        print(f"  poza top-10 ({label}): " + (", ".join(miss) if miss else "— brak"))

    # --- Odmowa ---
    print("\n=== ODMOWA na pytaniach spoza korpusu ===")
    for r in eval_refusal(ooc):
        print(f"  [{'OK' if r['refused'] else '‼ ZMYŚLIŁ'}] {r['question']}  ->  {r['resp'][:60]}")


if __name__ == "__main__":
    main()
