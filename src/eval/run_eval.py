"""
src/eval/run_eval.py
Etap 4/5: ewaluacja na gold secie. Porównuje konfiguracje retrievalu „przed/po":
  dense (baseline) | hybrid (dense+BM25/RRF) | dense+rerank | hybrid+rerank

Co liczymy:
- RETRIEVAL (obiektywnie, z chunk_id): recall@k (k=1,3,5,10) i MRR — pytania z korpusu.
  Kluczowe odczyty: recall@10 (SUFIT — czy hybryda dobiera to, czego dense nie wyciągnął)
  oraz recall@1/MRR (jak wysoko — co poprawia rerank).
- ODMOWA (anty-halucynacja): pytania spoza korpusu — czy system poprawnie odmawia.

Uruchomienie (wymaga Qdrant + Ollama; reranker z cache):
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
from src.retrieve.hybrid import hybrid_search
from src.retrieve.search import search

K_VALUES = [1, 3, 5, 10]
RETRIEVE_N = max(K_VALUES)


def load_gold() -> list[dict]:
    return [json.loads(line) for line in config.GOLD_PATH.open(encoding="utf-8")]


# Konfiguracje: nazwa -> funkcja(pytanie) -> lista chunk_id (posortowana)
CONFIGS = {
    "dense":      lambda q: [h["chunk_id"] for h in search(q, top_k=RETRIEVE_N, rerank=False)],
    "hybrid":     lambda q: [h["chunk_id"] for h in hybrid_search(q, top_k=RETRIEVE_N, rerank=False)],
    "dense+rr":   lambda q: [h["chunk_id"] for h in search(q, top_k=RETRIEVE_N, rerank=True)],
    "hybrid+rr":  lambda q: [h["chunk_id"] for h in hybrid_search(q, top_k=RETRIEVE_N, rerank=True)],
}


def eval_cfg(in_corpus: list[dict], retrieve_fn) -> list[dict]:
    rows = []
    for g in in_corpus:
        ids = retrieve_fn(g["question"])
        row = {"type": g["type"], "rr": reciprocal_rank(ids, g["relevant_chunk_ids"])}
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
    return [{"question": g["question"],
             "refused": "nie znajduję odpowiedzi w źródłach" in answer(g["question"])["answer"].lower()}
            for g in ooc]


def main() -> None:
    gold = load_gold()
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]
    print(f"Gold set: {len(gold)} par ({len(in_corpus)} z korpusu, {len(ooc)} spoza).\n")

    results = {name: eval_cfg(in_corpus, fn) for name, fn in CONFIGS.items()}
    aggs = {name: agg(rows) for name, rows in results.items()}

    names = list(CONFIGS.keys())
    print("=== RETRIEVAL — porównanie konfiguracji ===")
    print("  " + f"{'metryka':10}" + "".join(f"{n:>11}" for n in names))
    for key in [f"recall@{k}" for k in K_VALUES] + ["MRR"]:
        print("  " + f"{key:10}" + "".join(f"{aggs[n][key]:>11.3f}" for n in names))

    print("\n  recall@5 wg typu (dense -> hybrid+rr):")
    types = sorted({r["type"] for r in results["dense"]})
    for t in types:
        d = mean([r["h@5"] for r in results["dense"] if r["type"] == t])
        h = mean([r["h@5"] for r in results["hybrid+rr"] if r["type"] == t])
        n = sum(1 for r in results["dense"] if r["type"] == t)
        print(f"    {t:16} {d:.2f} -> {h:.2f}  (n={n})")

    print("\n  poza top-10 (miss etapu wyszukiwania):")
    for name in names:
        miss = [g["id"] for g, r in zip(in_corpus, results[name]) if r["rr"] == 0.0]
        print(f"    {name:10}: " + (", ".join(miss) if miss else "— brak"))

    print("\n=== ODMOWA na pytaniach spoza korpusu ===")
    ref = eval_refusal(ooc)
    for r in ref:
        print(f"  [{'OK' if r['refused'] else '‼ ZMYŚLIŁ'}] {r['question']}")
    print(f"  refusal accuracy: {mean([1.0 if r['refused'] else 0.0 for r in ref]):.3f}")


if __name__ == "__main__":
    main()
