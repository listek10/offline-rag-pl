"""
src/eval/run_eval.py
Etap 4 / checkpoint #11: uruchamia ewaluację na gold secie i wypisuje BASELINE.

Co liczymy:
- RETRIEVAL (obiektywnie, z chunk_id): recall@k dla k=1,3,5,10 oraz MRR — na pytaniach z korpusu.
  Dodatkowo rozbicie recall@5 wg TYPU pytania (faktograficzne/nazwa-własna/parafraza...) — to
  pokazuje, gdzie dense jest słaby (np. nazwy własne) i uzasadnia przyszłą hybrydę/rerank.
- ODMOWA (anty-halucynacja): dla pytań spoza korpusu sprawdzamy, czy system poprawnie odmawia
  ("Nie znajduję odpowiedzi w źródłach"). To tani, obiektywny wskaźnik. (Woła LLM — kilka pytań.)

Faithfulness/answer-relevance (LLM-sędzia) świadomie pomijamy na start (subiektywne, cięższe).

Uruchomienie (wymaga Qdrant + Ollama):
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


def eval_retrieval(in_corpus: list[dict]) -> list[dict]:
    rows = []
    for g in in_corpus:
        ids = [h["chunk_id"] for h in search(g["question"], top_k=RETRIEVE_N)]
        row = {"id": g["id"], "type": g["type"], "rr": reciprocal_rank(ids, g["relevant_chunk_ids"])}
        for k in K_VALUES:
            row[f"h@{k}"] = hit_at_k(ids, g["relevant_chunk_ids"], k)
        rows.append(row)
    return rows


def eval_refusal(ooc: list[dict]) -> list[dict]:
    from src.generate.answer import answer  # import tu, żeby retrieval-only nie ładował LLM
    rows = []
    for g in ooc:
        resp = answer(g["question"])["answer"]
        refused = "nie znajduję odpowiedzi w źródłach" in resp.lower()
        rows.append({"id": g["id"], "question": g["question"], "refused": refused, "resp": resp})
    return rows


def main() -> None:
    gold = load_gold()
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]

    print(f"Gold set: {len(gold)} par ({len(in_corpus)} z korpusu, {len(ooc)} spoza).\n")

    # --- Retrieval ---
    rows = eval_retrieval(in_corpus)
    print("=== RETRIEVAL (dense baseline) ===")
    for k in K_VALUES:
        print(f"  recall@{k}: {mean([r[f'h@{k}'] for r in rows]):.3f}")
    print(f"  MRR:       {mean([r['rr'] for r in rows]):.3f}")

    print("\n  recall@5 wg typu pytania:")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["h@5"])
    for t, xs in sorted(by_type.items()):
        print(f"    {t:16} {mean(xs):.2f}  (n={len(xs)})")

    print("\n  Pytania, gdzie trafny chunk NIE trafił do top-10 (MRR=0):")
    misses = [r["id"] for r in rows if r["rr"] == 0.0]
    print("    " + (", ".join(misses) if misses else "— brak, wszystkie trafione"))

    # --- Odmowa (anty-halucynacja) ---
    print("\n=== ODMOWA na pytaniach spoza korpusu ===")
    ref = eval_refusal(ooc)
    for r in ref:
        mark = "OK" if r["refused"] else "‼ ZMYŚLIŁ"
        print(f"  [{mark}] {r['question']}  ->  {r['resp'][:70]}")
    print(f"  refusal accuracy: {mean([1.0 if r['refused'] else 0.0 for r in ref]):.3f}")


if __name__ == "__main__":
    main()
