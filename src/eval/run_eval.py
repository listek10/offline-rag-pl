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

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.eval.metrics import hit_at_k, mean, reciprocal_rank, relevant_by_evidence
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


def truth_map(in_corpus: list[dict], truth: str) -> dict[str, list[str]]:
    """Zwraca id pytania -> lista trafnych chunk_id, wg wybranej definicji prawdy.

    `evidence` wyznacza trafne chunki NA ŻYWO z aktualnego pliku chunków, więc działa dla
    dowolnego podziału tekstu; `chunk_ids` bierze je z gold setu (ważne tylko dla podziału,
    przy którym gold set powstał). Patrz docstring `metrics.py`.
    """
    if truth == "chunk_ids":
        return {g["id"]: g["relevant_chunk_ids"] for g in in_corpus}

    chunks = [json.loads(l) for l in config.CHUNKS_PATH.open(encoding="utf-8") if l.strip()]
    mapa = {}
    for g in in_corpus:
        cyt = g.get("evidence")
        if not cyt:
            sys.exit(f"[BŁĄD] {g['id']} nie ma pola `evidence` — dodaj je albo użyj --truth chunk_ids.")
        trafne = relevant_by_evidence(chunks, cyt)
        if not trafne:
            # Cytat, którego nie ma w ŻADNYM chunku, to nie „zero trafień" — to zepsuty pomiar
            # (np. wariant chunkingu rozciął zdanie). Musi być widoczny, nie wliczony jako pudło.
            print(f"  [UWAGA] {g['id']}: cytat nie mieści się w żadnym chunku tego wariantu "
                  f"-> pytanie NIEMOŻLIWE do trafienia przy tym podziale")
        mapa[g["id"]] = trafne
    return mapa


def eval_cfg(in_corpus: list[dict], retrieve_fn, prawda: dict[str, list[str]]) -> list[dict]:
    rows = []
    for g in in_corpus:
        ids = retrieve_fn(g["question"])
        relevant = prawda[g["id"]]
        row = {"id": g["id"], "type": g["type"], "rr": reciprocal_rank(ids, relevant),
               "n_relevant": len(relevant)}
        for k in K_VALUES:
            row[f"h@{k}"] = hit_at_k(ids, relevant, k)
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
    ap = argparse.ArgumentParser(description="Ewaluacja retrievalu na gold secie")
    ap.add_argument("--truth", choices=["chunk_ids", "evidence"], default="chunk_ids",
                    help="definicja trafienia: id z gold setu (domyślnie) albo cytat-dowód")
    ap.add_argument("--json", dest="json_out", help="zapisz metryki do pliku JSON")
    ap.add_argument("--skip-refusal", action="store_true",
                    help="pomiń pomiar odmowy (wymaga LLM-a; niepotrzebny przy A/B chunkingu)")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    help="ogranicz do wybranych konfiguracji retrievalu")
    args = ap.parse_args()

    gold = load_gold()
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]
    print(f"Gold set: {len(gold)} par ({len(in_corpus)} z korpusu, {len(ooc)} spoza).")
    print(f"Prawda: {args.truth} | chunki: {config.CHUNKS_PATH.name} | "
          f"kolekcja: {config.QDRANT_COLLECTION}\n")

    prawda = truth_map(in_corpus, args.truth)
    wybrane = {n: CONFIGS[n] for n in (args.configs or CONFIGS)}
    results = {name: eval_cfg(in_corpus, fn, prawda) for name, fn in wybrane.items()}
    aggs = {name: agg(rows) for name, rows in results.items()}
    CONFIGS_USED = wybrane

    names = list(CONFIGS_USED)
    print("=== RETRIEVAL — porównanie konfiguracji ===")
    print("  " + f"{'metryka':10}" + "".join(f"{n:>11}" for n in names))
    for key in [f"recall@{k}" for k in K_VALUES] + ["MRR"]:
        print("  " + f"{key:10}" + "".join(f"{aggs[n][key]:>11.3f}" for n in names))

    # Rozbicie po typie pytania robimy między NAJSŁABSZĄ i NAJLEPSZĄ z policzonych konfiguracji —
    # przy `--configs` nie ma gwarancji, że „dense" i „hybrid+rr" w ogóle były liczone.
    pierwsza, ostatnia = names[0], names[-1]
    print(f"\n  recall@5 wg typu ({pierwsza} -> {ostatnia}):")
    for t in sorted({r["type"] for r in results[pierwsza]}):
        d = mean([r["h@5"] for r in results[pierwsza] if r["type"] == t])
        h = mean([r["h@5"] for r in results[ostatnia] if r["type"] == t])
        n = sum(1 for r in results[pierwsza] if r["type"] == t)
        print(f"    {t:16} {d:.2f} -> {h:.2f}  (n={n})")

    print("\n  poza top-10 (miss etapu wyszukiwania):")
    for name in names:
        miss = [r["id"] for r in results[name] if r["rr"] == 0.0]
        print(f"    {name:10}: " + (", ".join(miss) if miss else "— brak"))

    ref = None
    if args.skip_refusal:
        print("\n(pomiar odmowy pominięty: --skip-refusal)")
    else:
        print("\n=== ODMOWA na pytaniach spoza korpusu ===")
        ref = eval_refusal(ooc)
        for r in ref:
            print(f"  [{'OK' if r['refused'] else '‼ ZMYŚLIŁ'}] {r['question']}")
        print(f"  refusal accuracy: {mean([1.0 if r['refused'] else 0.0 for r in ref]):.3f}")

    if args.json_out:
        # Zapisujemy też liczbę trafnych chunków per pytanie: przy definicji `evidence` mówi,
        # ile „biletów" dał dany podział (zakładka duplikuje fragmenty) — bez tego nie da się
        # uczciwie zinterpretować różnic między wariantami.
        Path(args.json_out).write_text(json.dumps({
            "truth": args.truth,
            "chunks_path": str(config.CHUNKS_PATH),
            "collection": config.QDRANT_COLLECTION,
            "chunk_target_chars": config.CHUNK_TARGET_CHARS,
            "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
            "n_in_corpus": len(in_corpus),
            "aggregates": aggs,
            "per_question": results,
            "refusal": ref,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nZapisano metryki -> {args.json_out}")


if __name__ == "__main__":
    main()
