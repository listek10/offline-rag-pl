"""
src/eval/eval_traps.py
Pomiar PODMIANY PODMIOTU — najgroźniejszego znanego trybu awarii tego systemu.

DLACZEGO OSOBNY HARNESS, A NIE KOLEJNA LINIA W eval_answers.py:
istniejące metryki odpowiedzi są na tę awarię ŚLEPE. Dla pytania „wydatki obronne Danii"
pipeline podaje 1,75% PKB (liczbę Norwegii), a `faithfulness` liczy to jako wierne (sędzia
głosuje TAK), `false-refusal` jako poprawne (system odpowiedział), `relevance` jako na temat.
Najpoważniejszy tryb awarii liczył się więc jako SUKCES. Tu prawdą jest WARTOŚĆ, sprawdzana
deterministycznie względem `gold_set/entity_traps.jsonl` — bez sędziego, bo sędzia jest właśnie
tym, co w tej awarii zawodzi.

CO MIERZYMY (cztery klasy + odmowa, nie jedna liczba):
  poprawna        — jest wartość właściwego podmiotu, nie ma wartości innego
  PODMIANA        — jest wartość INNEGO podmiotu, nie ma właściwej  <- klasa niebezpieczna
  niejednoznaczna — obie wartości (zwykle poprawna, rozwlekła odpowiedź)
  inna-bledna     — żadnej znanej wartości
  odmowa          — system odmówił (guard #1/#2/#3)
Guard może tylko PRZESUNĄĆ pytanie z `PODMIANA` do `odmowa` — nie umie poprawić odpowiedzi.
Dlatego raportujemy obie kolumny osobno: zamiana pewnego kłamstwa na odmowę jest zyskiem,
ale odmowa na pytaniu, na które korpus ma odpowiedź, jest kosztem. Sumowanie ich w jedną
„poprawę" byłoby oszukiwaniem samego siebie.

DLACZEGO POWTÓRZENIA: guard #3 jest stochastyczny (to samo pytanie raz przechodzi, raz nie —
zmierzone: h19 przeciekł 28.07, w powtórce został złapany; h18 odwrotnie). Pojedynczy przebieg
nie jest dowodem o KONKRETNYM pytaniu, tylko rozkład na całym zbiorze. Domyślnie 3 powtórzenia.

Uruchomienie (wymaga Qdrant + Ollama):
    python src/eval/eval_traps.py --label baseline-gate           # bez guardu #3
    python src/eval/eval_traps.py --label baseline-verify --verify
    python src/eval/eval_traps.py --label po-entity --verify --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.eval.metrics import classify_trap_answer
from src.generate.answer import answer, refused

TRAPS_PATH = config.ROOT / "gold_set" / "entity_traps.jsonl"
RESULTS = config.ROOT / "results"
CLASSES = ["poprawna", "niejednoznaczna", "podmiana", "inna-bledna", "odmowa"]


def load_traps() -> list[dict]:
    return [json.loads(line) for line in TRAPS_PATH.open(encoding="utf-8")]


def classify_one(trap: dict, text: str) -> str:
    """Odmowa ma pierwszeństwo: jeśli system odmówił, nie ma czego klasyfikować wartościowo."""
    if refused(text):
        return "odmowa"
    return classify_trap_answer(trap["answer"], trap["trap_values"], text)


def run(traps: list[dict], gate: bool, verify: bool, repeats: int) -> list[dict]:
    rows = []
    for rep in range(1, repeats + 1):
        for t in traps:
            res = answer(t["question"], gate=gate, verify=verify)
            text = res["answer"]
            klasa = classify_one(t, text)
            rows.append({"rep": rep, "id": t["id"], "entity": t["entity"],
                         "subtype": t["subtype"], "klasa": klasa,
                         "blocked_by": res.get("blocked_by"), "answer": text})
            flag = "  <-- PODMIANA" if klasa == "podmiana" else ""
            print(f"  [rep{rep}] {t['id']} {t['entity']:10} {klasa:16}{flag}")
    return rows


def report(rows: list[dict], repeats: int, n_traps: int) -> dict:
    cnt = Counter(r["klasa"] for r in rows)
    total = len(rows)
    print(f"\n=== ROZKŁAD na {total} przebiegach ({n_traps} pytań x {repeats} powtórzeń) ===")
    for k in CLASSES:
        print(f"  {k:16} {cnt[k]:3}/{total}  = {cnt[k]/total:.3f}")

    # Stabilność per pytanie: guard #3 jest stochastyczny, więc pytanie może raz wpaść
    # w PODMIANĘ, a raz nie. Bez tej kolumny raportowałbym średnią, która ukrywa wariancję.
    print(f"\n=== STABILNOŚĆ per pytanie (ile z {repeats} powtórzeń dało którą klasę) ===")
    per_q: dict[str, Counter] = {}
    for r in rows:
        per_q.setdefault(r["id"], Counter())[r["klasa"]] += 1
    for qid in sorted(per_q):
        c = per_q[qid]
        opis = ", ".join(f"{k}x{v}" for k, v in c.most_common())
        stab = "STABILNE" if len(c) == 1 else "NIESTABILNE"
        print(f"  {qid}  {stab:12} {opis}")

    return {"n_traps": n_traps, "repeats": repeats,
            "rozkład": {k: cnt[k] for k in CLASSES},
            "udziały": {k: round(cnt[k] / total, 3) for k in CLASSES},
            "per_pytanie": {q: dict(c) for q, c in per_q.items()},
            "przebiegi": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pomiar podmiany podmiotu na zbiorze pułapek.")
    ap.add_argument("--label", required=True, help="nazwa pomiaru (do pliku w results/)")
    ap.add_argument("--repeats", type=int, default=3, help="powtórzenia (guard #3 jest stochastyczny)")
    ap.add_argument("--no-gate", action="store_true", help="wyłącz bramkę trafności (guard #2)")
    ap.add_argument("--verify", action="store_true", help="włącz guard #3 (weryfikacja ugruntowania)")
    args = ap.parse_args()

    traps = load_traps()
    print(f"Pułapki: {len(traps)} pytań  |  gate={not args.no_gate}  verify={args.verify}  "
          f"powtórzeń={args.repeats}")
    print(f"Generator: {config.LLM_MODEL} ({config.LLM_BACKEND})  |  sędzia: {config.JUDGE_MODEL}\n")

    rows = run(traps, gate=not args.no_gate, verify=args.verify, repeats=args.repeats)
    summary = report(rows, args.repeats, len(traps))
    summary["config"] = {"gate": not args.no_gate, "verify": args.verify,
                         "model": config.LLM_MODEL, "judge": config.JUDGE_MODEL,
                         "backend": config.LLM_BACKEND}

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"entity_traps_{args.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
