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

═══ POMIAR JEST PAROWANY — i to nie jest kosmetyka, a warunek poprawności ═══
Generujemy odpowiedzi RAZ (`--prepare`), a potem oceniamy na NICH warianty guardów
(`--compare`). Powód jest zmierzony: temperatura 0 daje dekodowanie zachłanne, ale NIE
powtarzalne. Pytanie t02 (Norwegia) odpowiadało poprawnie 3/3, a po przerwie odmawiało 3/3
przy IDENTYCZNYM retrievalu (top_score 0,9964, te same 5 chunków). Wykluczone pomiarem:
flaga `verify` nie dotyka generacji, zimne przeładowanie modelu nie przywraca zachowania,
kolejność żądań nie tłumaczy. Zostaje niedeterminizm redukcji zmiennoprzecinkowych na GPU —
przy pytaniu brzegowym przewraca argmax, a z nim całą odpowiedź.
Gdybyśmy generowali osobno dla każdego wariantu guardu, różnica „przed/po" mieszałaby efekt
guardu z tym szumem. Ten sam wzorzec jest już użyty w `eval_answers.py` („guard #3 symulowany
na TEJ SAMEJ odpowiedzi") i w `compare_backends.py` (`--prepare` / `--compare`).

CO MIERZYMY (cztery klasy + odmowa, nie jedna liczba):
  poprawna        — jest wartość właściwego podmiotu, nie ma wartości innego
  PODMIANA        — jest wartość INNEGO podmiotu, nie ma właściwej  <- klasa niebezpieczna
  niejednoznaczna — obie wartości (zwykle poprawna, rozwlekła odpowiedź)
  inna-bledna     — żadnej znanej wartości
  odmowa          — system odmówił (guard #1) albo guard #2/#3/#4 zablokował
Guard może tylko PRZESUNĄĆ pytanie z `PODMIANA` do `odmowa` — nie umie poprawić odpowiedzi.
Dlatego raportujemy obie kolumny osobno: zamiana pewnego kłamstwa na odmowę jest zyskiem,
ale odmowa na pytaniu, na które korpus MA odpowiedź, jest kosztem. Sumowanie ich w jedną
„poprawę" byłoby oszukiwaniem samego siebie.

DLACZEGO POWTÓRZENIA TYLKO DLA GUARDU #3: on jest stochastyczny (sędzia LLM). Guard #4
(przypisanie) jest deterministyczny, więc jedno przejście wystarcza i to jest jego zaleta.

Uruchomienie (wymaga Qdrant + Ollama):
    python src/eval/eval_traps.py --prepare                 # generacja RAZ -> plik z odpowiedziami
    python src/eval/eval_traps.py --compare --repeats 3      # warianty guardów na tych odpowiedziach
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
from src.generate.answer import answer, build_context, refused
from src.generate.attribution import check_attribution

TRAPS_PATH = config.ROOT / "gold_set" / "entity_traps.jsonl"
RESULTS = config.ROOT / "results"
ANSWERS = RESULTS / "entity_traps_answers.json"
CLASSES = ["poprawna", "niejednoznaczna", "podmiana", "inna-bledna", "odmowa"]


def load_traps() -> list[dict]:
    return [json.loads(line) for line in TRAPS_PATH.open(encoding="utf-8")]


def prepare() -> None:
    """Generuje odpowiedzi RAZ, z guardami POST-generacyjnymi wyłączonymi (verify=False).

    Bramka (guard #2) zostaje włączona, bo jest PRZED generacją — nie zmienia odpowiedzi,
    tylko decyduje, czy w ogóle powstanie. Zapisujemy też kontekst, żeby warianty guardów
    liczyły się na dokładnie tym, co widział model.
    """
    traps = load_traps()
    rows = []
    for t in traps:
        res = answer(t["question"], gate=True, verify=False)
        rows.append({"id": t["id"], "question": t["question"], "entity": t["entity"],
                     "subtype": t["subtype"], "answer": res["answer"],
                     "context": build_context(res["sources"]),
                     "refused_by_model": refused(res["answer"])})
        print(f"  {t['id']} {t['entity']:10} {'ODMOWA' if refused(res['answer']) else 'odpowiedź'}: "
              f"{' '.join(res['answer'].split())[:80]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ANSWERS.write_text(json.dumps({"model": config.LLM_MODEL, "backend": config.LLM_BACKEND,
                                   "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano {len(rows)} odpowiedzi: {ANSWERS.relative_to(config.ROOT)}")


def guard3_blocks(row: dict) -> bool:
    """Guard #3 na ZAPISANEJ odpowiedzi — bez powtórnej generacji (stochastyczny: sędzia LLM)."""
    from src.generate.verify import grounded_fraction
    return grounded_fraction(row["context"], row["answer"])[0] < config.VERIFY_MIN_SUPPORTED


def guard4_blocks(row: dict) -> tuple[bool, list[dict]]:
    """Guard #4 na ZAPISANEJ odpowiedzi — deterministyczny, bez LLM-a.

    `None` (abstynencja) NIE blokuje: guard poza swoim zakresem oddaje decyzję guardowi #3.
    """
    ok, details = check_attribution(row["question"], row["answer"], row["context"])
    return ok is False, details


def classify(trap: dict, row: dict, blocked: bool) -> str:
    if blocked or row["refused_by_model"]:
        return "odmowa"
    return classify_trap_answer(trap["answer"], trap["trap_values"], row["answer"])


def compare(repeats: int) -> dict:
    traps = {t["id"]: t for t in load_traps()}
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    rows = data["rows"]

    # Guard #4 policzony RAZ — jest deterministyczny.
    g4 = {r["id"]: guard4_blocks(r) for r in rows}

    # Guard #3 powtórzony `repeats` razy, bo sędzia jest stochastyczny.
    g3: dict[str, list[bool]] = {r["id"]: [] for r in rows}
    for rep in range(repeats):
        for r in rows:
            g3[r["id"]].append(False if r["refused_by_model"] else guard3_blocks(r))
        print(f"  guard #3: powtórzenie {rep + 1}/{repeats} gotowe")

    warianty = {
        "bez guardów post":   lambda r, rep: False,
        "guard #3 (sędzia)":  lambda r, rep: g3[r["id"]][rep],
        "guard #4 (przypis.)": lambda r, rep: g4[r["id"]][0],
        "guard #3 + #4":      lambda r, rep: g3[r["id"]][rep] or g4[r["id"]][0],
    }

    out = {"repeats": repeats, "n_traps": len(rows), "warianty": {}, "per_pytanie": {}}
    print(f"\n=== WARIANTY GUARDÓW na TYCH SAMYCH {len(rows)} odpowiedziach "
          f"({repeats} powtórzeń dla stochastycznego #3) ===")
    print(f"  {'wariant':22} {'poprawna':>9} {'niejedn.':>9} {'PODMIANA':>9} {'inna-bł.':>9} {'odmowa':>8}")
    for name, blocks in warianty.items():
        cnt = Counter()
        for rep in range(repeats):
            for r in rows:
                cnt[classify(traps[r["id"]], r, blocks(r, rep))] += 1
        total = repeats * len(rows)
        out["warianty"][name] = {"licznik": {k: cnt[k] for k in CLASSES},
                                 "udziały": {k: round(cnt[k] / total, 3) for k in CLASSES}}
        print(f"  {name:22} {cnt['poprawna']:9} {cnt['niejednoznaczna']:9} "
              f"{cnt['podmiana']:9} {cnt['inna-bledna']:9} {cnt['odmowa']:8}")

    print(f"\n=== PER PYTANIE (dlaczego — guard ma być audytowalny) ===")
    for r in rows:
        blocked4, det = g4[r["id"]]
        g3_blocks = sum(g3[r["id"]])
        powod = next((f"{d['wartość']} -> {d['przypisana']}" for d in det
                      if d.get("ocena") == "PODMIANA PODMIOTU"), "—")
        print(f"  {r['id']} {r['entity']:10} model_odmowil={str(r['refused_by_model']):5} "
              f"#3_blokuje={g3_blocks}/{repeats} #4_blokuje={str(blocked4):5} {powod}")
        out["per_pytanie"][r["id"]] = {"entity": r["entity"],
                                       "refused_by_model": r["refused_by_model"],
                                       "guard3_blokad": g3_blocks, "guard4_blokuje": blocked4,
                                       "guard4_szczegoly": det, "answer": r["answer"]}

    path = RESULTS / "entity_traps_compare.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {path.relative_to(config.ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pomiar podmiany podmiotu (parowany).")
    ap.add_argument("--prepare", action="store_true", help="wygeneruj odpowiedzi RAZ")
    ap.add_argument("--compare", action="store_true", help="oceń warianty guardów na zapisanych")
    ap.add_argument("--repeats", type=int, default=3, help="powtórzenia dla stochastycznego guardu #3")
    args = ap.parse_args()

    if args.prepare:
        print(f"Generacja RAZ: {config.LLM_MODEL} ({config.LLM_BACKEND})\n")
        prepare()
    if args.compare:
        if not ANSWERS.exists():
            sys.exit(f"Brak {ANSWERS} — uruchom najpierw --prepare")
        compare(args.repeats)
    if not (args.prepare or args.compare):
        ap.error("podaj --prepare albo --compare")


if __name__ == "__main__":
    main()
