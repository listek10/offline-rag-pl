"""
src/eval/calibrate_gate.py
Kalibracja progu BRAMKI TRAFNOŚCI (`config.RELEVANCE_GATE_MIN`) na zbiorze HELD-OUT.

DLACZEGO OSOBNY ZBIÓR (to jest sedno tego pliku):
Pierwotnie próg 0.1 dobraliśmy patrząc na rozkład score'ów rerankera NA GOLD SECIE — czyli na
tym samym zbiorze, na którym raportujemy `refusal`. To dopasowanie hiperparametru do zbioru
testowego: raportowane `refusal = 1.00` przestawało być prognozą zachowania na NOWYM pytaniu,
a stawało się opisem tego, jak dobrze próg pasuje do pytań, które już widział.
Tutaj kalibrujemy na `gold_set/heldout_gate.jsonl`, którego NIGDY nie raportujemy — gold set
zostaje nietknięty i dopiero pomiar na nim jest uczciwy (out-of-sample).

CZEGO NIE MA W HELD-OUCIE i dlaczego: nie ma `answer` ani `relevant_chunk_ids`. Bramka patrzy
WYŁĄCZNIE na najwyższy score rerankera, więc pełna para gold byłaby zbędnym kosztem budowy.
Etykieta `label` (in_corpus / out_of_corpus) to wszystko, czego potrzebujemy.

CO REALNIE WYZNACZA PRÓG: nie pytania trywialnie dalekie ("ile to 37×12", score ~0.02) — te
spełnia każdy próg. Granicę wyznaczają przypadki najbliższe granicy decyzyjnej z obu stron:
  - najWYŻSZY score wśród pytań SPOZA korpusu (subtype `bliski-temat` = hard negatives:
    nasz region i tematyka, ale konkretu w 5 raportach nie ma)  -> dolna granica progu,
  - najNIŻSZY score wśród pytań Z korpusu (zwykle „pojęciowe", bo mają najsłabszy retrieval)
    -> górna granica progu.
Próg musi zmieścić się w przerwie między nimi. Jeśli przerwy NIE MA (rozkłady się nakładają),
to znaczy, że żaden pojedynczy próg na tej jednej liczbie nie rozdziela klas — skrypt to
wykrywa i mówi wprost, że trzeba wybrać punkt kompromisu wg asymetrii kosztów.

Uruchomienie (wymaga Qdrant + modeli; LLM nie jest potrzebny — bramka działa przed LLM):
    python src/eval/calibrate_gate.py
    python src/eval/calibrate_gate.py --cache   # użyj policzonych score'ów z results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.retrieve.hybrid import hybrid_search

HELDOUT_PATH = config.ROOT / "gold_set" / "heldout_gate.jsonl"
OUT_PATH = config.ROOT / "results" / "gate_calibration.json"

# Siatka progów do przemiatania. Cały zakres score'a rerankera (sigmoid, ~0..1) — gdyby siatka
# kończyła się wcześniej, „największy próg bez false-refusal" byłby artefaktem obcięcia siatki,
# a nie właściwością danych. Krok 0.005 — drobniejszy niż różnice, które realnie rozstrzygamy.
GRID = [round(0.005 * i, 3) for i in range(0, 201)]   # 0.000 ... 1.000


def load_heldout() -> list[dict]:
    rows = [json.loads(l) for l in HELDOUT_PATH.open(encoding="utf-8") if l.strip()]
    bad = [r["id"] for r in rows if r["label"] not in ("in_corpus", "out_of_corpus")]
    if bad:
        raise ValueError(f"Zła etykieta w held-oucie: {bad}")
    return rows


def top_gate_score(question: str) -> float:
    """Liczy DOKŁADNIE ten sygnał, na który patrzy bramka w `answer()`.

    Ta zgodność jest krytyczna: gdybyśmy kalibrowali na innym torze retrievalu (np. samo dense
    albo bez rerankera), próg nie przenosiłby się na produkcję. Dlatego wołamy tę samą funkcję
    z tymi samymi parametrami z configu.
    """
    hits = hybrid_search(question, rerank=True)
    return max((h.get("rerank_score", 0.0) for h in hits), default=0.0)


def compute_scores(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows, 1):
        s = top_gate_score(r["question"])
        print(f"  [{i:2}/{len(rows)}] {r['id']} {r['label']:14} {r['subtype']:14} "
              f"top_score={s:7.4f}  {r['question'][:56]}")
        out.append({**r, "top_score": s})
    return out


def sweep(scored: list[dict]) -> list[dict]:
    """Dla każdego progu-kandydata liczy DWIE stopy błędu (bo bramka myli się w dwie strony).

    - leak  (przeciek): pytanie SPOZA korpusu przeszło bramkę -> LLM dostanie nietrafny kontekst
      i może zmyślić. W kontekście analitycznym to błąd droższy.
    - false_refusal:    pytanie Z korpusu zostało ubite -> system odmawia, choć MA odpowiedź.
      Tego nie wolno pomijać: próg 999 daje zero przecieków i bezużyteczny system.
    """
    ins = [r for r in scored if r["label"] == "in_corpus"]
    ooc = [r for r in scored if r["label"] == "out_of_corpus"]
    table = []
    for t in GRID:
        leaks = [r["id"] for r in ooc if r["top_score"] >= t]          # przepuszczone OOC
        frs = [r["id"] for r in ins if r["top_score"] < t]             # ubite IN
        table.append({
            "threshold": t,
            "leak_rate": len(leaks) / len(ooc) if ooc else 0.0,
            "false_refusal_rate": len(frs) / len(ins) if ins else 0.0,
            "leaks": leaks,
            "false_refusals": frs,
        })
    return table


def summarize(scored: list[dict]) -> dict:
    ins = [r for r in scored if r["label"] == "in_corpus"]
    ooc = [r for r in scored if r["label"] == "out_of_corpus"]
    min_in = min(ins, key=lambda r: r["top_score"])
    max_ooc = max(ooc, key=lambda r: r["top_score"])
    gap = min_in["top_score"] - max_ooc["top_score"]

    table = sweep(scored)
    # Największy próg, przy którym NIC z korpusu nie zostaje ubite (najmniej przecieków
    # pod warunkiem false_refusal == 0). To „agresywna" granica — bez zapasu bezpieczeństwa.
    safe = [row for row in table if row["false_refusal_rate"] == 0.0]
    max_safe = max((row["threshold"] for row in safe), default=0.0)
    # Rekomendacja: środek przerwy. Zapas z obu stron, bo held-out to tylko próbka —
    # nowe pytanie może wypaść trochę niżej (IN) lub trochę wyżej (OOC) niż nasze ekstrema.
    midpoint = round((min_in["top_score"] + max_ooc["top_score"]) / 2, 4) if gap > 0 else None

    return {
        "n_in_corpus": len(ins),
        "n_out_of_corpus": len(ooc),
        "min_in": {"id": min_in["id"], "subtype": min_in["subtype"],
                   "score": min_in["top_score"], "question": min_in["question"]},
        "max_ooc": {"id": max_ooc["id"], "subtype": max_ooc["subtype"],
                    "score": max_ooc["top_score"], "question": max_ooc["question"]},
        "gap": round(gap, 4),
        "separable": gap > 0,
        "max_threshold_without_false_refusal": max_safe,
        "recommended_threshold": midpoint,
        "current_threshold": config.RELEVANCE_GATE_MIN,
        "sweep": table,
    }


def report(scored: list[dict], s: dict) -> None:
    print("\n=== ROZKŁAD top-score PER SUBTYPE (posortowane) ===")
    for label in ("in_corpus", "out_of_corpus"):
        subs = sorted({r["subtype"] for r in scored if r["label"] == label})
        for sub in subs:
            vals = sorted((r["top_score"] for r in scored
                           if r["label"] == label and r["subtype"] == sub))
            rng = f"{vals[0]:.4f} … {vals[-1]:.4f}"
            print(f"  {label:14} {sub:14} n={len(vals):2}  {rng}")

    print("\n=== GRANICA DECYZYJNA (to ona wyznacza próg) ===")
    print(f"  najniższy IN   {s['min_in']['score']:.4f}  [{s['min_in']['id']}, "
          f"{s['min_in']['subtype']}] {s['min_in']['question'][:60]}")
    print(f"  najwyższy OOC  {s['max_ooc']['score']:.4f}  [{s['max_ooc']['id']}, "
          f"{s['max_ooc']['subtype']}] {s['max_ooc']['question'][:60]}")
    print(f"  PRZERWA = {s['gap']:+.4f}  -> {'rozdzielalne' if s['separable'] else 'ROZKŁADY SIĘ NAKŁADAJĄ'}")

    print("\n=== PRZEMIATANIE PROGU (tylko punkty, w których coś się zmienia) ===")
    print(f"  {'próg':>7}  {'przecieki':>10}  {'false-ref':>10}")
    prev = None
    for row in s["sweep"]:
        key = (row["leak_rate"], row["false_refusal_rate"])
        if key != prev:
            print(f"  {row['threshold']:7.3f}  {row['leak_rate']:10.2f}  {row['false_refusal_rate']:10.2f}")
            prev = key

    print("\n=== WNIOSEK ===")
    print(f"  próg obecny w configu:                {s['current_threshold']}")
    print(f"  największy próg bez false-refusal:    {s['max_threshold_without_false_refusal']}")
    if s["separable"]:
        print(f"  REKOMENDACJA (środek przerwy):        {s['recommended_threshold']}")
        print("  Zapas z obu stron, bo held-out to próbka — nowe pytanie może wypaść poza nasze ekstrema.")
    else:
        print("  Rozkłady się NAKŁADAJĄ: nie istnieje próg z zerem przecieków i zerem false-refusali.")
        print("  Opcje: (a) punkt kompromisu wg asymetrii kosztów (w analityce przeciek droższy),")
        print("         (b) dwa progi = strefa szara, w której nadal pytamy LLM, ale ostrzej,")
        print("         (c) mocniejszy sygnał (np. średnia top-3 zamiast max, + pokrycie leksykalne).")


def e2e_refusal(scored: list[dict]) -> dict:
    """POMIAR END-TO-END trzech warstw obrony — OBIE strony błędu naraz.

    Rozkład score'ów mówi tylko o bramce. Ale przeciek przez bramkę nie oznacza automatycznie
    halucynacji (LLM z kontekstem „na temat, ale bez odpowiedzi" może sam odmówić), a guard #3
    może z kolei ubić poprawną PARAFRAZĘ. Dlatego dla każdej konfiguracji mierzymy jednocześnie:
      - refusal na 18 pytaniach SPOZA korpusu (ile złego zatrzymujemy),
      - false-refusal na 12 pytaniach Z korpusu (ile dobrego przy tym tracimy).
    Bez drugiej kolumny pierwsza jest bez znaczenia — próg 999 daje refusal 1.00 i bezużyteczny system.
    """
    from src.generate.answer import answer, refused

    ooc = [r for r in scored if r["label"] == "out_of_corpus"]
    ins = [r for r in scored if r["label"] == "in_corpus"]
    configs = [
        ("sam prompt (guard #1)", False, False),
        (f"+ bramka {config.RELEVANCE_GATE_MIN} (#2)", True, False),
        ("+ weryfikacja tez (#3)", True, True),
    ]
    print(f"\n=== END-TO-END na held-oucie: {len(ooc)} pytań OOC + {len(ins)} z korpusu ===")
    print(f"  {'konfiguracja':30} {'refusal OOC':>12} {'false-ref IN':>13}   szczegóły")
    results = {}
    for label, gate, verify in configs:
        leaks, frs = [], []
        for r in ooc:
            if not refused(answer(r["question"], gate=gate, verify=verify)["answer"]):
                leaks.append(f"{r['id']}({r['subtype'][:6]})")
        for r in ins:
            res = answer(r["question"], gate=gate, verify=verify)
            if refused(res["answer"]):
                frs.append(f"{r['id']}({res.get('blocked_by', 'prompt')})")
        refusal = (len(ooc) - len(leaks)) / len(ooc)
        fr = len(frs) / len(ins)
        print(f"  {label:30} {refusal:12.2f} {fr:13.2f}   przecieki: {', '.join(leaks) or '—'}"
              f"{'  | FR: ' + ', '.join(frs) if frs else ''}")
        results[label] = {"refusal_ooc": refusal, "false_refusal_in": fr,
                          "leaks": leaks, "false_refusals": frs}
    return results


def main() -> None:
    use_cache = "--cache" in sys.argv
    if use_cache and OUT_PATH.exists():
        scored = json.loads(OUT_PATH.read_text(encoding="utf-8"))["scored"]
        print(f"(z cache: {OUT_PATH})")
    else:
        rows = load_heldout()
        print(f"Held-out: {len(rows)} pytań z {HELDOUT_PATH.name}")
        print("Liczę top-score rerankera tym samym torem co produkcja (hybryda + rerank)...\n")
        scored = compute_scores(rows)

    s = summarize(scored)
    report(scored, s)

    e2e = e2e_refusal(scored) if "--e2e" in sys.argv else None

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps({"scored": scored, "summary": s, "e2e": e2e},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH}")


if __name__ == "__main__":
    main()
