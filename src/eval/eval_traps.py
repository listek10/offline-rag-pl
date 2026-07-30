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


def _krotka(path: Path) -> str:
    """Ścieżka względem repo, ale bez wyjątku dla plików spoza repo (testy podają katalog tmp)."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


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
    print(f"\nZapisano {len(rows)} odpowiedzi: {_krotka(ANSWERS)}")


# Jednostki, których korpus używa w skrócie („25 tys.", „0,9 mln km2"), a model rozwija w słowa.
# Lista jest ROZSZERZENIEM produkcyjnego `_UNIT`, mierzonym jako wariant — patrz `compare()`.
_UNIT_SZEROKI = (r"(?:%|proc\.|mld|mln|tys\.|tysięcy|tysiące|tysiąca|tysiąc"
                 r"|miliard\w*|milion\w*|km2|km²)")


class _units:
    """Podmienia wzorzec jednostki w module guardu na czas jednego wariantu pomiaru.

    `_quantity_positions` i `quantities` budują swoje wyrażenia przy KAŻDYM wywołaniu, czytając
    `_UNIT` z modułu — więc podmiana globalu wystarcza i nie trzeba duplikować logiki guardu
    w harnessie (duplikat rozjechałby się z produkcją przy pierwszej zmianie).
    """

    def __init__(self, unit: str) -> None:
        self.unit = unit

    def __enter__(self) -> None:
        from src.generate import attribution
        self.orig = attribution._UNIT
        attribution._UNIT = self.unit

    def __exit__(self, *exc: object) -> None:
        from src.generate import attribution
        attribution._UNIT = self.orig


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


def audyt() -> dict:
    """Czy pytanie W OGÓLE mogło zmierzyć to, po co je dodałem — na ZAPISANYCH kontekstach.

    DLACZEGO TO OSOBNY KROK (i dlaczego bez niego liczby kłamią): „podmiana nie zaszła" ma dwa
    zupełnie różne znaczenia. Albo model MIAŁ w kontekście cudzą liczbę i jej nie wziął (sukces),
    albo cudzej liczby tam nie było, bo retrieval jej nie przyniósł — i wtedy pytanie nie było
    pułapką, tylko zwykłym pytaniem. Wliczanie takich pytań do mianownika zaniża awaryjność
    tym mocniej, im większy zbiór. Ta sama logika co `value_match`, który wyrzuca z mianownika
    pytania bez liczby.

    Dla guardu #4 mierzymy drugi warunek: czy guard MA PODSTAWĘ do werdyktu (nie abstynuje).
    Dokładnie to zabiło pomiar z etapu 10 — 6 z 7 pytań było poza zasięgiem guardu.
    """
    from src.eval.metrics import extract_values

    traps = {t["id"]: t for t in load_traps()}
    rows = json.loads(ANSWERS.read_text(encoding="utf-8"))["rows"]
    out = {}
    print(f"=== AUDYT WARUNKU WSTĘPNEGO na {len(rows)} pytaniach ===")
    print(f"  {'id':5} {'podmiot':11} {'cudza-w-kontekście':19} {'prawda-w-kontekście':20} guard#4")
    for r in rows:
        t = traps[r["id"]]
        w_kontekscie = extract_values(r["context"])
        cudze = sorted(set(t["trap_values"]) & w_kontekscie)
        prawda = sorted(extract_values(t["answer"]) & w_kontekscie)
        werdykt = guard4_blocks(r)
        ma_podstawe = check_attribution(r["question"], r["answer"], r["context"])[0] is not None
        out[r["id"]] = {"cudze_wartosci_w_kontekscie": cudze, "prawda_w_kontekscie": prawda,
                        "guard4_ma_podstawe": ma_podstawe, "guard4_blokuje": werdykt[0]}
        print(f"  {r['id']:5} {r['entity']:11} {str(cudze) or '—':19} {str(prawda) or '—':20} "
              f"{'werdykt' if ma_podstawe else 'ABSTYNENCJA'}")

    # DETEKTOR PYTANIA NIEJEDNOZNACZNEGO — wykrył realny błąd w moich danych (t08, t10).
    # Sygnał: odpowiedź wypadła jako `inna-bledna` (żadnej znanej wartości), a guard #4 mówi,
    # że podana liczba NALEŻY do pytanego podmiotu. Znaczy to zwykle, że korpus ma dla tego
    # podmiotu DRUGĄ prawdziwą liczbę, której nie wpisałem do `answer` — czyli metryka karze
    # model za wybór innego prawdziwego zdania. To błąd zbioru, nie systemu, i musi być widoczny.
    from src.eval.metrics import classify_trap_answer
    podejrzane = []
    for r in rows:
        t = traps[r["id"]]
        if r["refused_by_model"]:
            continue
        klasa = classify_trap_answer(t["answer"], t["trap_values"], r["answer"])
        zgodne = [d for d in check_attribution(r["question"], r["answer"], r["context"])[1]
                  if d.get("ocena") == "zgodna"]
        out[r["id"]]["klasa_bez_guardow"] = klasa
        if klasa == "inna-bledna" and zgodne:
            podejrzane.append(r["id"])
    if podejrzane:
        print(f"\n  [UWAGA] pytania prawdopodobnie NIEJEDNOZNACZNE (klasa `inna-bledna`, ale guard #4"
              f" uznaje podaną liczbę za należącą do pytanego podmiotu): {podejrzane}")
        print("          sprawdź, czy korpus nie ma DRUGIEJ prawdziwej wartości dla tego pytania")

    pulapki = [k for k, v in out.items() if v["cudze_wartosci_w_kontekscie"]]
    odpowiadalne = [k for k, v in out.items() if v["prawda_w_kontekscie"]]
    w_zasiegu = [k for k, v in out.items() if v["guard4_ma_podstawe"]]
    print(f"\n  REALNE pułapki (cudza liczba w kontekście):        {len(pulapki)}/{len(rows)}  {pulapki}")
    print(f"  odpowiadalne (prawda w kontekście):               {len(odpowiadalne)}/{len(rows)}")
    print(f"  w zasięgu guardu #4 (ma podstawę do werdyktu):     {len(w_zasiegu)}/{len(rows)}  {w_zasiegu}")
    print(f"  ROZSTRZYGAJĄCE dla guardu (pułapka I w zasięgu):   "
          f"{len(set(pulapki) & set(w_zasiegu))}/{len(rows)}  {sorted(set(pulapki) & set(w_zasiegu))}")
    path = RESULTS / "entity_traps_audyt.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {_krotka(path)}")
    return out


def compare(repeats: int) -> dict:
    traps = {t["id"]: t for t in load_traps()}
    data = json.loads(ANSWERS.read_text(encoding="utf-8"))
    rows = data["rows"]

    # Guard #4 policzony RAZ — jest deterministyczny.
    g4 = {r["id"]: guard4_blocks(r) for r in rows}

    # WARIANT „szersze jednostki": audyt pokazał, że część abstynencji guardu #4 NIE wynika
    # z jego projektu (anafora), a z dziury w liście jednostek — korpus pisze „25 tys.",
    # a model odpowiada „25 tysięcy" i „900 km2". Guard widzi wtedy odpowiedź BEZ wielkości
    # i milczy, choć ma wszystko, żeby orzec. Mierzę to jako osobny wariant, a nie po cichu
    # w produkcyjnym module: rozszerzenie jednostek może dokładać BLOKADY na poprawnych
    # odpowiedziach, więc musi mieć zmierzone oba kierunki błędu, jak każda zmiana guardu.
    with _units(_UNIT_SZEROKI):
        g4w = {r["id"]: guard4_blocks(r) for r in rows}

    # Guard #3 powtórzony `repeats` razy, bo sędzia jest stochastyczny.
    g3: dict[str, list[bool]] = {r["id"]: [] for r in rows}
    for rep in range(repeats):
        for r in rows:
            g3[r["id"]].append(False if r["refused_by_model"] else guard3_blocks(r))
        print(f"  guard #3: powtórzenie {rep + 1}/{repeats} gotowe")

    warianty = {
        "bez guardów post":    lambda r, rep: False,
        "guard #3 (sędzia)":   lambda r, rep: g3[r["id"]][rep],
        "guard #4 (przypis.)": lambda r, rep: g4[r["id"]][0],
        "guard #4 + jednostki": lambda r, rep: g4w[r["id"]][0],
        "guard #3 + #4":       lambda r, rep: g3[r["id"]][rep] or g4[r["id"]][0],
        "guard #3 + #4 + jedn.": lambda r, rep: g3[r["id"]][rep] or g4w[r["id"]][0],
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
        blocked4w, detw = g4w[r["id"]]
        g3_blocks = sum(g3[r["id"]])
        powod = next((f"{d['wartość']} -> {d['przypisana']}" for d in detw
                      if d.get("ocena") == "PODMIANA PODMIOTU"), "—")
        print(f"  {r['id']} {r['entity']:10} model_odmowil={str(r['refused_by_model']):5} "
              f"#3_blokuje={g3_blocks}/{repeats} #4={str(blocked4):5} #4+jedn.={str(blocked4w):5} {powod}")
        out["per_pytanie"][r["id"]] = {"entity": r["entity"],
                                       "refused_by_model": r["refused_by_model"],
                                       "guard3_blokad": g3_blocks, "guard4_blokuje": blocked4,
                                       "guard4_jednostki_blokuje": blocked4w,
                                       "guard4_szczegoly": det,
                                       "guard4_jednostki_szczegoly": detw, "answer": r["answer"]}

    path = RESULTS / "entity_traps_compare.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {_krotka(path)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pomiar podmiany podmiotu (parowany).")
    ap.add_argument("--prepare", action="store_true", help="wygeneruj odpowiedzi RAZ")
    ap.add_argument("--compare", action="store_true", help="oceń warianty guardów na zapisanych")
    ap.add_argument("--audyt", action="store_true",
                    help="czy pytania W OGÓLE mogły zmierzyć podmianę (bez LLM-a)")
    ap.add_argument("--repeats", type=int, default=3, help="powtórzenia dla stochastycznego guardu #3")
    args = ap.parse_args()

    if args.prepare:
        print(f"Generacja RAZ: {config.LLM_MODEL} ({config.LLM_BACKEND})\n")
        prepare()
    if args.audyt or args.compare:
        if not ANSWERS.exists():
            sys.exit(f"Brak {ANSWERS} — uruchom najpierw --prepare")
    if args.audyt:
        audyt()
    if args.compare:
        compare(args.repeats)
    if not (args.prepare or args.compare or args.audyt):
        ap.error("podaj --prepare, --audyt albo --compare")


if __name__ == "__main__":
    main()
