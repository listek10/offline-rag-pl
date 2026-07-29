"""
src/eval/ab_claims.py
Etap 13 / checkpoint #8: A/B DEFINICJI TEZY — zamknięcie dziury fail-open w guardzie #3
(pozycja 3 z `TODO.md`).

PROBLEM (zmierzony, nie hipotetyczny): `grounded_fraction()` zwracało 1.0, gdy odpowiedź nie
miała ANI JEDNEJ tezy — „nie ma czego podważać". Ale tezą nie był fragment ≤15 znaków, więc
wyjścia `"[1]"` (sam marker cytowania) i `"120 [1]"` (przeciek arytmetyczny) miały zero tez
i przechodziły przez WSZYSTKIE TRZY guardy do użytkownika. Dowód, że to dotyczy produkcji,
a nie tylko odrzuconego wariantu promptu z etapu 11: `results/backend_answers_ollama.json`
ma dla `g25` `answer="[1]"` i `grounded_fraction=1.0`.

CZEGO NIE DAŁO SIĘ ZROBIĆ JEDNĄ LINIJKĄ: „brak tez -> blokuj" ubija `65% [1]` — POPRAWNĄ
odpowiedź na `g13` (gold: „Około 65%.", evidence: „może przyjąć ok. 65% mieszkańców"). Przeciek
`120 [1]` i poprawna `65% [1]` mają IDENTYCZNY kształt; różni je tylko to, czy kontekst tę
wielkość twierdzi. Reguła patrząca na odpowiedź nie ma jak ich rozdzielić — musi to zrobić
sędzia, który widzi kontekst. Dlatego poprawka jest dwuczęściowa:
  (a) `to_claims()`: tezą jest fragment z choć jednym znakiem alfanumerycznym (zamiast >15 znaków)
      -> nagła wielkość IDZIE do sędziego zamiast znikać z pomiaru;
  (b) `grounded_fraction()`: zero tez = 0.0 (fail-CLOSED), bo po (a) „zero tez" znaczy już tylko
      „po usunięciu cytowań nie ma treści".

CO MIERZY TEN HARNESS — trzy rzeczy, każda innym kosztem:
1. `--struktura` (BEZ LLM-a, bez GPU): dowód ZASIĘGU zmiany. Dla każdej zapisanej odpowiedzi
   z `results/*.json` porównuje zbiór tez starą i nową regułą. To jest właściwa forma dowodu
   „brak regresji": `grounded_fraction()` ma tylko dwa wejścia (kontekst i lista tez), więc gdy
   listy są identyczne, ciąg wywołań sędziego jest bit w bit ten sam i decyzja też — bez ani
   jednego wywołania modelu. Przepuszczanie gold setu przez sędziego mierzyłoby jego wariancję,
   a nie skutek zmiany.
2. `--sedzia` (Ollama, bez generacji): PAROWANE przed/po na tych odpowiedziach, które zmiana
   realnie dotyka. Ten sam tekst, ten sam kontekst — różni się wyłącznie reguła tezy.
3. `--wzorce` (Ollama, bez generacji): strona KOSZTU na n większym niż jeden. Krótkich poprawnych
   odpowiedzi w zapisanych danych jest mało, bo generator jest rozwlekły — a to właśnie one są
   zagrożone. Bierzemy więc WZORCOWE odpowiedzi z gold setu, które są ≤15 znaków (6 z 29), i
   pytamy sędziego o nie w produkcyjnym kontekście. To pomiar semi-syntetyczny (tekst odpowiedzi
   jest wzorcowy, nie modelowy) i jest tak nazwany w raporcie — ale mierzy dokładnie ryzyko,
   o które tu chodzi: czy sędzia potwierdza krótką, ugruntowaną tezę bez podmiotu („Około 65%").

Uruchomienie:
    python src/eval/ab_claims.py --struktura                 # dowód zasięgu, bez LLM-a
    python src/eval/ab_claims.py --sedzia --powtorzenia 3    # przed/po na zapisanych odpowiedziach
    python src/eval/ab_claims.py --wzorce --powtorzenia 3    # strona kosztu (gold, krótkie)
    python src/eval/ab_claims.py --raport                    # tabele z zapisanego pliku
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.eval.metrics import value_match
from src.generate.verify import to_claims
from src.ingest.chunk import split_sentences

RESULTS = config.ROOT / "results"
OUT_PATH = RESULTS / "ab_claims.json"
PROMPTS = RESULTS / "backend_prompts.json"          # konteksty gold setu (w .gitignore — duży)
HELDOUT_PATH = config.ROOT / "gold_set" / "heldout_gate.jsonl"
TRAPS_PATH = config.ROOT / "gold_set" / "entity_traps.jsonl"

STARY_PROG = 15


def tezy_stara_regula(answer_text: str) -> list[str]:
    """Definicja tezy PRZED poprawką — kopia usuniętej linijki z `verify.to_claims`.

    Harness musi trzymać starą regułę u siebie, bo po zmianie nie ma jej już w runtime, a bez
    niej nie ma z czym porównywać. Jedyne miejsce w repo, gdzie próg 15 znaków jeszcze żyje.
    """
    txt = re.sub(r"\[\d+\]", "", answer_text)
    return [s for s in split_sentences(txt) if len(s) > STARY_PROG]


# ─────────────────────────── zbieranie zapisanych odpowiedzi ───────────────────────────

def _konteksty_gold() -> dict[str, str]:
    """Konteksty pytań gold setu z `backend_prompts.json` (ten sam retrieval, więc ten sam tekst).

    Plik jest w `.gitignore` (pół megabajta promptów), więc jego brak NIE jest błędem — wtedy
    część wierszy nie ma kontekstu i harness mówi to wprost, zamiast po cichu ich pominąć.
    """
    if not PROMPTS.exists():
        return {}
    return {r["id"]: r["context"] for r in json.loads(PROMPTS.read_text(encoding="utf-8"))["rows"]}


def zbierz_odpowiedzi() -> list[dict]:
    """Wszystkie zapisane odpowiedzi z `results/*.json` + kontekst tam, gdzie da się go ustalić.

    Chodzimy po plikach REKURENCYJNIE i szukamy słowników z kluczem `answer` — pliki wyników
    z różnych etapów mają różne kształty (`rows`, `zysk_rows`, `koszt_rows`), a ujednolicanie ich
    wstecz byłoby przepisywaniem historii pomiarów. Kontekst bierzemy z tego samego słownika,
    a dla gold setu z `backend_prompts.json` po `id`.
    """
    gold_ctx = _konteksty_gold()
    out: list[dict] = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == OUT_PATH.name:
            continue
        try:
            dane = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        def obejdz(o) -> None:
            if isinstance(o, dict):
                if isinstance(o.get("answer"), str):
                    ident = o.get("id") or o.get("question") or "?"
                    out.append({"plik": path.name, "id": ident, "answer": o["answer"],
                                "context": o.get("context") or gold_ctx.get(ident),
                                "variant": o.get("variant"), "rep": o.get("rep")})
                for v in o.values():
                    obejdz(v)
            elif isinstance(o, list):
                for v in o:
                    obejdz(v)

        obejdz(dane)
    return out


# ─────────────────────────── etykiety: co POWINNO się stać ───────────────────────────

def _etykiety_pytan() -> tuple[set[str], dict[str, str]]:
    """(id pytań SPOZA korpusu, id -> wzorcowa odpowiedź dla pytań Z korpusu).

    Trzy zbiory naraz: gold set, held-out bramki i pułapki podmiany. Bez tego nie da się
    powiedzieć, czy blokada danego wiersza jest ZYSKIEM (przeciek) czy KOSZTEM (false-refusal) —
    a tabela bez tego rozróżnienia nie rozstrzyga niczego.
    """
    ooc, wzorce = set(), {}
    for g in (json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8")):
        (ooc.add(g["id"]) if not g["relevant_chunk_ids"] else wzorce.update({g["id"]: g["answer"]}))
    for h in (json.loads(l) for l in HELDOUT_PATH.open(encoding="utf-8")):
        if h["label"] == "out_of_corpus":
            ooc.add(h["id"])
    for t in (json.loads(l) for l in TRAPS_PATH.open(encoding="utf-8")):
        wzorce[t["id"]] = t["answer"]
    return ooc, wzorce


def oczekiwanie(row: dict, ooc: set[str], wzorce: dict[str, str]) -> str:
    """`blokada` | `przejście` | `?` — czego wymagamy od guardu na tym wierszu.

    Kolejność reguł jest treścią, nie kosmetyką:
    1. Brak treści po usunięciu cytowań (`"[1]"`) -> BLOKADA zawsze, także dla pytania Z korpusu.
       To nie jest ani odpowiedź, ani odmowa; użytkownik widzi uszkodzony przypis.
    2. Pytanie SPOZA korpusu -> BLOKADA (każda treść jest tu przeciekiem).
    3. Pytanie Z korpusu i odpowiedź zawiera wzorcową wartość -> MUSI PRZEJŚĆ (blokada = koszt).
    4. Reszta (odpowiedź na temat, ale nie umiem orzec poprawności) -> `?`, raportowana osobno.
       Wrzucenie jej do którejkolwiek kolumny zafałszowałoby wynik w wygodną stronę.
    """
    if not to_claims(row["answer"]):
        return "blokada"
    if row["id"] in ooc:
        return "blokada"
    if row["id"] in wzorce and value_match(wzorce[row["id"]], row["answer"])[0] is True:
        return "przejście"
    return "?"


# ─────────────────────────── 1) struktura: dowód zasięgu (bez LLM-a) ───────────────────────────

def roznica(rows: list[dict]) -> list[dict]:
    """Wiersze, na których decyzja guardu MOŻE się zmienić (czysta funkcja — testowalna bez plików).

    Poprawka ma dwie części i każda dotyka innego zbioru wierszy — mierzenie tylko jednej dałoby
    zaniżony zasięg i fałszywe poczucie bezpieczeństwa:
      `nowe tezy` — zbiór tez się zmienił (nagła wielkość stała się tezą). Decyduje SĘDZIA,
                    więc wynik jest stochastyczny i wymaga powtórzeń.
      `zero tez`  — zbiór tez jest pusty w OBU regułach (`"[1]"`). Zmieniło się to, co guard robi
                    z pustką: 1.0 (przepuść) -> 0.0 (blokuj). Decyzja DETERMINISTYCZNA, zero
                    wywołań sędziego. To ten wiersz jest właściwą dziurą fail-open.
    """
    out = []
    for r in rows:
        stare, nowe = tezy_stara_regula(r["answer"]), to_claims(r["answer"])
        powod = "nowe tezy" if stare != nowe else ("zero tez" if not nowe else None)
        if powod:
            out.append({**r, "tezy_stare": stare, "tezy_nowe": nowe, "powód": powod})
    return out


def raport_struktury(rows: list[dict], zmienione: list[dict]) -> dict:
    ooc, wzorce = _etykiety_pytan()
    print(f"\n=== ZASIĘG ZMIANY na {len(rows)} zapisanych odpowiedziach "
          f"({len({r['plik'] for r in rows})} plików wyników) ===")
    print(f"  BEZ ZMIANY (tezy te same, niepuste):  {len(rows) - len(zmienione)}  "
          f"<- operacja tożsamościowa: te same wywołania sędziego, ta sama decyzja")
    for powod in ("nowe tezy", "zero tez"):
        n = len([r for r in zmienione if r["powód"] == powod])
        print(f"  {powod.upper():36} {n:4}")

    c = Counter((r["answer"].strip(), r["powód"], oczekiwanie(r, ooc, wzorce)) for r in zmienione)
    print(f"\n  {'n':>4}  {'mechanizm':10} {'oczekiwanie':11} {'odpowiedź':22} tezy stare -> nowe")
    for (ans, powod, ocz), n in c.most_common():
        stare, nowe = tezy_stara_regula(ans), to_claims(ans)
        print(f"  {n:4}  {powod:10} {ocz:11} {ans!r:22} {stare} -> {nowe}")

    bez_ctx = [r for r in zmienione if not r["context"]]
    if bez_ctx:
        # Świadomie NIE pomijamy tego w ciszy: wiersz bez kontekstu jest nieoceniany przez
        # sędziego, więc musi być widoczny jako granica pomiaru, a nie zniknąć z tabeli.
        print(f"\n  BEZ KONTEKSTU (nieocenialne przez sędziego): {len(bez_ctx)} wierszy — "
              f"{sorted({(r['plik'], r['id']) for r in bez_ctx})}")
        if not PROMPTS.exists():
            print(f"  (brak {PROMPTS.name} — plik jest w .gitignore; z nim konteksty gold setu wracają)")
    return {"n_odpowiedzi": len(rows), "n_zmienionych": len(zmienione),
            "kształty": [{"odpowiedź": a, "mechanizm": m, "oczekiwanie": o, "n": n}
                         for (a, m, o), n in c.items()],
            "bez_kontekstu": sorted({f"{r['plik']}:{r['id']}" for r in bez_ctx})}


# ─────────────────────────── 2) sędzia: przed/po (parowane) ───────────────────────────

def _ocena(context: str, claims: list[str], fail_open: bool, cache: dict) -> tuple[float, list]:
    """Ułamek ugruntowania dla PODANEJ listy tez. `fail_open` = jak reguła traktuje zero tez.

    Cache po treści tezy: stara i nowa reguła mają zwykle część tez wspólnych, a sędzia jest
    stochastyczny — bez cache'u „różnica" zawierałaby jego wariancję na tych samych tezach.
    Parowanie musi być mocne: ta sama teza = ta sama ocena w tym samym powtórzeniu.
    """
    if not claims:
        return (1.0 if fail_open else 0.0), []
    from src.generate.verify import judge_grounded

    verdicts = []
    for c in claims:
        if c not in cache:
            cache[c] = judge_grounded(context, c)
        ok, reason = cache[c]
        verdicts.append({"claim": c, "grounded": ok, "reason": reason[:200]})
    return sum(v["grounded"] for v in verdicts) / len(verdicts), verdicts


def pomiar_sedziego(zmienione: list[dict], powtorzenia: int) -> list[dict]:
    """Przed/po dla wierszy, które zmiana dotyka i dla których mamy kontekst.

    Deduplikacja po (odpowiedź, kontekst): identyczne wyjścia z różnych plików/powtórzeń dałyby
    sędziemu identyczne pytanie. Liczbę wystąpień zachowujemy (`n_wystapien`), żeby waga wiersza
    w produkcyjnym przebiegu nie zniknęła z raportu.
    """
    ooc, wzorce = _etykiety_pytan()
    grupy: dict[tuple[str, str], dict] = {}
    for r in zmienione:
        if not r["context"]:
            continue
        k = (r["answer"], r["context"])
        g = grupy.setdefault(k, {"id": r["id"], "ids": set(), "answer": r["answer"],
                                 "context": r["context"], "pliki": set(), "n_wystapien": 0,
                                 "oczekiwanie": oczekiwanie(r, ooc, wzorce)})
        # Grupujemy po (odpowiedź, kontekst), więc DWA RÓŻNE pytania mogą wpaść do jednej grupy —
        # tak jest z t05/t06, którym retrieval zwrócił identyczne chunki. Wypisujemy wszystkie id,
        # bo tabela pokazująca jedno udawałaby, że drugie pytanie nie było mierzone.
        g["ids"].add(r["id"])
        g["pliki"].add(r["plik"])
        g["n_wystapien"] += 1

    prog = config.VERIFY_MIN_SUPPORTED
    out = []
    for i, g in enumerate(grupy.values(), 1):
        stare, nowe = tezy_stara_regula(g["answer"]), to_claims(g["answer"])
        przed, po = [], []
        for rep in range(powtorzenia):
            cache: dict[str, tuple[bool, str]] = {}         # parowanie WEWNĄTRZ powtórzenia
            f_old, v_old = _ocena(g["context"], stare, fail_open=True, cache=cache)
            f_new, v_new = _ocena(g["context"], nowe, fail_open=False, cache=cache)
            przed.append({"frac": f_old, "blokuje": f_old < prog, "verdicts": v_old})
            po.append({"frac": f_new, "blokuje": f_new < prog, "verdicts": v_new})
            print(f"  [{i}/{len(grupy)}] {g['id']} rep{rep + 1} {g['answer'].strip()!r:14} "
                  f"przed: blokuje={f_old < prog} (frac {f_old:.2f})   "
                  f"po: blokuje={f_new < prog} (frac {f_new:.2f})", flush=True)
        out.append({**g, "pliki": sorted(g["pliki"]), "ids": sorted(g["ids"]),
                    "id": "+".join(sorted(g["ids"])), "tezy_stare": stare, "tezy_nowe": nowe,
                    "przed": przed, "po": po})
    return out


# ─────────────────────── 3) wzorce: strona kosztu na większym n ───────────────────────

def pomiar_wzorcow(powtorzenia: int) -> list[dict]:
    """Krótkie WZORCOWE odpowiedzi z gold setu (≤15 znaków) w formacie „odpowiedź + cytat".

    PO CO: strona kosztu w zapisanych danych stoi na jednym pytaniu (`g13`), bo generator jest
    rozwlekły i krótkich odpowiedzi prawie nie produkuje. Ryzyko jest jednak systematyczne:
    prompt sędziego wymaga zdania JAWNIE łączącego podmiot z wartością, a krótka odpowiedź
    („Około 65%.", „W 2008 r.") podmiotu NIE nazywa — nosi go pytanie. Jeśli sędzia tego nie
    potwierdza, nowa reguła zamienia dziurę fail-open na false-refusal i trzeba to wiedzieć
    PRZED wdrożeniem, a nie po skargach użytkownika.
    Semi-syntetyczne i tak nazwane: tekst odpowiedzi jest wzorcowy (poprawny z definicji gold
    setu), kontekst i sędzia są produkcyjne. Mierzymy zachowanie SĘDZIEGO, nie generatora.
    """
    gold_ctx = _konteksty_gold()
    if not gold_ctx:
        sys.exit(f"Brak {PROMPTS.name} — bez kontekstów gold setu nie ma czego mierzyć "
                 f"(odtwórz: python src/eval/compare_backends.py --prompty).")
    prog = config.VERIFY_MIN_SUPPORTED
    out = []
    gold = [json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8")]
    krotkie = [g for g in gold if g["relevant_chunk_ids"] and len(g["answer"].strip()) <= STARY_PROG]
    for i, g in enumerate(krotkie, 1):
        # Format taki, jaki wymusza prompt produkcyjny: teza + numer źródła.
        ans = f"{g['answer'].strip()} [1]"
        ctx = gold_ctx.get(g["id"])
        if not ctx:
            print(f"  [{i}/{len(krotkie)}] {g['id']} POMINIĘTE — brak kontekstu")
            continue
        stare, nowe = tezy_stara_regula(ans), to_claims(ans)
        przed, po = [], []
        for rep in range(powtorzenia):
            cache: dict[str, tuple[bool, str]] = {}
            f_old, _ = _ocena(ctx, stare, fail_open=True, cache=cache)
            f_new, v_new = _ocena(ctx, nowe, fail_open=False, cache=cache)
            przed.append({"frac": f_old, "blokuje": f_old < prog})
            po.append({"frac": f_new, "blokuje": f_new < prog, "verdicts": v_new})
            print(f"  [{i}/{len(krotkie)}] {g['id']} rep{rep + 1} {ans!r:20} "
                  f"przed: blokuje={f_old < prog}   po: blokuje={f_new < prog} "
                  f"({v_new[0]['reason'][:90] if v_new else ''})", flush=True)
        out.append({"id": g["id"], "question": g["question"], "answer": ans,
                    "context": ctx, "oczekiwanie": "przejście",
                    "tezy_stare": stare, "tezy_nowe": nowe, "przed": przed, "po": po})
    return out


# ─────────────────────────── raport ───────────────────────────

def _tabela(nazwa: str, wiersze: list[dict]) -> None:
    print(f"\n=== {nazwa} ===")
    print(f"  {'oczekiwanie':11} {'id':10} {'odpowiedź':16} {'n':>3}  "
          f"{'PRZED blokuje':>14} {'PO blokuje':>11}  uzasadnienie sędziego (po)")
    for w in wiersze:
        rep = len(w["po"])
        pb = sum(x["blokuje"] for x in w["przed"])
        ob = sum(x["blokuje"] for x in w["po"])
        powod = next((v["reason"] for x in w["po"] for v in x["verdicts"] if not v["grounded"]), "")
        print(f"  {w['oczekiwanie']:11} {w['id']:10} {w['answer'].strip()!r:16} "
              f"{w.get('n_wystapien', 1):3}  {f'{pb}/{rep}':>14} {f'{ob}/{rep}':>11}  {powod[:70]}")
    # Podsumowanie w kategoriach obu stron błędu — jedna liczba per strona, bo to ona rozstrzyga.
    for ocz, etykieta in [("blokada", "PRZECIEKI (blokada = zysk)"),
                          ("przejście", "POPRAWNE (blokada = FALSE-REFUSAL)")]:
        grupa = [w for w in wiersze if w["oczekiwanie"] == ocz]
        if not grupa:
            continue
        n = sum(len(w["po"]) for w in grupa)
        pb = sum(x["blokuje"] for w in grupa for x in w["przed"])
        ob = sum(x["blokuje"] for w in grupa for x in w["po"])
        print(f"  {etykieta:38} przed {pb}/{n}  ->  po {ob}/{n}")
    nieznane = [w for w in wiersze if w["oczekiwanie"] == "?"]
    if nieznane:
        print(f"  (wiersze bez orzeczonej poprawności, nieliczone do żadnej strony: "
              f"{[w['id'] for w in nieznane]})")


def raport(dane: dict) -> None:
    if dane.get("struktura"):
        s = dane["struktura"]
        print(f"\n=== ZASIĘG ZMIANY (z zapisu) ===")
        print(f"  odpowiedzi: {s['n_odpowiedzi']}, zbiór tez zmieniony w: {s['n_zmienionych']}")
        for k in s["kształty"]:
            print(f"    {k['n']:4}  {k.get('mechanizm', '?'):10} {k['oczekiwanie']:11} "
                  f"{k['odpowiedź']!r}")
    if dane.get("sedzia"):
        _tabela("SĘDZIA na ZAPISANYCH odpowiedziach (parowane: ten sam tekst i kontekst)",
                dane["sedzia"])
    if dane.get("wzorce"):
        _tabela("SĘDZIA na KRÓTKICH WZORCOWYCH odpowiedziach z gold setu (semi-syntetyczne)",
                dane["wzorce"])


def _zapisz(dane: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(dane, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUT_PATH.relative_to(config.ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B definicji tezy — dziura fail-open guardu #3.")
    ap.add_argument("--struktura", action="store_true", help="zasięg zmiany, bez LLM-a")
    ap.add_argument("--sedzia", action="store_true", help="przed/po na zapisanych odpowiedziach")
    ap.add_argument("--wzorce", action="store_true", help="strona kosztu: krótkie wzorce z gold setu")
    ap.add_argument("--powtorzenia", type=int, default=3, help="ile razy ocenić TĘ SAMĄ tezę")
    ap.add_argument("--raport", action="store_true", help="tabele z zapisanego pliku")
    args = ap.parse_args()

    if args.raport:
        if not OUT_PATH.exists():
            sys.exit(f"Brak {OUT_PATH} — uruchom najpierw pomiar.")
        raport(json.loads(OUT_PATH.read_text(encoding="utf-8")))
        return

    dane = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    rows = zbierz_odpowiedzi()
    zmienione = roznica(rows)

    if args.struktura or not (args.sedzia or args.wzorce):
        dane["struktura"] = raport_struktury(rows, zmienione)
        _zapisz(dane)

    if args.sedzia:
        print(f"\nSędzia ({config.JUDGE_MODEL}) na zapisanych odpowiedziach, "
              f"{args.powtorzenia} powtórzeń:")
        dane["sedzia"] = pomiar_sedziego(zmienione, args.powtorzenia)
        _zapisz(dane)

    if args.wzorce:
        print(f"\nSędzia ({config.JUDGE_MODEL}) na krótkich wzorcach z gold setu, "
              f"{args.powtorzenia} powtórzeń:")
        dane["wzorce"] = pomiar_wzorcow(args.powtorzenia)
        _zapisz(dane)

    raport(dane)


if __name__ == "__main__":
    main()
