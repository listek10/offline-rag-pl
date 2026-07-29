"""
src/eval/ab_prompt.py
Etap 11 / checkpointy #7 i #8: A/B INSTRUKCJI SYSTEMOWEJ — obie strony błędu w jednym przebiegu.

PROBLEM (zmierzony 29.07, etap 10): korpus mówi „łotewskie 1,548 mld", pytanie mówi „Łotwy" —
i model ODMAWIA, choć chunk z odpowiedzią stoi na pozycji 1 (rerank 0,931), a `blocked_by=None`
(czyli odmawia sam generator, guard #1, nie bramka). Dowód rozstrzygający jest w tym samym
pomiarze: pytany o ESTONIĘ model odpowiada „…Estonii 2,316 mld, podczas gdy łotewskie 1,548,
a litewskie 2,048" — więc fakt jest w kontekście, który model widzi, i potrafi go przeczytać.
Barierą jest wyłącznie forma gramatyczna nazwy podmiotu.

DLACZEGO OSOBNY HARNESS, A NIE FLAGA W eval_traps.py: tam pomiar jest PAROWANY po odpowiedzi
(generacja RAZ, potem warianty guardów na tych samych tekstach). Przy zmianie promptu generacja
jest właśnie tym, co się zmienia — parowania po odpowiedzi utrzymać się NIE DA. Zamiast udawać,
że da, parujemy po PYTANIU i mierzymy rozkład na N powtórzeniach generacji. Powód jest zmierzony:
temperatura 0 daje dekodowanie zachłanne, ale nie powtarzalne (t02 odpowiadało 3/3 poprawnie,
a po przerwie 3/3 odmawiało przy identycznym retrievalu). Jedna liczba z jednego przebiegu
nie odróżniłaby efektu promptu od tego szumu.

DRABINA WARIANTÓW — JEDNA ZMIANA NA SZCZEBEL (zadeklarowana PRZED pomiarem, nie dobrana po):
  v2     produkcyjny prompt (baseline, `answer.SYSTEM_PROMPT`)
  v2bis  TEN SAM tekst co v2 — kontrola POZIOMU SZUMU (patrz akapit o przeplocie)
  v3     v2 + JEDNO zdanie: przymiotnik odmiejscowy nazywa ten sam podmiot co rzeczownik z pytania
  v4     v3 + JEDNO zdanie: nie przenoś wartości między podmiotami
v4 istnieje, bo v3 z założenia przesuwa model ku ŚMIELSZEMU utożsamianiu nazw — a to ten sam
mechanizm, który przestrzelony produkuje PODMIANĘ PODMIOTU (etap 10: pytanie o Danię -> liczba
Norwegii). Gdybym dorzucił oba zdania naraz, nie wiedziałbym, czy podmiana nie wzrosła dzięki
regule anty-przenoszeniowej, czy dlatego, że reguła formy jest bezpieczna. To dwa różne wnioski.

PRZEPLOT WARIANTÓW I KONTROLA SZUMU — poprawka po PIERWSZYM, WADLIWYM przebiegu:
pierwsza wersja tego skryptu puszczała warianty BLOKAMI (wszystkie powtórzenia v2, potem v3,
potem v4) i zapisała wynik do `results/ab_prompt_blokowy.json`. Ten układ jest bezużyteczny
w tym systemie, bo dryf w czasie jest w nim ZMIERZONYM faktem (etap 10: t02 odpowiadało 3/3
poprawnie, a po przerwie 3/3 odmawiało przy identycznym prompcie i retrievalu) — więc „czas"
i „numer wariantu" były w pomiarze tą samą zmienną i nie dało się ich rozdzielić po fakcie.
Poprawka jest dwuczęściowa:
  (a) PRZEPLOT: pętla powtórzeń jest ZEWNĘTRZNA, warianty lecą wewnątrz jednego powtórzenia —
      porównania są między odpowiedziami wygenerowanymi obok siebie w czasie, a dryf rozkłada
      się na wszystkie warianty zamiast obciążać ostatni (klasyczny randomized block design);
  (b) v2bis: ten sam prompt puszczony DRUGI raz w tym samym powtórzeniu. Każda różnica v2 vs
      v2bis jest czystym szumem, bo tekst promptu jest identyczny (pilnuje tego test kontraktowy).
      To daje PRÓG ISTOTNOŚCI: różnica v2 vs v3 mniejsza od różnicy v2 vs v2bis nic nie znaczy.

OBIE STRONY BŁĘDU, BO ROZLUŹNIENIE PROMPTU JEST HANDLEM:
  ZYSK  — pułapki z `gold_set/entity_traps.jsonl`: ile `odmowa` przechodzi w `poprawna`
          (i KRYTYCZNE: czy przy tym nie rośnie kolumna `PODMIANA` — patrz akapit o v4).
  KOSZT — refusal na pytaniach SPOZA korpusu: gold set (n=7) + held-out (n=17).
          Mierzone z BRAMKĄ WYŁĄCZONĄ (`gate=False`) i to nie jest niedbalstwo, a warunek
          poprawności: bramka działa PRZED generacją, więc przy niej włączonej pytania spoza
          korpusu nie dotarłyby do promptu i każdy wariant dałby refusal ~1,00. Zobaczyłbym
          „prompt nic nie kosztuje" jako artefakt pomiaru. Ta izolacja ma też znaczenie
          praktyczne: w UI jest tryb demo z rozbrojoną bramką, a na obcym PDF-ie próg nie jest
          skalibrowany — wtedy prompt zostaje JEDYNĄ linią obrony.

CZEGO TU NIE MIERZYMY (świadomie): false-refusal na pytaniach Z korpusu w gold secie.
Rozluźnienie promptu może go tylko zmniejszyć, a odzyskanie t05/t06 JEST tym zyskiem, tylko
zmierzonym dokładniej — na pytaniach, o których wiem, że model na nich upada.

Uruchomienie (wymaga Qdrant + Ollama; ~25 min dla 3 warianty x 3 powtórzenia):
    python src/eval/ab_prompt.py                          # pełny przebieg, zapis do results/
    python src/eval/ab_prompt.py --repeats 1 --warianty v2 v3
    python src/eval/ab_prompt.py --tylko zysk             # albo: --tylko koszt
    python src/eval/ab_prompt.py --raport                 # tabele z zapisanego pliku
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
from src.generate.answer import SYSTEM_PROMPT, answer, build_context, refused

RESULTS = config.ROOT / "results"
OUT_PATH = RESULTS / "ab_prompt.json"
TRAPS_PATH = config.ROOT / "gold_set" / "entity_traps.jsonl"
HELDOUT_PATH = config.ROOT / "gold_set" / "heldout_gate.jsonl"
CLASSES = ["poprawna", "niejednoznaczna", "podmiana", "inna-bledna", "odmowa"]

# Warianty budujemy przez DOPISANIE zdania do produkcyjnego promptu, a nie przez przepisanie go.
# Dzięki temu „jedna zmiana na szczebel" jest własnością tekstu, którą widać w kodzie, a nie
# obietnicą w komentarzu — i nie da się przy okazji przestawić kolejności starych reguł.
REGULA_FORMY = """
5. TA SAMA NAZWA W INNEJ FORMIE GRAMATYCZNEJ to TEN SAM podmiot: przymiotnik odmiejscowy nazywa \
to samo państwo co rzeczownik z pytania — „łotewskie wydatki" to „wydatki Łotwy", „fiński budżet" \
to „budżet Finlandii". Sama różnica formy NIE jest powodem do odmowy: jeśli fakt stoi w KONTEKŚCIE \
w formie przymiotnikowej, podaj go i zacytuj źródło."""

REGULA_NIEPRZENOSZENIA = """
6. NIE PRZENOŚ wartości między podmiotami: liczba stojąca przy jednym państwie NIE jest \
odpowiedzią o inne państwo, nawet jeśli oba są w tym samym zdaniu. Jeśli w KONTEKŚCIE jest \
wartość dla innego podmiotu, a nie dla tego z pytania — odmów zgodnie z zasadą 2."""

WARIANTY: dict[str, str] = {
    "v2": SYSTEM_PROMPT,                                              # baseline = produkcja
    # v2bis MUSI być bit w bit tym samym tekstem co v2 — inaczej przestaje mierzyć szum,
    # a zaczyna mierzyć kolejną zmianę promptu. Test kontraktowy tego pilnuje.
    "v2bis": SYSTEM_PROMPT,
    "v3": SYSTEM_PROMPT + REGULA_FORMY,
    "v4": SYSTEM_PROMPT + REGULA_FORMY + REGULA_NIEPRZENOSZENIA,
}


# ─────────────────────────── dane ───────────────────────────

def load_traps() -> list[dict]:
    return [json.loads(l) for l in TRAPS_PATH.open(encoding="utf-8")]


def load_ooc() -> list[dict]:
    """Pytania SPOZA korpusu z DWÓCH zbiorów — z etykietą, z którego pochodzą.

    Gold set (n=7) jest raportowany w README, held-out (n=17) służył do kalibracji progu i jako
    metryki go nie raportujemy. Tu nie kalibruję progu, tylko mierzę prompt, więc mogę użyć obu —
    ale trzymam je w OSOBNYCH kolumnach, żeby nie zatarła się ta różnica statusu.
    """
    gold = [json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8")]
    rows = [{"id": g["id"], "question": g["question"], "zbior": "gold"}
            for g in gold if not g["relevant_chunk_ids"]]
    held = [json.loads(l) for l in HELDOUT_PATH.open(encoding="utf-8")]
    rows += [{"id": h["id"], "question": h["question"], "zbior": "held-out"}
             for h in held if h["label"] == "out_of_corpus"]
    return rows


# ─────────────────────── generacja (GPU) ───────────────────────

def generuj_zysk(warianty: list[str], repeats: int) -> list[dict]:
    """Odpowiedzi na pułapki: każdy wariant promptu x każde powtórzenie.

    `verify=False`, bo guard #3 dokładamy PÓŹNIEJ na zapisanej odpowiedzi (jest stochastyczny
    i kosztuje wywołania sędziego — nie ma powodu mieszać go do generacji).
    Bramka (guard #2) WŁĄCZONA: pułapki są na temat korpusu, więc i tak ją przechodzą, a jej
    wyłączenie zmieniłoby tor względem produkcji bez żadnego zysku poznawczego.
    """
    traps = load_traps()
    rows = []
    for rep in range(repeats):                 # PRZEPLOT: powtórzenie na zewnątrz, wariant wewnątrz
        for w in warianty:
            for t in traps:
                res = answer(t["question"], system=WARIANTY[w], gate=True, verify=False)
                odm = refused(res["answer"])
                rows.append({"variant": w, "rep": rep, "id": t["id"], "entity": t["entity"],
                             "answer": res["answer"], "context": build_context(res["sources"]),
                             "refused": odm})
                print(f"  [zysk] {w} rep{rep + 1} {t['id']} {t['entity']:10} "
                      f"{'ODMOWA' if odm else ' '.join(res['answer'].split())[:70]}", flush=True)
    return rows


def generuj_koszt(warianty: list[str], repeats: int) -> list[dict]:
    """Refusal na pytaniach spoza korpusu, BRAMKA WYŁĄCZONA (izolacja guardu #1 — patrz docstring)."""
    ooc = load_ooc()
    rows = []
    for rep in range(repeats):                 # PRZEPLOT — jak w generuj_zysk (patrz docstring modułu)
        for w in warianty:
            przecieki = []
            for q in ooc:
                res = answer(q["question"], system=WARIANTY[w], gate=False, verify=False)
                odm = refused(res["answer"])
                if not odm:
                    przecieki.append(q["id"])
                rows.append({"variant": w, "rep": rep, "id": q["id"], "zbior": q["zbior"],
                             "refused": odm, "answer": res["answer"]})
            print(f"  [koszt] {w} rep{rep + 1}: przecieki {przecieki or '—'}", flush=True)
    return rows


def dolicz_guard3(rows: list[dict]) -> None:
    """Dokłada do wierszy pułapek werdykt guardu #3 (sędzia LLM na ZAPISANEJ odpowiedzi).

    Po co: zysk widziany „bez guardów" może w produkcji nie istnieć, jeśli odzyskaną odpowiedź
    ubije weryfikacja tez. Raportuję więc obie kolumny — sam prompt i konfigurację produkcyjną.
    Odmowy modelu pomijamy: nie ma czego weryfikować (tak samo jak w eval_traps.py).
    """
    from src.generate.verify import grounded_fraction
    for i, r in enumerate(rows, 1):
        if r["refused"]:
            r["guard3_blokuje"] = False
            continue
        frac, _ = grounded_fraction(r["context"], r["answer"])
        r["guard3_blokuje"] = frac < config.VERIFY_MIN_SUPPORTED
        print(f"  [guard #3] {i}/{len(rows)} {r['variant']} {r['id']} "
              f"frac={frac:.2f} blokuje={r['guard3_blokuje']}", flush=True)


# ─────────────── A/B SĘDZIEGO (guard #3) — ta sama ślepota, warstwę niżej ───────────────

# Pomiar generatora pokazał, że reguła formy działa (t06 Litwa: odmowa -> poprawna), ale
# w konfiguracji produkcyjnej zysk ZNIKA, bo guard #3 blokuje tę poprawną odpowiedź 3/3.
# Uzasadnienie sędziego, zacytowane dosłownie: „Kontekst mówi o wydatkach obronnych Estonii,
# Łotwy i Litwy w określonym okresie" — czyli widzi temat i odmawia potwierdzenia wartości,
# choć kontekst zawiera „litewskie 2,048 mld dolarów". To NIE jest przypadek: prompt sędziego
# wymaga zdania „JAWNIE łączącego podmiot z wartością" i każe najpierw szukać pułapki „inne
# państwo" — czyli utwardzenie, które naprawiło h18, produkuje fałszywą odmowę na drugim końcu
# tej samej osi. Wniosek do zmierzenia: naprawa jednej warstwy tylko PRZENOSI odmowę do następnej.
REGULA_FORMY_SEDZIA = """
FORMA GRAMATYCZNA NAZWY: przymiotnik odmiejscowy oznacza TEN SAM podmiot co nazwa państwa \
w twierdzeniu — „łotewskie wydatki" to wydatki Łotwy, „litewskie 2,048 mld" to 2,048 mld Litwy. \
Takie zdanie JAWNIE łączy podmiot z wartością i NIE jest pułapką „innego państwa"."""


def ab_sedziego(rows: list[dict], traps: list[dict], powtorzenia: int = 1) -> dict:
    """Guard #3 dwoma promptami sędziego na TYCH SAMYCH zapisanych odpowiedziach.

    Pomiar jest tu parowany w mocnym sensie (ta sama odpowiedź, ta sama teza, ten sam kontekst) —
    różni się WYŁĄCZNIE instrukcja sędziego, więc różnica w blokadach jest przypisywalna jej.
    Raportujemy obie strony naraz: czy reguła odblokowuje t06 (zysk) i czy nie przestaje blokować
    t01 (koszt — t01 to prawdziwa podmiana podmiotu, jedyna rzecz, którą guard #3 dziś ratuje).

    `powtorzenia` > 1 ocenia TĘ SAMĄ odpowiedź wielokrotnie, bo sędzia jest stochastyczny: jedna
    ocena na odpowiedź daje rozkład zero-jedynkowy, z którego nie widać, czy „przepuścił raz na
    trzy" to efekt promptu, czy wariancja sędziego. Przy decyzji o guardzie ta różnica jest treścią.
    """
    from src.generate.verify import JUDGE_GROUNDED, grounded_fraction

    prompty = {"produkcyjny": JUDGE_GROUNDED,
               "+ reguła formy": JUDGE_GROUNDED + REGULA_FORMY_SEDZIA}
    by_id = {t["id"]: t for t in traps}
    out: dict[str, dict] = {}
    do_oceny = [r for r in rows if not r["refused"]]
    rozwiniete = [r for _ in range(powtorzenia) for r in do_oceny]
    for nazwa, sysp in prompty.items():
        for i, r in enumerate(rozwiniete, 1):
            frac, verdicts = grounded_fraction(r["context"], r["answer"], system=sysp)
            blokuje = frac < config.VERIFY_MIN_SUPPORTED
            powt = (i - 1) // len(do_oceny)
            k = f"{r['variant']}|{r['id']}|{r['rep']}|{powt}"
            out.setdefault(k, {})[nazwa] = {
                "blokuje": blokuje, "frac": frac,
                "klasa": "odmowa" if blokuje else klasa(by_id[r["id"]], r),
                "uzasadnienia": [v["reason"][:200] for v in verdicts]}
            print(f"  [sędzia {nazwa}] {i}/{len(rozwiniete)} {r['variant']} {r['id']} "
                  f"frac={frac:.2f} blokuje={blokuje}", flush=True)
    return out


def raport_sedziego(sedzia: dict) -> None:
    print("\n=== A/B SĘDZIEGO (guard #3) na tych samych odpowiedziach ===")
    warianty = sorted({k.split("|")[0] for k in sedzia})
    print(f"  {'wariant':8} {'prompt sędziego':16} {'poprawna':>9} {'PODMIANA':>9} "
          f"{'inna-bł.':>9} {'odmowa':>8}")
    for w in warianty:
        for nazwa in ("produkcyjny", "+ reguła formy"):
            c = Counter(v[nazwa]["klasa"] for k, v in sedzia.items()
                        if k.startswith(f"{w}|") and nazwa in v)
            if c:
                print(f"  {w:8} {nazwa:16} {c['poprawna']:9} {c['podmiana']:9} "
                      f"{c['inna-bledna']:9} {c['odmowa']:8}")
    print("\n  PER PYTANIE (blokady sędziego, powtórzenia zsumowane):")
    for w in warianty:
        for pyt in sorted({k.split("|")[1] for k in sedzia if k.startswith(f"{w}|")}):
            wiersze = [v for k, v in sedzia.items() if k.startswith(f"{w}|{pyt}|")]
            prod = sum(v["produkcyjny"]["blokuje"] for v in wiersze if "produkcyjny" in v)
            reg = sum(v["+ reguła formy"]["blokuje"] for v in wiersze if "+ reguła formy" in v)
            if prod or reg:
                print(f"    {w:6} {pyt}  produkcyjny blokuje {prod}/{len(wiersze)}   "
                      f"+reguła formy {reg}/{len(wiersze)}")


# ──────────────── podsumowania (czyste funkcje, testowalne bez GPU) ────────────────

def klasa(trap: dict, row: dict, blocked: bool = False) -> str:
    """Klasa wiersza. Odmowa (modelu albo guardu) zjada klasyfikację wartości — nie ma czego oceniać."""
    if blocked or row["refused"]:
        return "odmowa"
    return classify_trap_answer(trap["answer"], trap["trap_values"], row["answer"])


def tabela_zysku(traps: list[dict], rows: list[dict], z_guardem3: bool = False) -> dict:
    """Rozkład klas per wariant promptu. `z_guardem3` = konfiguracja produkcyjna."""
    by_id = {t["id"]: t for t in traps}
    out: dict[str, Counter] = {}
    for r in rows:
        blocked = bool(r.get("guard3_blokuje")) if z_guardem3 else False
        out.setdefault(r["variant"], Counter())[klasa(by_id[r["id"]], r, blocked)] += 1
    return {w: {k: c[k] for k in CLASSES} for w, c in out.items()}


def zysk_per_pytanie(rows: list[dict], traps: list[dict]) -> dict:
    """Parowanie po PYTANIU: dla każdej pułapki lista klas w kolejnych powtórzeniach, per wariant.

    To jest widok, który rozstrzyga — średnia po zbiorze potrafi ukryć, że zysk na t05 został
    opłacony stratą na t01. Bez tej tabeli nie wolno ogłosić poprawy.
    """
    by_id = {t["id"]: t for t in traps}
    out: dict[str, dict[str, list[str]]] = {}
    for r in sorted(rows, key=lambda x: (x["id"], x["variant"], x["rep"])):
        out.setdefault(r["id"], {}).setdefault(r["variant"], []).append(klasa(by_id[r["id"]], r))
    return out


def tabela_kosztu(rows: list[dict]) -> dict:
    """Refusal per wariant i per zbiór + lista przeciekających pytań (audytowalność)."""
    out: dict[str, dict] = {}
    for r in rows:
        d = out.setdefault(r["variant"], {}).setdefault(r["zbior"], {"n": 0, "odmowy": 0,
                                                                    "przecieki": []})
        d["n"] += 1
        if r["refused"]:
            d["odmowy"] += 1
        else:
            d["przecieki"].append(f"{r['id']}(rep{r['rep'] + 1})")
    for zbiory in out.values():
        for d in zbiory.values():
            d["refusal"] = round(d["odmowy"] / d["n"], 3) if d["n"] else 0.0
    return out


# ─────────────────────────── raport ───────────────────────────

def raport(dane: dict) -> None:
    traps = load_traps()
    zysk, koszt = dane.get("zysk_rows") or [], dane.get("koszt_rows") or []
    warianty = [w for w in WARIANTY if any(r["variant"] == w for r in zysk + koszt)]

    if zysk:
        n = len({r["id"] for r in zysk})
        rep = len({r["rep"] for r in zysk})
        print(f"\n=== ZYSK: {n} pułapek x {rep} powtórzeń generacji (parowane po PYTANIU) ===")
        for etykieta, g3 in [("sam prompt (bez guardów post)", False),
                             ("+ guard #3 (konfiguracja produkcyjna)", True)]:
            if g3 and "guard3_blokuje" not in (zysk[0] if zysk else {}):
                continue
            t = tabela_zysku(traps, zysk, z_guardem3=g3)
            print(f"\n  {etykieta}")
            print(f"  {'wariant':8} {'poprawna':>9} {'niejedn.':>9} {'PODMIANA':>9} "
                  f"{'inna-bł.':>9} {'odmowa':>8}")
            for w in warianty:
                c = t.get(w)
                if c:
                    print(f"  {w:8} {c['poprawna']:9} {c['niejednoznaczna']:9} {c['podmiana']:9} "
                          f"{c['inna-bledna']:9} {c['odmowa']:8}")

        print("\n  PER PYTANIE (klasy w kolejnych powtórzeniach):")
        per = zysk_per_pytanie(zysk, traps)
        for t in traps:
            if t["id"] in per:
                print(f"    {t['id']} {t['entity']:10} " + "   ".join(
                    f"{w}: {','.join(per[t['id']].get(w, []))}" for w in warianty))

    if koszt:
        print(f"\n=== KOSZT: refusal na pytaniach SPOZA korpusu, BRAMKA WYŁĄCZONA ===")
        t = tabela_kosztu(koszt)
        zbiory = ["gold", "held-out"]
        print(f"  {'wariant':8} " + " ".join(f"{z + ' refusal':>18}" for z in zbiory)
              + "   przecieki")
        for w in warianty:
            if w not in t:
                continue
            kol = " ".join(f"{t[w][z]['refusal']:12.2f} ({t[w][z]['n']:3})" if z in t[w]
                           else f"{'—':>18}" for z in zbiory)
            prz = sorted({p.split("(")[0] for z in zbiory
                          for p in t[w].get(z, {}).get("przecieki", [])})
            print(f"  {w:8} {kol}   {', '.join(prz) or '—'}")
        print("  (refusal 1,00 = zero przecieków. Liczba w nawiasie = pytania x powtórzenia.)")

    if dane.get("sedzia_ab"):
        raport_sedziego(dane["sedzia_ab"])


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B instrukcji systemowej — obie strony błędu.")
    ap.add_argument("--warianty", nargs="+", default=list(WARIANTY), choices=list(WARIANTY))
    ap.add_argument("--repeats", type=int, default=3, help="powtórzenia GENERACJI per wariant")
    ap.add_argument("--tylko", choices=["zysk", "koszt"], help="zmierz tylko jedną stronę błędu")
    ap.add_argument("--sedzia", action="store_true",
                    help="A/B promptu SĘDZIEGO na zapisanych odpowiedziach (bez generacji)")
    ap.add_argument("--sedzia-powtorzenia", type=int, default=1,
                    help="ile razy ocenić TĘ SAMĄ odpowiedź (sędzia jest stochastyczny)")
    ap.add_argument("--pytania", nargs="+", help="ogranicz A/B sędziego do tych id pułapek")
    ap.add_argument("--raport", action="store_true", help="tabele z zapisanego pliku, bez GPU")
    args = ap.parse_args()

    if args.raport:
        if not OUT_PATH.exists():
            sys.exit(f"Brak {OUT_PATH} — uruchom najpierw pomiar.")
        raport(json.loads(OUT_PATH.read_text(encoding="utf-8")))
        return

    if args.sedzia:
        if not OUT_PATH.exists():
            sys.exit(f"Brak {OUT_PATH} — najpierw zmierz stronę zysku (potrzebne odpowiedzi).")
        dane = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        # Filtr po wariantach: sędzia jest drogi (2 prompty x każda teza), a rozstrzyga porównanie
        # v2 vs v3 — dokładanie v2bis/v4 kosztuje czas i nie zmienia wniosku.
        wiersze = [r for r in dane["zysk_rows"] if r["variant"] in args.warianty
                   and (not args.pytania or r["id"] in args.pytania)]
        dane["sedzia_ab"] = ab_sedziego(wiersze, load_traps(), args.sedzia_powtorzenia)
        OUT_PATH.write_text(json.dumps(dane, ensure_ascii=False, indent=2), encoding="utf-8")
        raport_sedziego(dane["sedzia_ab"])
        print(f"\nZapisano: {OUT_PATH.relative_to(config.ROOT)}")
        return

    print(f"A/B promptu: {config.LLM_MODEL} ({config.LLM_BACKEND}), warianty "
          f"{', '.join(args.warianty)}, {args.repeats} powtórzeń generacji\n")
    # Wczytujemy poprzedni wynik, żeby `--tylko` dokładało stronę błędu, a nie kasowało drugiej.
    dane = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    dane.update({"model": config.LLM_MODEL, "backend": config.LLM_BACKEND,
                 "repeats": args.repeats, "warianty": args.warianty})

    if args.tylko != "koszt":
        zysk = generuj_zysk(args.warianty, args.repeats)
        print("\nGuard #3 na zapisanych odpowiedziach (sędzia LLM)...")
        dolicz_guard3(zysk)
        dane["zysk_rows"] = zysk
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(dane, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.tylko != "zysk":
        print()
        dane["koszt_rows"] = generuj_koszt(args.warianty, args.repeats)
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(dane, ensure_ascii=False, indent=2), encoding="utf-8")

    raport(dane)
    print(f"\nZapisano: {OUT_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
