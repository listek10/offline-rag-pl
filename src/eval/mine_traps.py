"""
src/eval/mine_traps.py
MINER KANDYDATÓW NA PUŁAPKI — powiększenie `gold_set/entity_traps.jsonl` z n=7 do uczciwego n.

PO CO: dwa wyniki tego projektu stoją na próbce, której nie da się obronić. Wynik negatywny
guardu #4 (etap 10) opiera się na 7 pytaniach, z czego podmiana zaszła realnie w JEDNYM.
Decyzja o prompcie sędziego (etap 11) stoi na DWÓCH pytaniach. Uczciwie mogę dziś powiedzieć
tylko „nie zadziałał tam, gdzie mogłem zmierzyć".

DLACZEGO SZUKAM KODEM GUARDU, A NIE GŁOWĄ (to jest cała idea tego narzędzia):
gdybym wymyślał pytania sam i dopiero potem sprawdzał, czy wpadają w zakres guardu #4,
zbiór byłby mieszanką typów, a wynik znowu rozmyty (tak powstało n=7: 6 pytań, na których
guard ABSTYNUJE, i jedno rozstrzygające). Dlatego kandydatów wybieram **funkcjami
produkcyjnymi guardu** — `quantities()` i `attributed_entities()` z `src/generate/attribution.py`.
Konsekwencja jest mocna: „typ bałtycki" (wielkość i podmiot w JEDNYM zdaniu) przestaje być
moją oceną, a staje się definicją operacyjną — kandydat przechodzi wtedy i tylko wtedy, gdy
guard umie wydać werdykt. Pytania typu polarnego (anafora „wyniosły ONE 8,4 mld") wypadają
same, bo tam `attributed_entities` nie zwraca właściciela.

CZEGO CELOWO NIE ROBIĘ: nie rozszerzam `ENTITY_FORMS`, żeby kandydatów było więcej.
Leksykon jest CZĘŚCIĄ mierzonego guardu — dopasowanie go do zbioru testowego to strojenie
guardu pod test. Podmioty spoza leksykonu wypadają, i to jest zapisany koszt, nie usterka.

DWA FILTRY (każdy odsiewa pytanie, które NIC by nie zmierzyło):
1. JEDNOZNACZNOŚĆ — wielkość musi mieć w chunku dokładnie jednego właściciela. Gdy korpus
   przypisuje ją kilku podmiotom, guard oznacza ją „wieloznaczna — nie rozstrzyga" i nie
   głosuje (patrz `check_attribution`), więc pytanie nie wnosi informacji o guardzie.
2. KONKURENT — w tym samym chunku musi stać para (inny podmiot, inna wielkość). Bez niej
   nie ma czego podmienić: model nie ma w kontekście cudzej liczby, którą mógłby skłamać.
   Siłę pułapki liczę po konkurentach w TEJ SAMEJ JEDNOSTCE (mld przy mld, % przy %) —
   „8,4 mld" i „1,3%" mieszają się modelowi znacznie rzadziej niż „8,4 mld" i „5,3 mld".

Cytat (`evidence`) jest brany ZNAK W ZNAK z chunka i weryfikowany zwrotnie w korpusie —
ten sam wymóg co dla gold setu (`relevant_by_evidence`).

Uruchomienie (nie wymaga GPU, Qdranta ani Ollamy — czysty odczyt korpusu):
    python src/eval/mine_traps.py                       # kandydaci: konkurent w tym samym chunku
    python src/eval/mine_traps.py --rywal dokument      # kandydaci: konkurent w dokumencie (szerzej)
    python src/eval/mine_traps.py --waliduj             # kontrola jakości danych zbioru pułapek
    python src/eval/mine_traps.py --entity Estonia --json

ZMIERZONY ZASIĘG KORPUSU (dlaczego zbiór ma 21 pułapek, a nie 50): 327 wystąpień wielkości
w 1497 chunkach, z czego 79 ma w zdaniu jednoznacznego właściciela, a tylko 10 par ma konkurenta
w TYM SAMYM chunku (3 z nich już były w zbiorze). Warunek dokumentowy podnosi to do 35 par.
Korpus pięciu raportów po prostu nie zawiera więcej zdań typu bałtyckiego — to granica danych,
nie minera, i lepiej ją zapisać niż udawać większe `n` pytaniami, które nic nie mierzą.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate.attribution import ENTITY_FORMS, _UNIT, attributed_entities, quantities
from src.ingest.chunk import split_sentences

CHUNKS = config.PROCESSED_DIR / "chunks.jsonl"
TRAPS_PATH = config.ROOT / "gold_set" / "entity_traps.jsonl"
OUT = config.ROOT / "results" / "trap_candidates.json"


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]


def unit_of(text: str, value: str) -> str | None:
    """Jednostka stojąca przy wielkości — potrzebna do oceny, czy konkurent jest realny.

    Ta sama osłona wzorca co w `_quantity_positions` (żeby „1,3" nie trafiło w „11,3"),
    ale zwracamy jednostkę, nie pozycję.
    """
    num = re.escape(value).replace(r"\.", "[.,]")
    m = re.search(rf"(?<![\d,.]){num}(?![\d])\s*({_UNIT})", text)
    return m.group(1) if m else None


def pairs_in_chunk(text: str) -> list[dict]:
    """Pary (podmiot, wielkość) JEDNOZNACZNIE przypisane w tym chunku — oczami guardu #4.

    Nie reimplementuję przypisania: wołam `attributed_entities()` z produkcji. Dzięki temu
    „kandydat w zasięgu guardu" znaczy dokładnie to samo, co znaczy w runtime.
    """
    out = []
    for value in sorted(quantities(text)):
        owners = attributed_entities(text, value)
        if len(owners) != 1:          # filtr 1: wieloznaczna wielkość nie rozstrzyga niczego
            continue
        out.append({"entity": next(iter(owners)), "value": value, "unit": unit_of(text, value)})
    return out


def sentence_with(text: str, value: str) -> str | None:
    """Zdanie z chunka zawierające tę wielkość — kandydat na `evidence` (znak w znak)."""
    num = re.escape(value).replace(r"\.", "[.,]")
    pat = re.compile(rf"(?<![\d,.]){num}(?![\d])\s*{_UNIT}")
    for sent in split_sentences(text):
        if pat.search(sent):
            return sent
    return None


_STOP = {"i", "w", "na", "z", "o", "do", "od", "za", "po", "przez", "dla", "oraz", "a", "ale",
         "to", "że", "jest", "są", "ma", "mają", "być", "tym", "ten", "ta", "te", "się", "nie",
         "jak", "co", "przy", "ok", "tys", "mln", "mld", "proc", "r", "tj", "czyli", "wynosi",
         "wyniosły", "wynoszą", "wyniósł", "których", "który", "która", "które", "ich", "jego"}


def measure_words(sent: str) -> set[str]:
    """Słowa treściowe zdania — czym MIERZY ta wielkość (schrony? wydatki? żołnierze?).

    Po co: konkurent w tej samej jednostce, ale INNEJ mierze („45 tys. schronów" vs „15 tys.
    żołnierzy") jest słabą pułapką — model nie ma powodu ich pomylić. Pułapka jest realna, gdy
    dwa podmioty mają wielkość TEJ SAMEJ miary. Nie filtruję po tym automatem (za łatwo o błąd
    w obie strony), tylko RANKUJĘ kandydatów, a ostateczny wybór jest ręczny i zapisany.
    """
    return {w for w in re.findall(r"[a-ząćęłńóśźż]{4,}", sent.lower()) if w not in _STOP}


def mine_document(only_entity: str | None = None) -> list[dict]:
    """Kandydaci przy warunku REALNYM: konkurent gdziekolwiek w TYM SAMYM DOKUMENCIE.

    DLACZEGO NIE „W TYM SAMYM CHUNKU": pułapka zachodzi, gdy cudza wielkość jest w KONTEKŚCIE
    podanym modelowi, a kontekst to top_k=5 chunków po rerankingu — nie jeden chunk. Warunek
    chunkowy zmierzyłem i jest za ostry: przepuszcza 10 par na całym korpusie, z czego 3 już są
    w zbiorze. Warunek dokumentowy jest przybliżeniem od góry (retrieval MOŻE nie przynieść
    konkurenta), więc spełnienie warunku sprawdzam potem NA ZAPISANYM KONTEKŚCIE (`--audyt`
    w eval_traps.py) i pytania bez konkurenta w kontekście wypadają z mianownika podmiany.
    """
    chunks = load_chunks()
    pool: dict[tuple[str, str], dict] = {}
    for c in chunks:
        for p in pairs_in_chunk(c["text"]):
            sent = sentence_with(c["text"], p["value"])
            if sent is None:
                continue
            key = (p["entity"], p["value"])
            # Przy zakładce ta sama para siedzi w dwóch chunkach — bierzemy KRÓTSZE zdanie
            # (mniejsze ryzyko, że cytat przeszedł przez granicę chunka i jest urwany).
            if key not in pool or len(sent) < len(pool[key]["evidence"]):
                pool[key] = {"entity": p["entity"], "value": p["value"], "unit": p["unit"],
                             "chunk_id": c["id"], "source": c["source"], "evidence": sent,
                             "slowa": measure_words(sent)}
    out = []
    for cand in pool.values():
        if only_entity and cand["entity"] != only_entity:
            continue
        rivals = []
        for other in pool.values():
            if (other["source"] != cand["source"] or other["entity"] == cand["entity"]
                    or other["value"] == cand["value"] or other["unit"] != cand["unit"]):
                continue
            wspolne = cand["slowa"] & other["slowa"]
            rivals.append({"entity": other["entity"], "value": other["value"],
                           "wspolne_slowa": sorted(wspolne), "chunk_id": other["chunk_id"]})
        if not rivals:
            continue
        rivals.sort(key=lambda r: -len(r["wspolne_slowa"]))
        c2 = {k: v for k, v in cand.items() if k != "slowa"}
        c2["rywale"] = rivals[:6]
        # Siła = ile słów treściowych ma wspólnych z NAJLEPSZYM konkurentem (ta sama miara?).
        c2["sila"] = len(rivals[0]["wspolne_slowa"])
        out.append(c2)
    return sorted(out, key=lambda d: (-d["sila"], d["entity"]))


def mine(only_entity: str | None = None) -> list[dict]:
    chunks = load_chunks()
    # Kandydat jest własnością PARY (podmiot, wielkość), nie chunka: przy 15% zakładce ten sam
    # fragment siedzi w dwóch chunkach i bez deduplikacji dostałbym to samo pytanie dwa razy.
    best: dict[tuple[str, str], dict] = {}
    for c in chunks:
        pairs = pairs_in_chunk(c["text"])
        if len(pairs) < 2:
            continue
        for p in pairs:
            if only_entity and p["entity"] != only_entity:
                continue
            rivals = [q for q in pairs if q["entity"] != p["entity"] and q["value"] != p["value"]]
            if not rivals:            # filtr 2: nie ma czego podmienić
                continue
            same_unit = [q for q in rivals if q["unit"] == p["unit"]]
            sent = sentence_with(c["text"], p["value"])
            if sent is None:
                continue
            cand = {
                "entity": p["entity"], "value": p["value"], "unit": p["unit"],
                "chunk_id": c["id"], "source": c["source"],
                "evidence": sent,
                "rywale": [f"{q['entity']}:{q['value']}{q['unit'] or ''}" for q in rivals],
                "rywale_ta_jednostka": [f"{q['entity']}:{q['value']}{q['unit'] or ''}"
                                        for q in same_unit],
                # Siła pułapki = ilu konkurentów w tej samej jednostce. Zero znaczy: technicznie
                # pułapka, ale model musiałby pomylić mld z procentami, żeby w nią wpaść.
                "sila": len(same_unit),
            }
            key = (p["entity"], p["value"])
            if key not in best or cand["sila"] > best[key]["sila"]:
                best[key] = cand
    return sorted(best.values(), key=lambda d: (-d["sila"], d["entity"], d["value"]))


def already_used() -> set[tuple[str, str]]:
    """Pary już obecne w zbiorze pułapek — żeby nie proponować tego samego drugi raz."""
    if not TRAPS_PATH.exists():
        return set()
    used = set()
    for line in TRAPS_PATH.open(encoding="utf-8"):
        t = json.loads(line)
        from src.eval.metrics import extract_values
        for v in extract_values(t["answer"]):
            used.add((t["entity"], v))
    return used


def validate(cands: list[dict]) -> list[str]:
    """Cytat MUSI występować w korpusie znak w znak — ten sam wymóg co dla gold setu."""
    raw = "\n".join(c["text"] for c in load_chunks())
    return [c["evidence"] for c in cands if c["evidence"] not in raw]


SUBTYPES = {"podmiana-podmiotu", "rola-wartosci"}


def waliduj_zbior() -> int:
    """Kontrola JAKOŚCI DANYCH zbioru pułapek. Zwraca liczbę błędów (0 = zbiór zdatny do pomiaru).

    DLACZEGO TO NIE JEST NADGORLIWOŚĆ: pułapka z błędem w danych daje wynik pomiaru, który
    wygląda jak wynik o systemie, a jest wynikiem o moim zbiorze. Najgroźniejsze pomyłki są ciche:
    cytat przepisany ręcznie (inny półpauza, doklejony numer przypisu) nie wskazuje żadnego chunka;
    oczekiwana wartość, której NIE MA w cytacie, robi z poprawnej odpowiedzi „inna-bledna";
    a pytanie, które nie nazywa podmiotu z leksykonu, wyprowadza guard #4 poza zakres — czyli
    pytanie nie mierzy tego, po co je dodałem.

    Sprawdzamy pięć rzeczy (każda odpowiada realnej pomyłce, którą da się popełnić):
      1. cytat wskazuje co najmniej jeden chunk i wśród nich jest zadeklarowany `source_chunk`,
      2. każda wartość z oczekiwanej odpowiedzi występuje w cytacie (prawda jest w korpusie),
      3. wartości pułapki są ROZŁĄCZNE z oczekiwanymi (inaczej metryka nie odróżni klas),
      4. wartości pułapki istnieją w korpusie jako wielkości (to CZYJAŚ liczba, nie wymyślona),
      5. `entity_in(question)` == zadeklarowany podmiot (guard #4 jest w zasięgu).
    """
    from src.eval.metrics import extract_values, relevant_by_evidence
    from src.generate.attribution import entity_in

    chunks = load_chunks()
    # DWA różne zbiory, bo dwa różne pytania. `korpus_wielkosci` (liczba + jednostka z leksykonu
    # guardu) służy do sprawdzenia, czy wartość pułapki jest wielkością, którą guard w ogóle
    # widzi. `korpus_liczby` (wszystkie liczby, ekstraktorem METRYKI) służy do sprawdzenia, czy
    # oczekiwana wartość istnieje w korpusie — bo metryka nie zna jednostek i porównuje liczby.
    # Pomieszanie ich dawało fałszywy alarm na „900 km2" (km2 nie jest jednostką guardu)
    # i na latach „2021-2027" (rok celowo nie jest wielkością).
    korpus_wielkosci, korpus_liczby = set(), set()
    for c in chunks:
        korpus_wielkosci |= quantities(c["text"])
        korpus_liczby |= extract_values(c["text"])

    traps = [json.loads(line) for line in TRAPS_PATH.open(encoding="utf-8")]
    bledy = 0
    # Nazwa pliku bez `relative_to`: testy podają walidatorowi plik z katalogu tymczasowego,
    # a `relative_to(ROOT)` rzucałby wtedy wyjątkiem — walidator ma działać na DOWOLNYM zbiorze.
    print(f"=== WALIDACJA {TRAPS_PATH} ({len(traps)} pułapek) ===")
    if len({t["id"] for t in traps}) != len(traps):
        print("  [BŁĄD] powtórzone id"); bledy += 1

    for t in traps:
        problemy = []
        trafione = relevant_by_evidence(chunks, t["evidence"])
        if not trafione:
            problemy.append("cytat nie występuje w korpusie (znak w znak, po normalizacji odstępów)")
        elif t["source_chunk"] not in trafione:
            problemy.append(f"source_chunk {t['source_chunk']} nie zawiera cytatu (jest w {trafione})")

        # Oczekiwana odpowiedź może dopuszczać KILKA prawdziwych liczb, jeśli korpus podaje
        # kilka (np. duńskie roszczenia szelfowe: 0,9 mln km2 w podziale „arktycznego tortu"
        # ORAZ 900 km2 na północ od wyspy — obie o Danii, w dwóch różnych zdaniach). Metryka
        # zalicza trafienie w JEDNĄ z nich, więc wymagam: co najmniej jedna wartość w cytacie
        # (pułapka ma swój dowód) i KAŻDA obecna w korpusie (nie wolno wpisać liczby z sufitu).
        want = extract_values(t["answer"])
        w_cytacie = want & extract_values(t["evidence"])
        poza_korpusem = want - korpus_liczby
        if not want:
            problemy.append("oczekiwana odpowiedź nie ma żadnej wartości liczbowej")
        elif not w_cytacie:
            problemy.append(f"żadna wartość z odpowiedzi nie jest w cytacie: {sorted(want)}")
        if poza_korpusem:
            problemy.append(f"wartości z odpowiedzi nie ma w korpusie: {sorted(poza_korpusem)}")

        wspolne = want & set(t["trap_values"])
        if wspolne:
            problemy.append(f"wartość pułapki = wartość poprawna: {sorted(wspolne)}")
        obce = set(t["trap_values"]) - korpus_wielkosci
        if obce:
            problemy.append(f"wartości pułapki nie ma w korpusie: {sorted(obce)}")

        w_zasiegu = entity_in(t["question"])
        if w_zasiegu != t["entity"]:
            problemy.append(f"pytanie wskazuje podmiot {w_zasiegu}, a zadeklarowano {t['entity']}")
        if t["subtype"] not in SUBTYPES:
            problemy.append(f"nieznany subtype: {t['subtype']}")

        # Ostrzeżenie, nie błąd: wartość „2" jest słabo rozróżniająca, bo trafi w każdą inną „2"
        # w odpowiedzi. Zapisuję to jawnie, żeby nie udawać, że wszystkie pytania są równie mocne.
        slabe = [v for v in want if "." not in v and len(v) == 1]
        print(f"  {t['id']} {t['entity']:11} {'OK' if not problemy else 'BŁĄD'}"
              f"{'  [uwaga: słabo rozróżniająca wartość ' + ','.join(slabe) + ']' if slabe else ''}")
        for p in problemy:
            print(f"      - {p}")
        bledy += len(problemy)

    print(f"\n  błędów: {bledy}")
    return bledy


def main() -> None:
    ap = argparse.ArgumentParser(description="Miner kandydatów na pułapki (podmiana podmiotu).")
    ap.add_argument("--entity", help="tylko ten podmiot kanoniczny")
    ap.add_argument("--json", action="store_true", help="zapisz results/trap_candidates.json")
    ap.add_argument("--min-sila", type=int, default=0, help="minimum konkurentów w tej samej jednostce")
    ap.add_argument("--rywal", choices=["chunk", "dokument"], default="chunk",
                    help="gdzie musi stać konkurencyjna wielkość (chunk = ostro, dokument = realnie)")
    ap.add_argument("--waliduj", action="store_true",
                    help="sprawdź jakość danych gold_set/entity_traps.jsonl (kod wyjścia != 0 przy błędzie)")
    args = ap.parse_args()

    if args.waliduj:
        sys.exit(1 if waliduj_zbior() else 0)

    miner = mine_document if args.rywal == "dokument" else mine
    cands = [c for c in miner(args.entity) if c["sila"] >= args.min_sila]
    used = already_used()
    for c in cands:
        c["już_w_zbiorze"] = (c["entity"], c["value"]) in used

    braki = validate(cands)
    if braki:
        sys.exit(f"[BŁĄD] {len(braki)} cytatów nie ma w korpusie znak w znak: {braki[:2]}")

    print(f"Korpus: {len(load_chunks())} chunków, leksykon: {len(ENTITY_FORMS)} podmiotów")
    print(f"Kandydaci (jednoznaczne przypisanie + konkurent w chunku): {len(cands)}"
          f" | nowi: {sum(not c['już_w_zbiorze'] for c in cands)}")
    per_ent: dict[str, int] = defaultdict(int)
    for c in cands:
        per_ent[c["entity"]] += 1
    print("  wg podmiotu: " + ", ".join(f"{k}={v}" for k, v in sorted(per_ent.items())))
    print(f"  z konkurentem w tej samej jednostce (sila>0): {sum(c['sila'] > 0 for c in cands)}\n")

    for i, c in enumerate(cands, 1):
        flag = " [JUŻ W ZBIORZE]" if c["już_w_zbiorze"] else ""
        if args.rywal == "dokument":
            riv = "; ".join(f"{r['entity']}:{r['value']}{c['unit'] or ''}"
                            f"({'+'.join(r['wspolne_slowa'][:4]) or 'brak wspólnych słów'})"
                            for r in c["rywale"][:3])
        else:
            riv = ",".join(c["rywale_ta_jednostka"]) or "(inna jednostka: " + ",".join(c["rywale"]) + ")"
        print(f"{i:3}. {c['entity']:11} {c['value']}{c['unit'] or '':6} sila={c['sila']} {riv}{flag}")
        print(f"     {c['chunk_id']}")
        print(f"     „{' '.join(c['evidence'].split())[:260]}\"")

    if args.json:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"n": len(cands), "kandydaci": cands},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nZapisano: {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
