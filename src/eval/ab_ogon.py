"""
src/eval/ab_ogon.py
Etap 14 / checkpointy #2 i #12: OGON ROZKŁADU DŁUGOŚCI CHUNKÓW — pomiar zamiast średniej.

PROBLEM, który to mierzy: wgrany w UI raport pisany WYPUNKTOWANIAMI (`IV ROK WOJNY`, 100 stron)
dawał chunki do 5722 znaków, choć target to 1200. Średnia tego nie pokazywała (1105 vs 1093 zn.
na korpusie bazowym — praktycznie tyle samo), bo patologia siedzi w OGONIE rozkładu, nie w
środku. Skutek nie był kosmetyczny: koszt guarda #3 to liczba tez × wielkość kontekstu, więc na
wielkim chunku weryfikacja trwała 264 s (vs 26,7 s) i odrzucała poprawną odpowiedź.

DWIE PRZYCZYNY (obie ujawnione pomiarem, nie z lektury kodu):
1. `split_sentences` szuka [.!?…] — lista bez kropek wraca jako JEDNO „zdanie" (2867 zn.);
2. pętla zakładki w `pack_chunks` brała ZAWSZE co najmniej jedno zdanie, choćby olbrzymie,
   więc moloch wchodził do następnego chunka jako zakładka: 2867 + 2854 = 5722 zn.
Sama poprawka (1) zostawiała chunk 2417 zn. — dlatego naprawa musiała być dwuczęściowa.

CO ROBIĄ TRYBY:
  --rozklad   rozkład długości chunków PRZED/PO na wskazanym dokumencie (lub korpusie bazowym).
              „Przed" odtwarzamy JAWNIE, przez wyłączenie obu progów, a nie z pamięci.
  --zasieg    DOWÓD ZASIĘGU na korpusie bazowym: ile chunków ruszyło i czy ruszony chunk dotyka
              cytatu `evidence` z gold setu. Ten sam chwyt dowodowy co w etapie 13: jeśli zbiór
              trafnych chunków się nie zmienił, prawda pomiaru się nie przesunęła.

              UWAGA NA GRANICĘ TEGO DOWODU (łatwo go przehandlować za mocniejszy, niż jest):
              nietknięte cytaty gwarantują stałą PRAWDĘ, ale nie stały recall — wymiana
              dystraktorów może przepchnąć obcy chunk przed trafny. Dlatego `--zasieg` nie
              zastępuje `run_eval.py`, tylko ogranicza ryzyko do rankingu (i to się ziściło:
              w dense `g02` spadł z pozycji 7 na 8, w produkcyjnym hybrid+rr nic).

Uruchomienie (nie wymaga GPU ani Qdranta — to czysty chunking):
    python src/eval/ab_ogon.py --rozklad                       # korpus bazowy
    python src/eval/ab_ogon.py --rozklad "data/raw_deferred/IV ROK WOJNY_2.pdf"
    python src/eval/ab_ogon.py --zasieg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.ingest.chunk import pack_chunks, split_overlong, split_sentences

RESULTS = config.ROOT / "results"
BEZ_LIMITU = 10 ** 9   # „przed": progi wyłączone, czyli zachowanie sprzed etapu 14


def potnij(text: str, max_sentence: int, carry_max: int) -> list[str]:
    """Cały tor chunkingu z JAWNYMI progami — żeby „przed" i „po" różniły się tylko nimi."""
    zdania = [
        part
        for sent in split_sentences(text)
        for part in split_overlong(sent, max_chars=max_sentence, target=config.CHUNK_TARGET_CHARS)
    ]
    return pack_chunks(zdania, target=config.CHUNK_TARGET_CHARS,
                       overlap=config.CHUNK_OVERLAP_CHARS, min_chars=config.CHUNK_MIN_CHARS,
                       carry_max=carry_max)


def rozklad(chunks: list[str]) -> dict:
    """Statystyki ogona. Percentyl 99 i max są tu ważniejsze od średniej — patrz docstring."""
    n = sorted(len(c) for c in chunks)
    udzial = lambda prog: round(100 * sum(1 for x in n if x > prog) / len(n), 2)
    return {
        "n_chunks": len(n),
        "avg": round(sum(n) / len(n)),
        "p50": n[len(n) // 2],
        "p99": n[int(len(n) * 0.99)],
        "max": n[-1],
        "pct_ponad_1500": udzial(1500),
        "pct_ponad_2000": udzial(2000),
    }


def wczytaj(zrodlo: str | None) -> tuple[str, str]:
    """Zwraca (etykieta, tekst). Bez argumentu: korpus bazowy z data/processed/*.txt."""
    if zrodlo is None:
        pliki = sorted(config.PROCESSED_DIR.glob("*.txt"))
        if not pliki:
            sys.exit(f"[BŁĄD] Brak .txt w {config.PROCESSED_DIR} — uruchom parse_pdf.py.")
        return f"korpus bazowy ({len(pliki)} raportów)", "\n\n".join(
            p.read_text(encoding="utf-8") for p in pliki)

    p = Path(zrodlo)
    if p.suffix.lower() == ".pdf":
        # Import lokalny: PyMuPDF nie jest potrzebny do trybu --zasieg ani do korpusu bazowego,
        # więc harness ma działać także tam, gdzie go nie ma (CI bez GPU/pymupdf).
        from src.ingest.parse_pdf import clean_pages, extract_pages
        return p.name, clean_pages(extract_pages(p))
    return p.name, p.read_text(encoding="utf-8")


def tryb_rozklad(zrodlo: str | None) -> dict:
    etykieta, text = wczytaj(zrodlo)
    przed = rozklad(potnij(text, BEZ_LIMITU, BEZ_LIMITU))
    po = rozklad(potnij(text, config.CHUNK_MAX_SENTENCE_CHARS, config.CHUNK_CARRY_MAX_CHARS))
    # Sama poprawka (1) — pokazuje, że jedna warstwa naprawy NIE wystarczała.
    tylko_zdania = rozklad(potnij(text, config.CHUNK_MAX_SENTENCE_CHARS, BEZ_LIMITU))

    print(f"\n=== ROZKŁAD DŁUGOŚCI CHUNKÓW — {etykieta} ===")
    print(f"target {config.CHUNK_TARGET_CHARS} zn. | limit zdania {config.CHUNK_MAX_SENTENCE_CHARS} "
          f"| kap zakładki {config.CHUNK_CARRY_MAX_CHARS}")
    kol = ("n_chunks", "avg", "p50", "p99", "max", "pct_ponad_1500", "pct_ponad_2000")
    print(f"{'wariant':34}" + "".join(f"{k:>16}" for k in kol))
    for nazwa, w in (("PRZED (oba progi wyłączone)", przed),
                     ("tylko limit zdania", tylko_zdania),
                     ("PO (limit + kap zakładki)", po)):
        print(f"{nazwa:34}" + "".join(f"{w[k]:>16}" for k in kol))
    print("\n  Średnia jest tu bezużyteczna — patrz max i p99. Wiersz środkowy pokazuje, że")
    print("  naprawa musiała mieć dwie warstwy: sam limit zdania ogona nie domykał.")
    return {"zrodlo": etykieta, "przed": przed, "tylko_limit_zdania": tylko_zdania, "po": po}


def tryb_zasieg() -> dict:
    """Ile chunków korpusu bazowego ruszyło i czy ruszony chunk dotyka cytatu z gold setu."""
    _, text = wczytaj(None)
    przed = potnij(text, BEZ_LIMITU, BEZ_LIMITU)
    po = potnij(text, config.CHUNK_MAX_SENTENCE_CHARS, config.CHUNK_CARRY_MAX_CHARS)
    wspolne = set(przed) & set(po)
    zniknely = [c for c in przed if c not in wspolne]
    nowe = [c for c in po if c not in wspolne]

    gold = [json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8") if l.strip()]
    # 7 pytań `poza-korpusem` nie ma cytatu — mierzą odmowę, nie retrieval.
    gold = [g for g in gold if "evidence" in g]

    dotkniete, biletyzm = [], []
    for g in gold:
        ev = g["evidence"]
        if any(ev in c for c in zniknely + nowe):
            dotkniete.append(g["id"])
        a, b = sum(1 for c in przed if ev in c), sum(1 for c in po if ev in c)
        if a != b:
            biletyzm.append({"id": g["id"], "przed": a, "po": b})

    wynik = {
        "chunkow_przed": len(przed), "chunkow_po": len(po),
        "identycznych": len(wspolne),
        "pct_identycznych": round(100 * len(wspolne) / len(przed), 1),
        "zniknelo": len(zniknely), "pojawilo_sie": len(nowe),
        "max_przed": max(len(c) for c in przed), "max_po": max(len(c) for c in po),
        "gold_pytan": len(gold),
        "cytat_w_ruszonym_chunku": dotkniete,
        "zmiana_liczby_trafnych_chunkow": biletyzm,
    }
    print(f"\n=== ZASIĘG ZMIANY NA KORPUSIE BAZOWYM ===")
    print(f"  chunków: {wynik['chunkow_przed']} -> {wynik['chunkow_po']}   "
          f"(identycznych znak w znak: {wynik['identycznych']}, {wynik['pct_identycznych']}%)")
    print(f"  zniknęło {wynik['zniknelo']}, pojawiło się {wynik['pojawilo_sie']}; "
          f"max {wynik['max_przed']} -> {wynik['max_po']} zn.")
    print(f"  gold set ({wynik['gold_pytan']} pytań z cytatem):")
    print(f"    cytat leży w ruszonym chunku: {dotkniete or 'ŻADEN'}")
    print(f"    zmiana liczby trafnych chunków: {biletyzm or 'ŻADNA'}")
    if not dotkniete:
        print("\n  Wniosek: zbiór trafnych chunków jest nietknięty, więc PRAWDA pomiaru się nie")
        print("  przesunęła. To NIE dowodzi stałego recall (dystraktory się wymieniły) —")
        print("  ranking sprawdza dopiero `run_eval.py --truth evidence`.")
    return wynik


def main() -> None:
    ap = argparse.ArgumentParser(description="Ogon rozkładu długości chunków (etap 14)")
    ap.add_argument("--rozklad", nargs="?", const=None, default="__brak__",
                    metavar="PLIK", help="rozkład przed/po (bez argumentu: korpus bazowy)")
    ap.add_argument("--zasieg", action="store_true", help="dowód zasięgu vs gold set")
    ap.add_argument("--json", dest="json_out", help="zapisz wynik do pliku JSON")
    a = ap.parse_args()

    wynik: dict = {}
    if a.rozklad != "__brak__":
        wynik["rozklad"] = tryb_rozklad(a.rozklad)
    if a.zasieg:
        wynik["zasieg"] = tryb_zasieg()
    if not wynik:
        ap.error("wybierz --rozklad i/lub --zasieg")

    if a.json_out:
        out = Path(a.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(wynik, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nZapisano -> {out}")


if __name__ == "__main__":
    main()
