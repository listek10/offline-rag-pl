"""
src/eval/ab_lexicon.py
A/B leksykonu guardu #4: forma jako DOWOLNY PODCIĄG (przed) vs PREFIKS SŁOWA (po).

SKĄD SIĘ WZIĄŁ TEN POMIAR: nie z przeglądu kodu, a z minera pułapek (`mine_traps.py`).
Miner miał tylko powiększyć zbiór pułapek, a wypluł kandydatów, w których „wzmianką Danii"
było słowo *badania*, a „wzmianką Finlandii" — *finansowanie*. Reguła dopasowania w
`ENTITY_FORMS` szukała formy jako podciągu W DOWOLNYM MIEJSCU słowa.

DLACZEGO TO JEST BŁĄD Z KONSEKWENCJĄ, A NIE BRZYDOTA: guard #4 przypisuje wielkość
NAJBLIŻSZEJ WCZEŚNIEJSZEJ wzmiance w zdaniu. Fałszywa wzmianka stojąca bliżej liczby wygrywa
z prawdziwą, więc błąd nie „lekko szumi" — on ODWRACA werdykt, i to w obie strony:
  * fałszywa wzmianka między podmiotem a liczbą  -> guard widzi cudzą wielkość  -> BLOKUJE
    poprawną odpowiedź (fałszywa odmowa),
  * fałszywa wzmianka pytanego podmiotu przy cudzej liczbie -> guard widzi „zgodną" wartość
    -> PRZEPUSZCZA podmianę.
Oba kierunki mierzymy osobno, bo to dwa różne koszty (ta sama zasada co w guardzie #3).

Uruchomienie (bez GPU, Qdranta i Ollamy — czysty tekst):
    python src/eval/ab_lexicon.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate import attribution as A

CHUNKS = config.PROCESSED_DIR / "chunks.jsonl"
ANSWERS = config.ROOT / "results" / "entity_traps_answers.json"
OUT = config.ROOT / "results" / "ab_lexicon.json"


def mentions_przed(text: str) -> list[tuple[int, str]]:
    """Stan PRZED naprawą — dokładnie ta reguła, która była w repo (podciąg, bez granicy słowa).

    Kopia zamiast flagi w produkcji: wariant „przed" ma być zamrożony w harnessie, żeby
    nie dało się go przypadkiem włączyć w runtime.
    """
    low = text.lower()
    out = []
    for ent, forms in _FORMS_PRZED.items():
        for f in forms:
            out.extend((m.start(), ent) for m in re.finditer(re.escape(f), low))
    return sorted(out)


_FORMS_PRZED: dict[str, tuple[str, ...]] = {
    "Dania":      ("dani", "duń", "duns", "kopenhag"),
    "Norwegia":   ("norwe", "norwes", "oslo"),
    "Szwecja":    ("szwecj", "szwed", "sztokholm"),
    "Finlandia":  ("finlandi", "fiń", "fin", "helsink"),
    "Islandia":   ("islandi", "islandz", "reykjav"),
    "Estonia":    ("estoni", "estoń", "eston", "tallin"),
    "Łotwa":      ("łotw", "łotew", "ryg"),
    "Litwa":      ("litw", "litew", "wilno", "wilnie"),
    "Kanada":     ("kanad", "ottaw"),
    "USA":        ("usa", "stany zjednoczone", "amerykan", "waszyngton"),
    "Rosja":      ("rosj", "rosyj", "moskw", "kreml"),
    "Niemcy":     ("niemc", "niemieck", "berlin"),
    "Polska":     ("polsk", "warszaw"),
    "Grenlandia": ("grenlandi", "grenlandz"),
}


def with_mentions(fn):
    """Podmienia regułę wzmianek w module produkcyjnym na czas jednego pomiaru.

    Guard woła `_mentions` wewnątrz `attributed_entities`, więc żeby zmierzyć wariant „przed"
    na PRODUKCYJNEJ ścieżce (a nie na jej przepisanej kopii), podmieniamy tę jedną funkcję.
    """
    class _Ctx:
        def __enter__(self):
            self.orig = A._mentions
            A._mentions = fn

        def __exit__(self, *exc):
            A._mentions = self.orig
    return _Ctx()


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]


def przypisania(chunks: list[dict]) -> dict[tuple[str, str], frozenset[str]]:
    """Komu korpus przypisuje każdą wielkość: (chunk_id, wartość) -> zbiór właścicieli."""
    out = {}
    for c in chunks:
        for v in A.quantities(c["text"]):
            out[(c["id"], v)] = frozenset(A.attributed_entities(c["text"], v))
    return out


def main() -> None:
    chunks = load_chunks()

    with with_mentions(mentions_przed):
        przed = przypisania(chunks)
        wzm_przed = sum(len(mentions_przed(c["text"])) for c in chunks)
    po = przypisania(chunks)
    wzm_po = sum(len(A._mentions(c["text"])) for c in chunks)

    zmienione = {k: (sorted(przed[k]), sorted(po[k])) for k in przed if przed[k] != po[k]}
    print("=== WZMIANKI PODMIOTÓW W KORPUSIE ===")
    print(f"  przed (podciąg):      {wzm_przed}")
    print(f"  po (prefiks słowa):   {wzm_po}   różnica: {wzm_po - wzm_przed}")

    print("\n=== PRZYPISANIE WIELKOŚCI (chunk × wartość) ===")
    print(f"  wielkości w korpusie:                       {len(przed)}")
    print(f"  z właścicielem PRZED / PO:                  "
          f"{sum(1 for v in przed.values() if v)} / {sum(1 for v in po.values() if v)}")
    print(f"  z JEDNOZNACZNYM właścicielem PRZED / PO:    "
          f"{sum(1 for v in przed.values() if len(v) == 1)} / "
          f"{sum(1 for v in po.values() if len(v) == 1)}")
    print(f"  ZMIENIONE przypisanie:                      {len(zmienione)}")

    # Rozbicie na to, co realnie znaczy dla werdyktu guardu.
    znikly = [k for k, (a, b) in zmienione.items() if a and not b]
    poprawione = [k for k, (a, b) in zmienione.items() if a and b and a != b]
    pojawily = [k for k, (a, b) in zmienione.items() if not a and b]
    print(f"    fałszywy właściciel ZNIKŁ (guard milknie): {len(znikly)}")
    print(f"    właściciel PODMIENIONY na innego:          {len(poprawione)}")
    print(f"    właściciel POJAWIŁ się (nowy zasięg):      {len(pojawily)}")

    print("\n  Przykłady zmian (do ręcznej weryfikacji — czy nowy właściciel jest PRAWDZIWY):")
    tekst = {c["id"]: c["text"] for c in chunks}
    for k in list(zmienione)[:12]:
        a, b = zmienione[k]
        frag = next((s for s in re.split(r"(?<=[.!?]) ", tekst[k[0]]) if k[1].replace(".", ",") in s
                     or k[1] in s), "")
        print(f"    {k[1]:>8}  {a or '—'} -> {b or '—'}   „{' '.join(frag.split())[:150]}\"")

    # Wpływ na WERDYKT guardu na zapisanych odpowiedziach z pomiaru etapu 10.
    werdykty = {}
    if ANSWERS.exists():
        rows = json.loads(ANSWERS.read_text(encoding="utf-8"))["rows"]
        print(f"\n=== WERDYKT GUARDU #4 na {len(rows)} zapisanych odpowiedziach (etap 10) ===")
        for r in rows:
            with with_mentions(mentions_przed):
                p = A.check_attribution(r["question"], r["answer"], r["context"])[0]
            n = A.check_attribution(r["question"], r["answer"], r["context"])[0]
            werdykty[r["id"]] = {"przed": p, "po": n}
            flaga = "  <- ZMIANA" if p is not n else ""
            print(f"  {r['id']} {r['entity']:10} przed={str(p):5} po={str(n):5}{flaga}")
    else:
        print(f"\n[uwaga] brak {ANSWERS.name} — pomiń kolumnę werdyktów")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "wzmianki": {"przed": wzm_przed, "po": wzm_po},
        "wielkosci": len(przed),
        "z_wlascicielem": {"przed": sum(1 for v in przed.values() if v),
                           "po": sum(1 for v in po.values() if v)},
        "jednoznaczne": {"przed": sum(1 for v in przed.values() if len(v) == 1),
                         "po": sum(1 for v in po.values() if len(v) == 1)},
        "zmienione": {f"{k[0]}|{k[1]}": {"przed": a, "po": b} for k, (a, b) in zmienione.items()},
        "werdykty_guardu4": werdykty,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano: {OUT.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
