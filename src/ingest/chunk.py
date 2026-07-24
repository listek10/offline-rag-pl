"""
src/ingest/chunk.py
Etap 2 projektu RAG: chunking oczyszczonego tekstu (checkpoint #2).

DLACZEGO chunking (i to top pytanie rekrutera):
Nie indeksujemy całego raportu jako jednego wektora — embedding to jeden wektor o stałej
długości, więc wciśnięcie weń 50 stron "rozmywa" znaczenie (średnia z wielu tematów = wektor
o niczym). Tniemy tekst na spójne kawałki, żeby (1) retrieval zwracał trafny fragment, a nie
cały dokument, (2) dało się ten fragment zacytować, (3) kilka fragmentów zmieściło się w oknie LLM.

STRATEGIA (i dlaczego taka):
- Cięcie PO ZDANIACH (nie "co N znaków"): granice zdań to naturalne granice myśli. Cięcie na
  ślepo co N znaków ucina słowa/zdania w połowie -> chunk traci sens, a rozbite słowo psuje
  embedding. Cięcie po zdaniach ma najlepszy stosunek jakość/koszt. (Semantyczne cięcie modelem
  jest teoretycznie lepsze, ale wolne i trudne do obronienia jako "czarna skrzynka" — pomijamy na start.)
- OVERLAP (~15%): sąsiednie chunki nachodzą na siebie, by myśl na styku nie wypadła z obu.
  Koszt: redundancja (te same zdania w dwóch chunkach) i lekki wzrost liczby tokenów.

TRADE-OFF wielkości chunka:
- za duży  -> jeden wektor miesza wiele wątków, retrieval mniej precyzyjny, więcej szumu w promptcie;
- za mały  -> ginie kontekst ("wzrosły o 12%" bez sąsiedztwa nic nie mówi), więcej chunków = wolniej.
Parametry (CHUNK_TARGET_CHARS itd.) są w config/config.py i dobierzemy je TWARDO w etapie 5
(recall@k "przed/po"), a nie "na oko".

Uruchomienie:
    python src/ingest/chunk.py          # czyta data/processed/*.txt -> data/processed/chunks.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Import konfiguracji (jedno źródło prawdy dla parametrów).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config


# --- Dzielenie na zdania (świadomie lekkie, bez ciężkiego NLP) ---------------

# Skróty polskie zakończone kropką, które NIE kończą zdania. Bez tej listy "m.in.",
# "2022 r." czy "art. 76" dałyby fałszywe cięcia w środku zdania.
# TRADE-OFF: idealny podział zdań po polsku wymagałby modelu NLP (np. spaCy/stanza) —
# to cięższa zależność i "czarna skrzynka". Wybieramy prostą, przejrzystą regułę +
# listę skrótów: 95% jakości za ułamek kosztu i pełną wytłumaczalność.
_ABBREV = {
    "np", "tj", "tzn", "tzw", "itp", "itd", "m.in", "in", "por", "zob", "ww", "pkt",
    "r", "w", "art", "ust", "nr", "ok", "s", "str", "im", "mln", "mld", "tys", "proc",
    "dr", "prof", "inż", "gen", "płk", "mjr", "kpt", "hab", "red", "wyd", "ul", "al",
}

# Koniec zdania: jeden lub więcej [.!?…] a po nim biały znak. Grupujemy interpunkcję (1)
# i biały znak (2), żeby wiedzieć, gdzie kończy się zdanie, a gdzie zaczyna następne.
_SENT_END = re.compile(r"([.!?…]+)(\s+)")


def split_sentences(text: str) -> list[str]:
    """
    Dzieli tekst na zdania, respektując akapity (podwójny newline) i polskie skróty.
    """
    sentences: list[str] = []
    # Najpierw akapity: podwójny newline to twarda granica (nagłówek/akapit/strona).
    for para in re.split(r"\n{2,}", text):
        para = " ".join(para.split())  # normalizacja białych znaków w obrębie akapitu
        if not para:
            continue

        start = 0
        for m in _SENT_END.finditer(para):
            # Słowo bezpośrednio przed interpunkcją — decyduje, czy to prawdziwy koniec zdania.
            prefix = para[start:m.start(1)]
            tok = re.search(r"[\w.]+$", prefix)
            token = tok.group(0).lower() if tok else ""

            # Nie tnij, jeśli to skrót, pojedyncza litera (np. inicjał "K.") lub liczba ("2022").
            if token in _ABBREV or len(token) == 1 or token.replace(".", "").isdigit():
                continue

            sent = para[start:m.end(1)].strip()
            if sent:
                sentences.append(sent)
            start = m.end()  # pomiń biały znak po interpunkcji

        tail = para[start:].strip()  # ostatnie zdanie akapitu (często bez kończącej spacji)
        if tail:
            sentences.append(tail)

    return sentences


# --- Pakowanie zdań w chunki z zakładką ---------------------------------------

def pack_chunks(
    sentences: list[str],
    target: int,
    overlap: int,
    min_chars: int,
) -> list[str]:
    """
    Zachłannie dokłada zdania do chunka aż do ~target znaków, potem zaczyna nowy chunk,
    PRZENOSZĄC na jego początek ostatnie zdania (~overlap znaków). Overlap też jest
    liczony w całych zdaniach — spójnie z naszą strategią (nie tniemy w połowie zdania).
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for s in sentences:
        # Jeśli dołożenie zdania przekroczy target, a chunk już coś ma — zamknij chunk.
        if cur and cur_len + 1 + len(s) > target:
            chunks.append(" ".join(cur))

            # Zbuduj zakładkę: bierz zdania od końca, aż uzbierasz ~overlap znaków.
            keep: list[str] = []
            klen = 0
            for t in reversed(cur):
                if keep and klen + len(t) > overlap:
                    break
                keep.insert(0, t)
                klen += len(t) + 1
            cur, cur_len = keep, klen

        cur.append(s)
        cur_len += len(s) + 1

    # Resztka na końcu dokumentu.
    if cur:
        text = " ".join(cur)
        if len(text) >= min_chars or not chunks:
            chunks.append(text)
        else:
            # Zbyt krótki "ochłap" doklejamy do poprzedniego chunka, żeby nie gubić treści.
            chunks[-1] = chunks[-1] + " " + text

    return chunks


# --- Spięcie: oczyszczone .txt -> chunks.jsonl --------------------------------

def chunk_file(txt_path: Path) -> list[dict]:
    """Zamienia jeden oczyszczony .txt na listę rekordów-chunków z metadanymi."""
    text = txt_path.read_text(encoding="utf-8")
    sentences = split_sentences(text)
    chunks = pack_chunks(
        sentences,
        target=config.CHUNK_TARGET_CHARS,
        overlap=config.CHUNK_OVERLAP_CHARS,
        min_chars=config.CHUNK_MIN_CHARS,
    )

    source = txt_path.stem  # nazwa raportu = źródło do cytowania (na razie poziom dokumentu)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"{source}__{i}",     # unikalne, czytelne id (dokument + numer chunka)
            "source": source,            # do cytowania: "wg raportu <source>"
            "chunk_index": i,
            "n_chars": len(chunk),
            "text": chunk,
        })
    return records


def main() -> None:
    processed_dir = config.PROCESSED_DIR
    txt_files = sorted(processed_dir.glob("*.txt"))
    if not txt_files:
        print(f"[BŁĄD] Brak plików .txt w {processed_dir}. Najpierw uruchom parse_pdf.py.")
        sys.exit(1)

    all_records: list[dict] = []
    for txt in txt_files:
        recs = chunk_file(txt)
        all_records.extend(recs)
        sizes = [r["n_chars"] for r in recs]
        avg = sum(sizes) // len(sizes) if sizes else 0
        print(f"[OK] {txt.name}: {len(recs)} chunków "
              f"(śr. {avg} zn., min {min(sizes)}, max {max(sizes)})")

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[RAZEM] {len(all_records)} chunków -> {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
