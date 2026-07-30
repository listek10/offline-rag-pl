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


# --- Reguła awaryjna: za długie "zdanie" (dokumenty wypunktowane) -------------

# Granica punktu listy: pozycja PRZED znacznikiem wypunktowania (•, ▪, ‣) albo przed
# myślnikiem rozpoczynającym podpunkt ("- wzmocnienie…"). `(?<=\s)` pilnuje, żeby nie ciąć
# na myślniku WEWNĄTRZ wyrazu ("polsko-niemiecki") ani w liczbie ("2022–2025").
_BULLET = re.compile(r"(?<=\s)(?=[•▪‣]|[-–—]\s)")


def split_overlong(sent: str, max_chars: int, target: int) -> list[str]:
    """Dzieli JEDNO za długie „zdanie" na kawałki ≤ max_chars. Poniżej limitu zwraca [sent].

    DLACZEGO TO ISTNIEJE (przyczyna zmierzona, nie domyślona): `split_sentences` szuka końca
    zdania jako [.!?…] + biały znak. Raport pisany WYPUNKTOWANIAMI nie ma kropek na końcu
    punktów, więc cały akapit-lista wraca jako jedno „zdanie" (zmierzone: 2867 zn. na raporcie
    `IV ROK WOJNY`). `pack_chunks` z zasady nie tnie w środku zdania, więc taki moloch wchodził
    do chunka w całości. Skutki były policzone: 3% chunków >2000 zn., generacja 43,5 s i
    weryfikacja guardem #3 aż 264 s (vs 26,7 s na korpusie bazowym), bo koszt guarda to
    liczba tez × wielkość kontekstu.

    DLACZEGO OSOBNA FUNKCJA, A NIE POPRAWKA W `split_sentences`: tamta jest współdzielona
    z guardem #3 (`verify.to_claims`), guardem #4 (`attribution.py`) i harnessem z etapu 13.
    Zmiana w niej po cichu przedefiniowałaby „tezę" u sędziego i unieważniła tamte pomiary.
    Reguła awaryjna należy do INGESTU i tylko do niego.

    DLACZEGO PRÓG, A NIE ZAWSZE: to zawór bezpieczeństwa, nie nowa strategia cięcia. Zdanie
    pod limitem wraca BIT W BIT — dzięki temu naprawa nie rusza korpusu, na którym zmierzone
    są liczby (recall@1, MRR, A/B chunkingu z etapu 8), poza policzonymi wyjątkami.
    """
    if len(sent) <= max_chars:
        return [sent]

    # 1. Najpierw granice naturalne: punkty listy. Sklejamy drobne punkty z powrotem
    #    do ~target, bo lista pojedynczych myślników to sieczka bez kontekstu.
    parts: list[str] = []
    cur = ""
    for p in (x.strip() for x in _BULLET.split(sent)):
        if not p:
            continue
        if cur and len(cur) + 1 + len(p) > target:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        parts.append(cur)

    # 2. Co nadal za długie (np. wyciąg z tabeli, gdzie nie ma żadnych znaczników) — cięcie
    #    po SPACJI, nigdy w środku słowa: rozbity wyraz to dla embeddingu i BM25 czysty szum,
    #    a cytat przerwany w połowie przestaje występować w korpusie znak w znak.
    out: list[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, target)
            if cut <= 0:                  # jeden wyraz dłuższy niż target (patologia) — tnij twardo
                cut = target
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


# --- Pakowanie zdań w chunki z zakładką ---------------------------------------

def pack_chunks(
    sentences: list[str],
    target: int,
    overlap: int,
    min_chars: int,
    carry_max: int = config.CHUNK_CARRY_MAX_CHARS,
) -> list[str]:
    """
    Zachłannie dokłada zdania do chunka aż do ~target znaków, potem zaczyna nowy chunk,
    PRZENOSZĄC na jego początek ostatnie zdania (~overlap znaków). Overlap też jest
    liczony w całych zdaniach — spójnie z naszą strategią (nie tniemy w połowie zdania).

    `carry_max` = najdłuższe JEDNO zdanie, które wolno przenieść do zakładki. DLACZEGO ten
    parametr istnieje (druga warstwa dziury z etapu 14, przyczyna ujawniona pomiarem): pętla
    zakładki brała zdania od końca i zabierała ZAWSZE co najmniej jedno — bez względu na jego
    długość, bo warunek przerwania sprawdzał `keep` (czy już coś wzięliśmy). Gdy tym jednym
    było zdanie-moloch, lądowało w nowym chunku w całości, a potem doklejało się do niego
    następne zdanie: 2867 + 2854 = 5722 zn. Ten sam mechanizm dawał 2548 zn. w korpusie
    bazowym (zdanie 1850 + 698), więc to NIE jest usterka samych dokumentów listowych —
    to niezamierzone zachowanie zakładki, które na normalnej prozie się nie ujawnia,
    bo tam zdania mają ~190 zn. Domyślna wartość z configu, bo wołający nie powinien mieć
    możliwości przypadkowego przywrócenia patologii przez pominięcie argumentu.
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
                # Pierwsze (najdalsze wstecz) zdanie wolno wziąć ponad budżet zakładki —
                # tak działa zakładka na normalnej prozie, gdzie zdanie > overlap jest częste.
                # Ale nie WIELOKROTNIE ponad: powyżej `carry_max` rezygnujemy z zakładki
                # (chunk zaczyna się bez niej) zamiast duplikować moloch.
                if not keep and len(t) > carry_max:
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

def chunk_text(text: str, source: str) -> list[dict]:
    """Oczyszczony TEKST + etykieta źródła -> rekordy chunków z metadanymi.

    DLACZEGO osobno od `chunk_file`: dokument wgrany w UI (etap 2) nie istnieje na dysku, więc
    nie ma z czego czytać ścieżki — a musi przejść DOKŁADNIE tę samą logikę cięcia, co korpus,
    na którym zmierzone są liczby. Parametry (target/overlap/min) nadal z jednego miejsca
    (`config`), więc „chunking demo" nie może się rozjechać z „chunkingiem zmierzonym".
    """
    sentences = split_sentences(text)
    # Reguła awaryjna PO podziale na zdania, PRZED pakowaniem: dokument wypunktowany bez kropek
    # inaczej wpycha do chunka jedno „zdanie" wielkości kilku chunków (patrz `split_overlong`).
    sentences = [
        part
        for sent in sentences
        for part in split_overlong(sent, max_chars=config.CHUNK_MAX_SENTENCE_CHARS,
                                   target=config.CHUNK_TARGET_CHARS)
    ]
    chunks = pack_chunks(
        sentences,
        target=config.CHUNK_TARGET_CHARS,
        overlap=config.CHUNK_OVERLAP_CHARS,
        min_chars=config.CHUNK_MIN_CHARS,
        carry_max=config.CHUNK_CARRY_MAX_CHARS,
    )

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


def chunk_file(txt_path: Path) -> list[dict]:
    """Zamienia jeden oczyszczony .txt na listę rekordów-chunków z metadanymi.
    Etykieta źródła = nazwa pliku (na razie cytujemy na poziomie dokumentu)."""
    return chunk_text(txt_path.read_text(encoding="utf-8"), source=txt_path.stem)


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
