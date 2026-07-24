"""
src/ingest/parse_pdf.py
Etap 1 projektu RAG: parsowanie i czyszczenie PDF-ów (raporty OSW/PISM).

DLACZEGO ten krok jest ważny (checkpoint #1):
PDF to format do WYŚWIETLANIA, nie do czytania maszynowego. Tekst bywa poukładany
w kolumnach, poprzeplatany nagłówkami/stopkami, a wyrazy przenoszone są myślnikiem
na końcu wiersza. Jeśli wrzucimy surowy tekst do chunkingu, śmieci (numery stron,
stopki) trafią do embeddingów i zaszumią wyszukiwanie. Dlatego najpierw czyścimy.

Zależności:
    pip install pymupdf        # import jako "fitz" — główny parser (szybki, dobra jakość)
    # opcjonalnie do trudnych układów / tabel:
    # pip install pdfplumber

Uruchomienie:
    python parse_pdf.py data/raw/raport.pdf data/processed/
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF


# --- Krok 1: wyciągnięcie surowego tekstu strona po stronie ------------------

def extract_pages(pdf_path: Path) -> list[str]:
    """
    Zwraca listę stron jako surowy tekst.

    DLACZEGO "blocks", a nie zwykłe get_text():
    PyMuPDF potrafi zwrócić tekst w blokach z współrzędnymi (x, y). Sortując bloki
    najpierw po pozycji poziomej (lewa/prawa kolumna), a potem pionowej, odtwarzamy
    poprawną KOLEJNOŚĆ CZYTANIA w dokumentach dwukolumnowych. Zwykłe get_text("text")
    często miesza kolumny, sklejając zdanie z lewej z połową zdania z prawej.
    """
    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, "tekst", block_no, block_type)
        page_width = page.rect.width

        # Prosty podział na kolumny: blok, którego środek jest w lewej połowie strony,
        # trafia do lewej kolumny; reszta do prawej. Dla dokumentów jednokolumnowych
        # wszystko wpada do jednej kolumny i sortuje się po prostu po y.
        left, right = [], []
        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            if not text.strip():
                continue
            center_x = (x0 + x1) / 2
            (left if center_x < page_width / 2 else right).append((y0, text))

        left.sort(key=lambda t: t[0])   # w obrębie kolumny czytamy z góry na dół
        right.sort(key=lambda t: t[0])
        ordered = [t[1] for t in left] + [t[1] for t in right]
        pages.append("\n".join(ordered))
    doc.close()
    return pages


# --- Krok 2: wykrycie i usunięcie powtarzalnych nagłówków/stopek -------------

def find_running_lines(pages: list[str], min_ratio: float = 0.5) -> set[str]:
    """
    Znajduje wiersze powtarzające się na wielu stronach (nagłówki, stopki, tytuł raportu).

    DLACZEGO: stopka typu "Raport OSW | 2025" pojawia się na KAŻDEJ stronie. Jako chunk
    to czysty szum, który psuje embeddingi. Traktujemy jako "running line" każdy krótki
    wiersz obecny na >= min_ratio stron.
    """
    line_counter: Counter[str] = Counter()
    for page in pages:
        # unikalne wiersze na stronie (żeby jeden powtórzony w treści nie zafałszował)
        unique_lines = {ln.strip() for ln in page.splitlines() if ln.strip()}
        line_counter.update(unique_lines)

    threshold = max(2, int(len(pages) * min_ratio))
    # tylko krótkie wiersze — długie zdania to raczej treść, nie stopka
    return {ln for ln, c in line_counter.items() if c >= threshold and len(ln) < 80}


# --- Krok 3: czyszczenie tekstu ----------------------------------------------

def clean_text(text: str, running: set[str]) -> str:
    """
    Sprząta: usuwa stopki/nagłówki, łączy przeniesione wyrazy, normalizuje spacje,
    usuwa samotne numery stron.
    """
    lines = []
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped in running:            # stopka / nagłówek
            continue
        if re.fullmatch(r"\d{1,4}", stripped):  # samotny numer strony
            continue
        lines.append(stripped)

    joined = "\n".join(lines)

    # Miękki dywiz (U+00AD "soft hyphen") — NIEWIDZIALNY znak, którym PDF-y OSW
    # przenoszą wyrazy na końcu wiersza. Wygląda jak zwykły myślnik, ale to inny
    # znak, więc regex na "-" go nie łapał. Efekt bez tego fixa: "stano­ wi" zamiast
    # "stanowi" -> dla embeddingu (i dla BM25) dwa śmieciowe fragmenty zamiast słowa.
    # DLATEGO robimy to PRZED zamianą złamań wiersza na spacje: najpierw sklejamy
    # przeniesienie (dywiz + newline znikają), potem usuwamy ewentualne dywizy w środku
    # linii (są zbędne). Kolejność jest kluczowa — inaczej newline zamieniłby się w spację.
    joined = re.sub(r"­[ \t]*\n[ \t]*", "", joined)   # przeniesienie: sklej wyraz
    joined = joined.replace("­", "")                   # pozostałe miękkie dywizy: usuń

    # Łączenie wyrazów przeniesionych ZWYKŁYM myślnikiem: "bezpie-\nczeństwo" -> "bezpieczeństwo".
    # DLACZEGO ważne dla polskiego: fleksja + długie wyrazy = dużo przenoszeń. Rozbity
    # wyraz to dla embeddingu dwa bezsensowne tokeny zamiast jednego znaczącego.
    joined = re.sub(r"(\w+)-\n(\w+)", r"\1\2", joined)

    # Żywa pagina OSW ("RAPORT OSW 4/2026") wtapia się w ŚRODEK akapitów przy scalaniu
    # bloków, więc wykrywacz powtarzalnych LINII (find_running_lines) jej nie łapie —
    # nie jest samodzielnym wierszem. Usuwamy ją dopasowaniem wzorca w całym tekście.
    # Świadomy trade-off: to reguła "pod OSW" (mało ogólna), ale tania i skuteczna;
    # ogólniejsze wykrywanie powtórek WEWNĄTRZ tekstu byłoby złożone i groziłoby
    # wycięciem treści. Warianty odstępu ("4/2026" i "4 /2026") obejmuje \s*.
    # Pagina potrafi wciąć się w ŚRODEK słowa ("kie[PAGINA]runek"), gdy wyraz pęka na
    # granicy strony — dlatego łapiemy fragmenty słowa (\w*) po OBU stronach i sklejamy
    # je z powrotem (\1\2). [ \t]* (nie \s*) po numerze, by nie zjeść granicy akapitu.
    joined = re.sub(r"(\w*)RAPORT OSW\s+\d+\s*/\s*\d+[ \t]*(\w*)", r"\1\2", joined)

    # Pojedyncze złamania wiersza wewnątrz akapitu zamieniamy na spację,
    # ale podwójne (granica akapitu) zostawiamy.
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", joined)
    joined = re.sub(r"[ \t]{2,}", " ", joined)      # zbite spacje
    joined = re.sub(r"\n{3,}", "\n\n", joined)       # nadmiar pustych linii

    return joined.strip()


# --- Spięcie: PDF -> czysty .txt ---------------------------------------------

def parse_pdf(pdf_path: Path, out_dir: Path) -> Path:
    pages = extract_pages(pdf_path)
    running = find_running_lines(pages)
    cleaned_pages = [clean_text(p, running) for p in pages]
    full_text = "\n\n".join(p for p in cleaned_pages if p)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.txt"
    out_path.write_text(full_text, encoding="utf-8")

    # Szybki sygnał jakości do wglądu w konsoli.
    print(f"[OK] {pdf_path.name}: {len(pages)} stron, "
          f"{len(running)} wykrytych stopek/nagłówków, "
          f"{len(full_text):,} znaków -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python parse_pdf.py <plik.pdf> <katalog_wyjściowy>")
        sys.exit(1)
    parse_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
