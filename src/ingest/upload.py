"""
src/ingest/upload.py
Etap 2 UI: przyjęcie PDF-u wgranego przez użytkownika — BAJTY -> rekordy chunków.

CO TU MIESZKA, A CO NIE (podział jest celowy):
- TUTAJ: polityka przyjęcia (limity, wykrywanie skanu, etykieta źródła) i spięcie
  „parsowanie -> czyszczenie -> chunking". Zero Qdranta, zero modeli, zero dysku.
- NIE tutaj: liczenie embeddingów i zapis do kolekcji (`src/index/build_index.py`) ani warstwa
  HTTP (`src/serve/api.py`). Dzięki temu CAŁA polityka jest testowalna bez GPU i bez usług —
  a to ona decyduje, co użytkownik zobaczy jako odmowę.

DLACZEGO POLITYKA JEST OSOBNYM ZESTAWEM MAŁYCH FUNKCJI (`check_*`), a nie jednym `if`-em
w środku parsowania: każdą z nich da się sprawdzić na liście stron, bez PDF-a. Testy odmów
w projekcie o wysokich wymaganiach bezpieczeństwa muszą się realnie uruchamiać, a test
wymagający pliku PDF i pymupdf w venvie dev nie uruchamiałby się (patrz `requirements-dev.txt`).

`UploadRejected` to ODMOWA (wina wejścia, komunikat dla człowieka, HTTP 400), a nie awaria
serwera (500). Rozróżnienie jest tu tak samo ważne, jak rozróżnienie trzech guardów w UI:
„twój plik jest skanem" i „serwer się wywalił" wymagają od użytkownika zupełnie różnych działań.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

_MB = 1024 * 1024

# Etykieta źródła: litery/cyfry (z polskimi znakami), spacja, kropka, myślnik, podkreślenie.
# Wszystko inne wypada — patrz `sanitize_source`.
_NIEDOZWOLONE = re.compile(r"[^\w .\-]", re.UNICODE)

SOURCE_FALLBACK = "dokument-wgrany"
SOURCE_MAX_LEN = 60


class UploadRejected(Exception):
    """Plik odrzucony z powodem DO POKAZANIA UŻYTKOWNIKOWI (nie błąd serwera)."""


def check_size(n_bytes: int) -> None:
    """Limit rozmiaru. Komunikat podaje liczby, bo „za duży plik" nie mówi, o ile za duży."""
    if n_bytes > config.UPLOAD_MAX_BYTES:
        raise UploadRejected(
            f"Plik ma {n_bytes / _MB:.1f} MB, a limit dla dokumentu demo to "
            f"{config.UPLOAD_MAX_BYTES / _MB:.0f} MB.")


def check_text_layer(pages: list[str]) -> float:
    """Zwraca średnią liczbę znaków na stronę; odrzuca dokument bez warstwy tekstowej.

    DLACZEGO ŚREDNIA, A NIE MINIMUM: strona tytułowa albo pusta strona rozdziałowa mają po
    kilka znaków i to jest normalne — odrzucanie po najsłabszej stronie wywalałoby poprawne
    raporty. DLACZEGO PRÓG NIE JEST ZEREM: skan zwykle daje kilkanaście śmieciowych znaków na
    stronę (numeracja, artefakty rozpoznane jako tekst), więc „zero znaków" nie wystąpi.
    """
    if not pages:
        raise UploadRejected("PDF nie ma ani jednej strony.")
    avg = sum(len(p.strip()) for p in pages) / len(pages)
    if avg < config.UPLOAD_MIN_CHARS_PER_PAGE:
        raise UploadRejected(
            f"PDF bez warstwy tekstowej (prawdopodobnie skan): średnio {avg:.0f} znaków na "
            f"stronę przy progu {config.UPLOAD_MIN_CHARS_PER_PAGE}. Taki dokument wymagałby "
            f"OCR, którego świadomie nie ma w tym pipeline.")
    return avg


def sanitize_source(filename: str) -> str:
    """Nazwa pliku -> etykieta źródła używana w cytowaniu (`[1] (źródło: X #3)`).

    DLACZEGO CZYŚCIMY, choć nic nie zapisujemy na dysk: ta etykieta trafia (a) do PROMPTU
    modelu i (b) do UI, gdzie cytaty wyłuskuje regex `/\\[(\\d+)\\]/`. Nazwa `raport[1].pdf`
    wstrzyknęłaby w kontekst fałszywy marker cytowania, a znaki nowej linii rozwaliłyby
    strukturę bloku `[n] (źródło: …)`. Obcinamy też katalogi (`../../etc/passwd.pdf`) — nie
    dlatego, że gdzieś je otwieramy, ale dlatego, że etykieta ma być nazwą dokumentu.
    """
    stem = Path(filename.strip()).name
    # Rozszerzenie zdejmujemy regexem, nie `Path.stem`: dla nazwy „.pdf" (sam rozszerzenie)
    # `Path` widzi plik ukryty bez sufiksu i zwraca „.pdf", czyli etykietę „pdf".
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    stem = _NIEDOZWOLONE.sub(" ", stem)
    stem = " ".join(stem.split())            # zbite/kontrolne białe znaki -> jedna spacja
    stem = stem.strip(" .-")[:SOURCE_MAX_LEN].strip()
    return stem or SOURCE_FALLBACK


def prepare_upload(data: bytes, filename: str) -> tuple[list[dict], dict]:
    """Bajty PDF-a -> (rekordy chunków, statystyki do UI). Plik NIE dotyka dysku.

    Kolejność kontroli jest kolejnością kosztów: rozmiar (darmowy) -> liczba stron (`page_count`)
    -> ekstrakcja -> warstwa tekstowa -> czyszczenie -> chunking. Nie ma sensu parsować 300 stron,
    żeby dowiedzieć się, że plik ma 60 MB.
    """
    check_size(len(data))

    # Ciężki import trzymamy w środku funkcji: `parse_pdf` importuje `fitz` na poziomie modułu,
    # a testy kontraktowe warstwy serwującej lecą w venvie bez pymupdf (i mają tak zostać).
    from src.ingest.chunk import chunk_text
    from src.ingest.parse_pdf import PdfTooLong, clean_pages, extract_pages_bytes

    try:
        pages = extract_pages_bytes(data, max_pages=config.UPLOAD_MAX_PAGES)
    except PdfTooLong as exc:
        # Za długi dokument to NIE to samo co uszkodzony: pierwszy da się skrócić, drugiego nie.
        raise UploadRejected(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Uszkodzony plik, .docx przemianowany na .pdf, PDF zaszyfrowany — z punktu widzenia
        # użytkownika to jedna rzecz: „tego pliku nie umiem przeczytać". Traceback zostaje
        # w logu serwera (patrz `api.py`), do UI idzie zdanie.
        raise UploadRejected(f"Nie udało się odczytać PDF-a: {exc}") from exc

    avg_chars = check_text_layer(pages)
    text = clean_pages(pages)
    source = sanitize_source(filename)
    records = chunk_text(text, source=source)
    if not records:
        raise UploadRejected("Po oczyszczeniu w dokumencie nie został żaden tekst do zaindeksowania.")

    stats = {
        "source": source,
        "pages": len(pages),
        "chars": len(text),
        "chunks": len(records),
        "chars_per_page": round(avg_chars),
    }
    return records, stats
