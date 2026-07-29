"""
tests/test_upload.py
Etap 2 UI: przyjęcie wgranego PDF-u — POLITYKA (limity, wykrywanie skanu, etykieta źródła)
oraz ścieżka „bajty -> rekordy chunków".

PODZIAŁ TESTÓW WEDŁUG ZALEŻNOŚCI (świadomy):
- Testy POLITYKI działają na listach stron i czystym tekście — zero `fitz`, więc lecą w venvie
  dev (61 MB, bez torcha). To tam mieszkają wszystkie decyzje typu „odrzucamy/przyjmujemy".
- Testy PARSOWANIA wymagają `pymupdf` i budują PRAWDZIWY PDF w pamięci (`fitz` umie tworzyć
  dokumenty). Bez pymupdf pomijają się — jak test `app.js` bez node'a. Dlaczego warto mimo
  pominięcia w dev: atrapa PDF-a nie sprawdziłaby niczego ciekawego, bo cała trudność jest
  właśnie w tym, co PyMuPDF zwraca z prawdziwego pliku.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.ingest import upload as upload_mod
from src.ingest.upload import UploadRejected

# Skip zamiast atrapy: `fitz` nie ma w venvie dev (celowo — patrz requirements-dev.txt).
wymaga_pymupdf = pytest.mark.skipif(
    importlib.util.find_spec("fitz") is None,
    reason="pymupdf niedostępny — ścieżkę PDF sprawdza env projektu (conda offline-rag-pl)")


# --- polityka: limity ---------------------------------------------------------------------

def test_za_duzy_plik_jest_odrzucony_z_komunikatem_po_polsku():
    """Limit nie jest zabezpieczeniem (serwer stoi na 127.0.0.1) — chroni DEMO przed
    zamrożeniem na minutę przy 2000-stronicowym tomie. Dlatego komunikat mówi, co się stało."""
    with pytest.raises(UploadRejected) as exc:
        upload_mod.check_size(config.UPLOAD_MAX_BYTES + 1)

    assert "MB" in str(exc.value)


def test_plik_w_limicie_przechodzi():
    upload_mod.check_size(config.UPLOAD_MAX_BYTES)


# --- polityka: wykrywanie skanu ----------------------------------------------------------

def test_pdf_bez_warstwy_tekstowej_jest_odrzucony_jako_skan():
    """Skan to obraz: PyMuPDF zwraca z niego kilkanaście śmieciowych znaków na stronę
    (numeracja, artefakty), a nie zero — dlatego próg NIE może być zerem."""
    strony = ["12", "", "3 |", "  "]

    with pytest.raises(UploadRejected) as exc:
        upload_mod.check_text_layer(strony)

    assert "OCR" in str(exc.value)          # mówimy, CZEGO brakuje w pipeline


def test_dokument_z_tekstem_przechodzi_i_zwraca_srednia_znakow():
    strony = ["x" * 500, "y" * 700]

    assert upload_mod.check_text_layer(strony) == 600.0


def test_prog_skanu_jest_liczony_na_srednia_nie_na_pojedyncza_strone():
    """Strona tytułowa ma kilka znaków i to normalne. Odrzucanie po NAJSŁABSZEJ stronie
    wywalałoby poprawne dokumenty — mierzymy średnią po całości."""
    strony = ["OSW", "z" * 4000]

    assert upload_mod.check_text_layer(strony) > config.UPLOAD_MIN_CHARS_PER_PAGE


def test_pusty_dokument_jest_odrzucony():
    with pytest.raises(UploadRejected):
        upload_mod.check_text_layer([])


# --- polityka: etykieta źródła -----------------------------------------------------------

def test_etykieta_zrodla_powstaje_z_nazwy_pliku():
    assert upload_mod.sanitize_source("Raport-OSW 2026.pdf") == "Raport-OSW 2026"


def test_etykieta_zrodla_nie_moze_zawierac_nawiasow_kwadratowych():
    """Etykieta źródła jedzie do promptu jako `[1] (źródło: X #3)` i do UI, gdzie cytaty
    wyłuskuje regex /\\[(\\d+)\\]/. Nazwa pliku typu `raport[1].pdf` wstrzyknęłaby fałszywy
    marker cytowania — czyli podpięłaby klikalny cytat pod nieistniejące źródło."""
    assert "[" not in upload_mod.sanitize_source("raport[1].pdf")
    assert "]" not in upload_mod.sanitize_source("raport[1].pdf")


def test_etykieta_zrodla_odrzuca_sciezki_i_znaki_kontrolne():
    """Nazwa pliku pochodzi od użytkownika. Nie zapisujemy jej na dysk, ale trafia do promptu,
    więc obcinamy wszystko, co nie jest treścią: katalogi, nowe linie, sterowanie."""
    assert upload_mod.sanitize_source("../../etc/passwd.pdf") == "passwd"
    assert "\n" not in upload_mod.sanitize_source("raport\nz\ndrugiej\nlinii.pdf")


def test_etykieta_zrodla_ma_awaryjna_wartosc():
    assert upload_mod.sanitize_source(".pdf") == "dokument-wgrany"
    assert upload_mod.sanitize_source("") == "dokument-wgrany"


def test_etykieta_zrodla_jest_skracana():
    assert len(upload_mod.sanitize_source("a" * 500 + ".pdf")) <= 60


# --- tekst -> rekordy chunków (bez PDF-a, bez dysku) -------------------------------------

def test_rekordy_z_tekstu_maja_id_stabilne_per_dokument():
    """`chunk.py` nadaje id jako `{source}__{i}` — dzięki temu id wgranego dokumentu nie
    przesuwają id chunków korpusu bazowego (gold set się nie rozjeżdża)."""
    from src.ingest.chunk import chunk_text

    zdania = " ".join(f"Zdanie numer {i} o bezpieczeństwie w regionie nordyckim." for i in range(60))
    rekordy = chunk_text(zdania, source="obcy-raport")

    assert len(rekordy) > 1
    assert [r["id"] for r in rekordy] == [f"obcy-raport__{i}" for i in range(len(rekordy))]
    assert all(r["source"] == "obcy-raport" for r in rekordy)
    assert all(r["chunk_index"] == i for i, r in enumerate(rekordy))


def test_chunk_file_uzywa_tej_samej_funkcji_co_upload(tmp_path):
    """Ingest z pliku i ingest z pamięci MUSZĄ przechodzić tę samą logikę cięcia — inaczej
    dokument demo byłby chunkowany inaczej niż korpus, na którym zmierzone są liczby."""
    from src.ingest.chunk import chunk_file, chunk_text

    tekst = "Pierwsze zdanie o Arktyce. Drugie zdanie o Bałtyku."
    plik = tmp_path / "raport-x.txt"
    plik.write_text(tekst, encoding="utf-8")

    assert chunk_file(plik) == chunk_text(tekst, source="raport-x")


# --- prawdziwy PDF w pamięci (wymaga pymupdf) --------------------------------------------

def _pdf_bytes(strony: list[str]) -> bytes:
    """Buduje PRAWDZIWY PDF w pamięci — bez pliku na dysku, tak jak w produkcyjnej ścieżce.

    UWAGA na treść stron w testach: domyślny Helvetica w PyMuPDF nie ma polskich znaków i
    zapisuje je jako „?". To ograniczenie GENERATORA PDF-ów w teście, nie parsera — dlatego
    strony budujemy z ASCII. Polską fleksję i miękkie dywizy sprawdzają testy chunkingu
    i prawdziwy korpus, nie ten plik.
    """
    import fitz

    doc = fitz.open()
    for tekst in strony:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), tekst, fontsize=11)
    dane = doc.tobytes()
    doc.close()
    return dane


@wymaga_pymupdf
def test_extract_pages_bytes_czyta_pdf_ze_strumienia():
    """`fitz.open(stream=...)` to sedno wymogu „plik nie dotyka dysku": PyMuPDF dostaje bajty,
    nie ścieżkę, więc na dysku nie powstaje ani `data/raw/`, ani nic w `/tmp`."""
    from src.ingest.parse_pdf import extract_pages_bytes

    strony = extract_pages_bytes(_pdf_bytes(["Arktyczny kwartet NATO.", "Druga strona raportu."]))

    assert len(strony) == 2
    assert "Arktyczny kwartet" in strony[0]


@wymaga_pymupdf
def test_extract_pages_bytes_odrzuca_dokument_ponad_limit_stron():
    """Limit sprawdzamy PRZED ekstrakcją (z `doc.page_count`), bo sens limitu jest w tym, żeby
    nie płacić czasu za 2000 stron — sprawdzenie po ekstrakcji byłoby limitem na papierze."""
    from src.ingest.parse_pdf import extract_pages_bytes

    with pytest.raises(ValueError) as exc:
        extract_pages_bytes(_pdf_bytes(["a", "b", "c"]), max_pages=2)

    assert "stron" in str(exc.value)


@wymaga_pymupdf
def test_pelna_sciezka_bajty_do_rekordow():
    """Kontrakt, na którym stoi warstwa serwująca: bajty PDF-a -> rekordy chunków + statystyki."""
    tekst = " ".join(f"Zdanie {i} o wspolpracy nordyckiej i Baltyku." for i in range(40))
    rekordy, stat = upload_mod.prepare_upload(_pdf_bytes([tekst, tekst]), "Obcy raport.pdf")

    assert stat["source"] == "Obcy raport"
    assert stat["pages"] == 2
    assert stat["chunks"] == len(rekordy) >= 1
    assert all(r["source"] == "Obcy raport" for r in rekordy)
    assert "wspolpracy nordyckiej" in rekordy[0]["text"]
    assert stat["chars"] > 0 and stat["chars_per_page"] > config.UPLOAD_MIN_CHARS_PER_PAGE


@wymaga_pymupdf
def test_zbyt_dlugi_dokument_mowi_o_limicie_a_nie_o_bledzie_odczytu(monkeypatch):
    """Sprawdzone na żywo na 518-stronicowym raporcie: bez tego rozróżnienia komunikat brzmiał
    „Nie udało się odczytać PDF-a: PDF ma 518 stron…" — czyli sklejał DWA różne powody. To ważne,
    bo prowadzą do różnych działań: „plik jest uszkodzony" (nic nie zrobisz) vs „skróć dokument"."""
    monkeypatch.setattr(config, "UPLOAD_MAX_PAGES", 2)

    with pytest.raises(UploadRejected) as exc:
        upload_mod.prepare_upload(_pdf_bytes(["a" * 300, "b" * 300, "c" * 300]), "dlugi.pdf")

    assert "3 stron" in str(exc.value) and "limit" in str(exc.value)
    assert "odczytać" not in str(exc.value)


@wymaga_pymupdf
def test_skan_jest_odrzucony_na_prawdziwym_pdf_ie():
    """PDF z samą numeracją stron = to, co zwraca skan bez warstwy tekstowej."""
    with pytest.raises(UploadRejected):
        upload_mod.prepare_upload(_pdf_bytes(["1", "2", "3"]), "skan.pdf")


@wymaga_pymupdf
def test_uszkodzony_plik_daje_odrzucenie_a_nie_traceback():
    """Użytkownik potrafi wrzucić .docx przemianowany na .pdf. To NIE jest awaria serwera,
    więc nie może lecieć jako 500 z tracebackiem."""
    with pytest.raises(UploadRejected):
        upload_mod.prepare_upload(b"to nie jest PDF", "podstep.pdf")
