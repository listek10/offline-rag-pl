"""
tests/test_metrics.py
Lekkie testy metryk (bez zależności; działa pod pytest i przez `python test_metrics.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.metrics import hit_at_k, reciprocal_rank, mean, relevant_by_evidence


def test_hit_at_k():
    ret = ["a", "b", "c", "d"]
    assert hit_at_k(ret, ["c"], k=3) == 1.0        # c jest w top-3
    assert hit_at_k(ret, ["c"], k=2) == 0.0        # c poza top-2
    assert hit_at_k(ret, ["x", "b"], k=2) == 1.0   # którykolwiek trafny wystarczy
    assert hit_at_k(ret, ["z"], k=4) == 0.0


def test_reciprocal_rank():
    ret = ["a", "b", "c"]
    assert reciprocal_rank(ret, ["a"]) == 1.0          # 1. pozycja
    assert reciprocal_rank(ret, ["b"]) == 0.5          # 2. pozycja
    assert abs(reciprocal_rank(ret, ["c"]) - 1/3) < 1e-9
    assert reciprocal_rank(ret, ["z"]) == 0.0          # brak
    assert reciprocal_rank(ret, ["c", "b"]) == 0.5     # bierze PIERWSZy trafny (b na poz. 2)


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")


# --- Prawda przez cytat-dowód (etap 8: A/B chunkingu) ------------------------------------
# Ta definicja zastępuje `chunk_id` przy porównywaniu strategii chunkingu: numeracja chunków
# przesuwa się po zmianie rozmiaru, więc identyfikatory przestają cokolwiek wskazywać.

def _ch(cid: str, text: str) -> dict:
    return {"id": cid, "text": text}


def test_evidence_znajduje_wszystkie_chunki_z_cytatem():
    """WSZYSTKIE, nie pierwszy: przy zakładce ten sam fragment jest w dwóch kolejnych chunkach
    i oba realnie zawierają odpowiedź."""
    chunks = [_ch("a__0", "Wstęp. Dania ma 1,3% PKB na obronę."),
              _ch("a__1", "Dania ma 1,3% PKB na obronę. Dalszy tekst."),
              _ch("a__2", "Nic o wydatkach.")]
    assert relevant_by_evidence(chunks, "Dania ma 1,3% PKB") == ["a__0", "a__1"]


def test_evidence_ignoruje_roznice_bialych_znakow():
    """Parser PDF-a zostawia artefakty typu „fińsko- -estońskiego" i złamania wierszy —
    dopasowanie ma być odporne na odstęp, nie na treść."""
    chunks = [_ch("a__0", "gazociągu\n Balticconnector   (październik 2023) uszkodzono")]
    assert relevant_by_evidence(chunks, "gazociągu Balticconnector (październik 2023)") == ["a__0"]


def test_evidence_bez_trafienia_daje_pusta_liste():
    """Pusta lista = pytanie NIEMOŻLIWE do trafienia przy tym podziale (np. cytat rozcięty
    na granicy chunków). Harness musi to zgłosić, a nie policzyć jako zwykłe pudło."""
    chunks = [_ch("a__0", "Dania ma"), _ch("a__1", "1,3% PKB na obronę.")]
    assert relevant_by_evidence(chunks, "Dania ma 1,3% PKB") == []
