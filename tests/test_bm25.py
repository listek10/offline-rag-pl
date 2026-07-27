"""
tests/test_bm25.py
Lekkie testy BM25 i RRF (bez zależności; działa pod pytest i przez `python test_bm25.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieve.bm25 import BM25, tokenize
from src.retrieve.hybrid import rrf_fuse


def test_tokenize_polskie_znaki():
    assert tokenize("Bezpieczeństwo NATO, art. 5!") == ["bezpieczeństwo", "nato", "art", "5"]


def test_bm25_trafia_w_slowo_kluczowe():
    docs = [
        {"id": "d0", "text": "Balticconnector to gazociąg fińsko-estoński."},
        {"id": "d1", "text": "Nordycka współpraca obronna NORDEFCO powstała w 2009 roku."},
        {"id": "d2", "text": "Arktyka kryje zasoby ropy i gazu."},
    ]
    bm = BM25(docs, k1=1.5, b=0.75)
    top = bm.search("Balticconnector", top_k=1)
    assert top[0][0] == "d0"                      # trafiamy w chunk z dokładnym termem
    assert bm.search("NORDEFCO", top_k=1)[0][0] == "d1"


def test_bm25_term_spoza_korpusu():
    docs = [{"id": "d0", "text": "Arktyka i Bałtyk."}]
    bm = BM25(docs, k1=1.5, b=0.75)
    assert bm.search("kangur", top_k=1)[0][1] == 0.0   # term spoza korpusu -> score 0


def test_rrf_faworyzuje_zgodnosc_rankingow():
    # 'x' jest wysoko w OBU rankingach -> powinien wygrać po fuzji.
    dense = ["x", "a", "b"]
    bm25 = ["x", "c", "d"]
    scores = rrf_fuse([dense, bm25], k=60)
    best = max(scores, key=lambda c: scores[c])
    assert best == "x"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"\n{len(tests)} testów przeszło.")
