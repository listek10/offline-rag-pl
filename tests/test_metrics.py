"""
tests/test_metrics.py
Lekkie testy metryk (bez zależności; działa pod pytest i przez `python test_metrics.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval.metrics import hit_at_k, reciprocal_rank, mean


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
