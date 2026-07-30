"""
tests/test_ab_ogon.py
Testy harnessu z etapu 14 (`src/eval/ab_ogon.py`) — mierzy ogon rozkładu długości chunków.

DLACZEGO harness ma testy: on produkuje LICZBY, które trafiają do README i na rozmowę.
Harness, który po cichu zmierzy coś innego niż deklaruje, jest gorszy niż brak pomiaru.
Kluczowy jest tu test odtworzenia patologii: gdyby wariant „przed" nie umiał jej pokazać,
dowód „przed/po" byłby porównywaniem naprawy z naprawą.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.eval.ab_ogon import BEZ_LIMITU, potnij, rozklad

# Dokument listowy w miniaturze: wypunktowania bez kropek, więc `split_sentences`
# zwraca jedno gigantyczne „zdanie" — dokładnie ten wzorzec, który psuł chunki.
_LISTA = " ".join(f"• wzmocnienie zdolności obronnych w wymiarze {i} oraz odporności cywilnej,"
                  for i in range(120))


def test_wariant_przed_odtwarza_patologie():
    # „Przed" MUSI pokazać ogon — inaczej dowód przed/po niczego nie dowodzi.
    przed = potnij(_LISTA, BEZ_LIMITU, BEZ_LIMITU)
    assert max(len(c) for c in przed) > 2 * config.CHUNK_TARGET_CHARS, \
        "wariant PRZED nie odtworzył patologii — dowód przed/po byłby fikcją"


def test_wariant_po_domyka_ogon():
    po = potnij(_LISTA, config.CHUNK_MAX_SENTENCE_CHARS, config.CHUNK_CARRY_MAX_CHARS)
    assert max(len(c) for c in po) <= config.CHUNK_MAX_SENTENCE_CHARS


def test_sam_limit_zdania_nie_wystarcza():
    # To jest pomiar, na którym stoi decyzja o DWUCZĘŚCIOWEJ naprawie (i wpis w notatniku):
    # bez kapu zakładki moloch nadal wchodzi do następnego chunka.
    tylko_limit = potnij(_LISTA, config.CHUNK_MAX_SENTENCE_CHARS, BEZ_LIMITU)
    pelna = potnij(_LISTA, config.CHUNK_MAX_SENTENCE_CHARS, config.CHUNK_CARRY_MAX_CHARS)
    assert max(len(c) for c in tylko_limit) > max(len(c) for c in pelna)


def test_rozklad_liczy_ogon_a_nie_srednia():
    # Rozkład z jednym molochem: średnia prawie się nie rusza, max i udziały MUSZĄ.
    chunks = ["x" * 1000] * 99 + ["y" * 5000]
    w = rozklad(chunks)
    assert w["n_chunks"] == 100
    assert w["max"] == 5000
    assert w["p50"] == 1000
    assert w["pct_ponad_2000"] == 1.0        # 1 chunk ze 100
    assert 1000 <= w["avg"] <= 1050          # średnia kłamie — o to w tym pomiarze chodzi
