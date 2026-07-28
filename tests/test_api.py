"""
tests/test_api.py
Kontrakt HTTP warstwy serwującej. Qdrant, Ollama i cała pętla RAG są PODMIENIONE — test
sprawdza KONTRAKT (kształt odpowiedzi, kolejność zdarzeń, kody błędów), a nie jakość modelu.

UWAGA na `TestClient(app)` BEZ menedżera kontekstu: dzięki temu `lifespan` się nie uruchamia,
więc rozgrzewka nie próbuje ładować BGE-M3. Test ma trwać milisekundy i działać bez GPU.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.serve import api as api_mod

client = TestClient(api_mod.app)


def test_health_zwraca_progi_i_modele_z_configu(monkeypatch):
    monkeypatch.setattr(api_mod, "_probe_qdrant",
                        lambda: {"ok": True, "collection": "kolekcja", "points": 1509})
    monkeypatch.setattr(api_mod, "_probe_ollama",
                        lambda: {"ok": True, "models": ["qwen2.5:7b"], "missing": []})

    body = client.get("/api/health").json()

    assert body["qdrant"]["points"] == 1509
    assert body["ollama"]["ok"] is True
    # Progi MUSZĄ pochodzić z configu — inaczej UI kłamałoby po zmianie progu.
    assert body["config"]["gate_min"] == config.RELEVANCE_GATE_MIN
    assert body["config"]["top_k"] == config.TOP_K
    assert body["config"]["judge"] == config.JUDGE_MODEL
    assert body["warm"] is False        # lifespan nie odpalony => nierozgrzane


def test_health_nie_wywala_sie_gdy_uslugi_leza(monkeypatch):
    def boom():
        raise ConnectionError("brak polaczenia")

    monkeypatch.setattr(api_mod, "_probe_qdrant", lambda: api_mod._safe_probe(boom, "qdrant"))
    monkeypatch.setattr(api_mod, "_probe_ollama", lambda: {"ok": False, "models": [], "missing": []})

    resp = client.get("/api/health")

    assert resp.status_code == 200          # health NIGDY nie zwraca 5xx — inaczej UI nie wie nic
    assert resp.json()["qdrant"]["ok"] is False
