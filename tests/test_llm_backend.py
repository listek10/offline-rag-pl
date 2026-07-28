"""
tests/test_llm_backend.py
Etap 7 / checkpoint #9 cz. 2: dyspozytor backendów LLM (Ollama <-> vLLM).

Testujemy ROUTING i KONTRAKT HTTP, nie jakość modelu: `urlopen` jest podmieniony atrapą,
więc test trwa milisekundy, nie potrzebuje GPU ani żadnego serwera i przechodzi offline.
DLACZEGO to warto testować: przełącznik backendu dotyka ścieżki produkcyjnej (`answer()`),
a błąd w nim objawia się dopiero pod obciążeniem benchmarku — wtedy trudno go odróżnić od
problemu wydajnościowego.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.generate import llm as llm_mod


class _FakeResponse(io.BytesIO):
    """Minimalna atrapa odpowiedzi `urlopen` (kontekstowa, z .read())."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _fake_urlopen(captured: dict, body: dict):
    """Podmiana `urllib.request.urlopen`: zapisuje wysłane żądanie i zwraca gotowy JSON."""

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    return _open


# --- Domyślny backend: bezpieczeństwo zmiany ---------------------------------------------

def test_domyslnym_backendem_jest_ollama():
    """Zmiana ma być ADDYTYWNA: bez ustawienia env cały projekt działa jak dotąd."""
    assert config.LLM_BACKEND == "ollama"


# --- Routing ----------------------------------------------------------------------------

@pytest.fixture
def stub_backends(monkeypatch):
    """Podmienia OBA backendy na atrapy — dyspozytor ma je rozwiązywać przez `globals()`."""
    monkeypatch.setattr(llm_mod, "_generate_ollama",
                        lambda *a, **kw: {"text": "ODP_OLLAMA", "completion_tokens": 1})
    monkeypatch.setattr(llm_mod, "_generate_vllm",
                        lambda *a, **kw: {"text": "ODP_VLLM", "completion_tokens": 2})


def test_call_llm_kieruje_do_ollamy(stub_backends):
    assert llm_mod.call_llm("pytanie", backend="ollama") == "ODP_OLLAMA"


def test_call_llm_kieruje_do_vllm(stub_backends):
    assert llm_mod.call_llm("pytanie", backend="vllm") == "ODP_VLLM"


def test_call_llm_uzywa_backendu_z_configu(stub_backends, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    assert llm_mod.call_llm("pytanie") == "ODP_VLLM"


def test_generate_zwraca_liczbe_tokenow_wyjscia(stub_backends):
    """Benchmark liczy tokeny/s — bez tej liczby mierzyłby sekundy na odpowiedź, a odpowiedzi
    obu silników mają różną długość (inna kwantyzacja = inny tekst)."""
    assert llm_mod.generate("pytanie", backend="vllm") == {"text": "ODP_VLLM",
                                                           "completion_tokens": 2}


def test_call_ollama_ignoruje_backend_z_configu(stub_backends, monkeypatch):
    """Sędzia guarda #3 musi trafiać do Ollamy nawet gdy generator liczy na vLLM."""
    monkeypatch.setattr(config, "LLM_BACKEND", "vllm")
    assert llm_mod.call_ollama("pytanie") == "ODP_OLLAMA"


def test_nieznany_backend_konczy_sie_jasnym_bledem():
    with pytest.raises(ValueError) as e:
        llm_mod.call_llm("pytanie", backend="openai")
    # Literówka w configu ma dać komunikat, który mówi CO wolno, a nie ciche wysłanie w próżnię.
    assert "openai" in str(e.value) and "ollama" in str(e.value) and "vllm" in str(e.value)


def test_call_llm_przekazuje_system_model_i_max_tokens(monkeypatch):
    seen = {}

    def _spy(prompt, system="", model=None, max_tokens=None):
        seen.update(prompt=prompt, system=system, model=model, max_tokens=max_tokens)
        return {"text": "ok", "completion_tokens": 0}

    monkeypatch.setattr(llm_mod, "_generate_vllm", _spy)
    llm_mod.call_llm("P", system="S", model="M", max_tokens=7, backend="vllm")
    assert seen == {"prompt": "P", "system": "S", "model": "M", "max_tokens": 7}


# --- Kontrakt HTTP vLLM (serwer zgodny z OpenAI) ----------------------------------------

def test_call_vllm_buduje_poprawne_zadanie_i_parsuje_odpowiedz(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen",
        _fake_urlopen(captured, {"choices": [{"message": {"content": "  odpowiedź  "}}]}),
    )

    out = llm_mod.call_vllm("PYTANIE", system="SYSTEM", max_tokens=64)

    assert out == "odpowiedź"                       # .strip() jak w ścieżce Ollamy

    assert captured["url"] == f"{config.VLLM_URL}/chat/completions"
    p = captured["payload"]
    assert p["model"] == config.VLLM_MODEL
    assert p["messages"] == [{"role": "system", "content": "SYSTEM"},
                             {"role": "user", "content": "PYTANIE"}]
    assert p["temperature"] == config.LLM_TEMPERATURE   # temp 0 obowiązuje na OBU backendach
    assert p["max_tokens"] == 64
    assert p["stream"] is False


def test_liczba_tokenow_pochodzi_z_serwera_nie_z_wlasnego_liczenia(monkeypatch):
    """Ollama nazywa to `eval_count`, vLLM `usage.completion_tokens` — obie ścieżki muszą
    trafić do tego samego klucza, bo benchmark porównuje je wprost."""
    captured: dict = {}
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", _fake_urlopen(
        captured, {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 128}}))
    assert llm_mod.generate("P", backend="vllm")["completion_tokens"] == 128

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", _fake_urlopen(
        captured, {"response": "x", "eval_count": 99}))
    assert llm_mod.generate("P", backend="ollama")["completion_tokens"] == 99


def test_brak_pola_z_tokenami_nie_wywala_generacji(monkeypatch):
    """Gdy serwer nie poda liczby tokenów, odpowiedź ma nadal wrócić (None), a nie KeyError —
    benchmark sam zdecyduje, że nie da się policzyć tokenów/s."""
    captured: dict = {}
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen",
                        _fake_urlopen(captured, {"response": "x"}))
    assert llm_mod.generate("P", backend="ollama") == {"text": "x", "completion_tokens": None}


def test_call_vllm_bez_systemu_nie_wysyla_pustej_wiadomosci(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen",
        _fake_urlopen(captured, {"choices": [{"message": {"content": "x"}}]}),
    )
    llm_mod.call_vllm("PYTANIE")
    assert captured["payload"]["messages"] == [{"role": "user", "content": "PYTANIE"}]


# --- Kontrakt HTTP Ollamy (regresja: przeniesienie funkcji nie zmieniło żądania) ---------

def test_call_ollama_wysyla_to_samo_co_wczesniej(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen",
        _fake_urlopen(captured, {"response": "  odp  "}),
    )

    out = llm_mod.call_ollama("PYTANIE", system="SYSTEM", model="model:x", max_tokens=64)

    assert out == "odp"
    assert captured["url"] == f"{config.OLLAMA_URL}/api/generate"
    p = captured["payload"]
    assert p["model"] == "model:x"
    assert p["system"] == "SYSTEM"
    assert p["prompt"] == "PYTANIE"
    assert p["stream"] is False
    assert p["options"]["temperature"] == config.LLM_TEMPERATURE
    assert p["options"]["num_ctx"] == config.LLM_NUM_CTX
    # W Ollamie limit tokenów wyjścia nazywa się num_predict — musi zadziałać, inaczej
    # benchmark porównywałby RÓŻNE ilości pracy na obu backendach.
    assert p["options"]["num_predict"] == 64


def test_call_ollama_bez_limitu_nie_ustawia_num_predict(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        llm_mod.urllib.request, "urlopen",
        _fake_urlopen(captured, {"response": "odp"}),
    )
    llm_mod.call_ollama("PYTANIE")
    assert "num_predict" not in captured["payload"]["options"]


# --- Stara ścieżka importu musi nadal działać -------------------------------------------

def test_answer_wciaz_eksportuje_call_ollama():
    """`verify.py` i `eval_answers.py` importują `call_ollama` z `answer.py` — nie wolno
    im tego zabrać przy przenoszeniu transportu do `llm.py`."""
    from src.generate.answer import call_ollama
    assert call_ollama is llm_mod.call_ollama
