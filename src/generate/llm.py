"""
src/generate/llm.py
Etap 7 / checkpoint #9 cz. 2: cienka warstwa serwowania LLM — jeden dyspozytor, dwa backendy.

    call_llm(...)  ->  Ollama (/api/generate)   |   vLLM (/v1/chat/completions, API OpenAI-owe)

DLACZEGO taka warstwa, a nie przepisanie pętli RAG:
- Backend to JEDNA decyzja infrastrukturalna, więc ma być JEDNĄ zmienną (`config.LLM_BACKEND`),
  nadpisywalną środowiskiem. Reszta projektu (`answer()`, guard #3, harness ewaluacyjny) nie
  wie, który silnik liczy — dzięki temu demo „przed/po" zmienia dokładnie jeden parametr.
- Transport piszemy sami na `urllib` (stdlib): nie dodajemy biblioteki `openai` ani klienta
  Ollamy. Mniej zależności = mniej rzeczy do audytu w środowisku o wysokich wymaganiach
  bezpieczeństwa (zasada z CLAUDE.md), a protokół to zwykły POST z JSON-em.

DWA POZIOMY API (celowo):
- `generate(...)` -> dict z tekstem I LICZBĄ TOKENÓW WYJŚCIA. Benchmark bez tej liczby mierzyłby
  „sekundy na odpowiedź", a odpowiedzi obu silników mają różną długość, więc byłoby to
  porównanie różnych ilości pracy. Liczbę bierzemy z serwera (Ollama: `eval_count`,
  vLLM: `usage.completion_tokens`), a nie z własnego liczenia słów.
- `call_llm(...)` -> sam tekst. Tego używa produkcyjna pętla RAG, której tokeny nie interesują.

CZYM SIĘ RÓŻNIĄ SILNIKI (sedno checkpointu #9):
- Dekodowanie LLM jest ograniczone PRZEPUSTOWOŚCIĄ PAMIĘCI: żeby wypluć jeden token, GPU czyta
  z VRAM-u wszystkie wagi. Stąd przewaga vLLM — continuous batching czyta wagi RAZ dla wielu
  równoległych żądań, więc 8 użytkowników kosztuje niemal tyle co 1.
- Ceną jest PagedAttention: pula bloków KV-cache prealokowana z góry (patrz
  `VLLM_GPU_MEMORY_UTILIZATION` w configu). Ollama ładuje leniwie i zwalnia kartę po
  bezczynności, więc grzecznie współdzieli GPU z BGE-M3 — vLLM z założenia nie.
- Przy równoległości 1 vLLM ma prawo PRZEGRAĆ z Ollamą i to nie jest argument przeciw niemu:
  to pomiar innej rzeczy (latencja) niż ta, którą optymalizuje (przepustowość serwera).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

# Jeden timeout dla obu backendów: pytanie z guardem #3 potrafi trwać kilkanaście sekund,
# a benchmark z 8 równoległymi żądaniami stoi w kolejce serwera jeszcze dłużej.
_TIMEOUT_S = 300


def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        return json.loads(resp.read())


# --- Backendy: każdy zwraca {"text": str, "completion_tokens": int | None} ----------------

def _generate_ollama(prompt: str, system: str = "", model: str | None = None,
                     max_tokens: int | None = None) -> dict:
    """Woła lokalny serwer Ollamy (/api/generate). Bez zewnętrznych bibliotek — sam urllib.

    `max_tokens` służy BENCHMARKOWI: w Ollamie limit tokenów wyjścia nazywa się `num_predict`.
    Bez wymuszenia tej samej wartości na obu silnikach porównywalibyśmy różne ilości pracy.
    """
    options = {
        "temperature": config.LLM_TEMPERATURE,  # 0.0 -> mniej zmyślania
        "num_ctx": config.LLM_NUM_CTX,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model or config.LLM_MODEL,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    data = _post_json(f"{config.OLLAMA_URL}/api/generate", payload)
    return {"text": data["response"].strip(), "completion_tokens": data.get("eval_count")}


def _generate_vllm(prompt: str, system: str = "", model: str | None = None,
                   max_tokens: int | None = None) -> dict:
    """Woła serwer vLLM przez jego API zgodne z OpenAI (/v1/chat/completions).

    Różnice wobec Ollamy, o których trzeba wiedzieć:
    - format to `messages` (role system/user), nie płaskie `prompt` + `system`;
    - okna kontekstu NIE ustawia się per żądanie — vLLM przyjmuje je raz przy starcie serwera
      (`--max-model-len`, u nas `VLLM_MAX_MODEL_LEN` = tyle co `LLM_NUM_CTX`);
    - `model` musi być nazwą, pod którą serwer wystawił wagi (u nas `VLLM_MODEL`).
    """
    messages = []
    if system:
        # Puste `system` pomijamy: część szablonów czatu wstawia wtedy pusty blok systemowy,
        # co bez potrzeby zmienia prompt względem ścieżki Ollamy.
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model or config.VLLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,   # temp 0 obowiązuje na OBU backendach
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    data = _post_json(f"{config.VLLM_URL}/chat/completions", payload)
    return {"text": data["choices"][0]["message"]["content"].strip(),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens")}


_BACKENDS = {"ollama": "_generate_ollama", "vllm": "_generate_vllm"}


# --- Dyspozytor --------------------------------------------------------------------------

def generate(prompt: str, system: str = "", model: str | None = None,
             max_tokens: int | None = None, backend: str | None = None) -> dict:
    """Jedno wejście do modelu; zwraca tekst + liczbę tokenów wyjścia (dla benchmarku).

    Rozwiązanie nazwy przez `globals()` (a nie słownik funkcji) jest celowe: testy podmieniają
    backendy w module i podmiana MUSI być widoczna dla dyspozytora.
    """
    backend = backend or config.LLM_BACKEND
    fn_name = _BACKENDS.get(backend)
    if fn_name is None:
        raise ValueError(f"Nieznany LLM_BACKEND: {backend!r}. Dozwolone: 'ollama', 'vllm'.")
    return globals()[fn_name](prompt, system=system, model=model, max_tokens=max_tokens)


def call_llm(prompt: str, system: str = "", model: str | None = None,
             max_tokens: int | None = None, backend: str | None = None) -> str:
    """Wersja dla produkcyjnej pętli RAG: sam tekst odpowiedzi."""
    return generate(prompt, system=system, model=model,
                    max_tokens=max_tokens, backend=backend)["text"]


def call_ollama(prompt: str, system: str = "", model: str | None = None,
                max_tokens: int | None = None) -> str:
    """WYMUSZONA Ollama, niezależnie od `LLM_BACKEND`. Tego wołają `verify.py` i
    `eval_answers.py`: sędzia (Gemma) stoi na Ollamie także wtedy, gdy generator liczy na
    vLLM — dwa silniki z modelami 7B i 9B nie zmieszczą się naraz w 12 GB VRAM."""
    return _generate_ollama(prompt, system=system, model=model, max_tokens=max_tokens)["text"]


def call_vllm(prompt: str, system: str = "", model: str | None = None,
              max_tokens: int | None = None) -> str:
    """WYMUSZONY vLLM, niezależnie od `LLM_BACKEND` (używa go benchmark)."""
    return _generate_vllm(prompt, system=system, model=model, max_tokens=max_tokens)["text"]
