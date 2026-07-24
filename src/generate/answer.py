"""
src/generate/answer.py
Etap 3 / checkpoint #7: pełna pętla RAG.

    pytanie -> retrieval (search.py) -> budowa promptu z kontekstem -> lokalny LLM (Ollama)
    -> odpowiedź z CYTATAMI źródeł.

KLUCZOWE DECYZJE (i dlaczego):
- CYTOWANIE: numerujemy chunki [1], [2]... z etykietą źródła i każemy modelowi podawać [n]
  przy każdej tezie. Dzięki temu odpowiedź jest WERYFIKOWALNA — analityk może sprawdzić źródło,
  a nie ufać "gadającej" maszynie. To realizuje projektowe "cytuje źródło".
- ANTY-HALUCYNACJE: najważniejsza instrukcja to nie samo "odpowiadaj z kontekstu", ale danie
  modelowi FURTKI: "jeśli odpowiedzi nie ma w źródłach — powiedz, że nie znajdujesz". Bez tej
  furtki model, zmuszony do odpowiedzi, zmyśla. + temperatura 0 (config). (Szerzej: checkpoint #8.)
- Wołamy Ollamę czystym urllib (stdlib) — bez dodatkowej zależności (cienka własna warstwa).

Uruchomienie (wymaga: Qdrant w Dockerze + Ollama z modelem):
    python src/generate/answer.py "Dlaczego Szwecja i Finlandia wstąpiły do NATO?"
    python src/generate/answer.py            # tryb interaktywny
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.retrieve.search import search

# Instrukcja systemowa — tu "mieszka" polityka anty-halucynacyjna i wymóg cytowania.
SYSTEM_PROMPT = """Jesteś skrupulatnym analitykiem. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów \
podanych w sekcji KONTEKST. Zasady:
1. Jeśli w KONTEKŚCIE nie ma odpowiedzi, napisz dokładnie: "Nie znajduję odpowiedzi w źródłach." \
i niczego nie zmyślaj ani nie zgaduj.
2. Po każdej tezie podawaj numer źródła w nawiasach kwadratowych, np. [1], [2].
3. Odpowiadaj zwięźle i rzeczowo, po polsku."""


def build_context(hits: list[dict]) -> str:
    """Skleja pobrane chunki w ponumerowany kontekst z etykietą źródła (pod cytowanie)."""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] (źródło: {h['source']} #{h['chunk_index']})\n{h['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict]) -> str:
    return (
        f"KONTEKST:\n{build_context(hits)}\n\n"
        f"PYTANIE: {question}\n\n"
        f"ODPOWIEDŹ (z cytatami [n]):"
    )


def call_ollama(prompt: str) -> str:
    """Woła lokalny serwer Ollamy (/api/generate). Bez zewnętrznych bibliotek — sam urllib."""
    payload = {
        "model": config.LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.LLM_TEMPERATURE,  # 0.0 -> mniej zmyślania
            "num_ctx": config.LLM_NUM_CTX,
        },
    }
    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["response"].strip()


def answer(question: str, top_k: int | None = None) -> dict:
    """Pełna pętla: retrieval -> prompt -> LLM. Zwraca {answer, sources}."""
    hits = search(question, top_k=top_k)
    text = call_ollama(build_prompt(question, hits))
    return {"answer": text, "sources": hits}


def _print(question: str, result: dict) -> None:
    print(f"\nPYTANIE: {question}\n")
    print(result["answer"])
    print("\nŹRÓDŁA (kandydaci przekazani do modelu):")
    for i, h in enumerate(result["sources"], 1):
        print(f"  [{i}] {h['source']} #{h['chunk_index']}  (score {h['score']:.3f})")


def main() -> None:
    args = sys.argv[1:]
    if args:
        q = " ".join(args)
        _print(q, answer(q))
    else:
        print("Tryb interaktywny RAG. Wpisz pytanie (puste = koniec).")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            _print(q, answer(q))


if __name__ == "__main__":
    main()
