"""
src/generate/answer.py
Etap 3 / checkpoint #7: pełna pętla RAG.

    pytanie -> retrieval (hybryda dense+BM25 + rerank) -> budowa promptu z kontekstem
    -> lokalny LLM (Ollama) -> odpowiedź z CYTATAMI źródeł.

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
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.retrieve.hybrid import hybrid_search

# Instrukcja systemowa — tu "mieszka" polityka anty-halucynacyjna i wymóg cytowania.
SYSTEM_PROMPT = """Jesteś skrupulatnym analitykiem. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów \
podanych w sekcji KONTEKST. Zasady:
1. NIE korzystaj z własnej wiedzy ogólnej i NIE wykonuj obliczeń. Jedynym dopuszczalnym źródłem \
jest KONTEKST poniżej — nawet jeśli znasz odpowiedź skądinąd.
2. Jeśli w KONTEKŚCIE nie ma odpowiedzi — również gdy pytanie dotyczy wiedzy ogólnej, definicji \
czy rachunków — napisz dokładnie: "Nie znajduję odpowiedzi w źródłach." i niczego nie zmyślaj.
3. Po każdej tezie podawaj numer źródła w nawiasach kwadratowych, np. [1], [2].
4. Odpowiadaj zwięźle i rzeczowo, po polsku."""


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


def call_ollama(prompt: str, system: str = SYSTEM_PROMPT, model: str | None = None) -> str:
    """Woła lokalny serwer Ollamy (/api/generate). Bez zewnętrznych bibliotek — sam urllib.
    `system`/`model` można nadpisać (np. porównanie promptów przed/po lub inny model-sędzia)."""
    payload = {
        "model": model or config.LLM_MODEL,
        "system": system,
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


REFUSAL_TEXT = "Nie znajduję odpowiedzi w źródłach."


def _ms(t0: float) -> int:
    """Czas etapu w milisekundach. Mierzymy W ŚRODKU pętli, bo tylko ona zna granice etapów —
    warstwa serwująca zobaczyłaby wyłącznie czas całości."""
    return round((time.perf_counter() - t0) * 1000)


def answer(question: str, top_k: int | None = None, system: str = SYSTEM_PROMPT,
           gate: bool = True, verify: bool | None = None,
           on_event=None) -> dict:
    """Pełna pętla: retrieval -> (guard #2 bramka) -> prompt -> LLM -> (guard #3 weryfikacja).

    Zwraca {answer, sources, blocked_by?, rejected_answer?, grounding?}.

    Retrieval: HYBRYDA (dense+BM25/RRF) + reranking — konfiguracja ZMIERZONA jako najlepsza
    (na 29 pytaniach: recall@1 0.83, recall@5 0.97, MRR 0.89 vs dense 0.45/0.76/0.62).
    `system`/`gate`/`verify` można nadpisać — harness mierzy efekt każdego guardu OSOBNO
    (przecieki i false-refusale), bo guard bez zmierzonego kosztu to nie zabezpieczenie.

    `on_event` (opcjonalny) dostaje zdarzenia o granicach etapów — używa go warstwa serwująca,
    żeby pokazać postęp w UI. Domyślnie None => no-op, więc CLI i harness działają bez zmian.
    Emitujemy ETAPY, a nie tokeny odpowiedzi: guard #3 może odpowiedź odrzucić, więc pokazanie
    jej w trakcie generowania zniweczyłoby sens tego guardu.
    """
    verify = config.VERIFY_GROUNDING if verify is None else verify
    emit = on_event or (lambda **_: None)

    emit(event="stage", stage="retrieval", status="start")
    t = time.perf_counter()
    hits = hybrid_search(question, top_k=top_k, rerank=True)
    # top_score liczymy ZAWSZE (nie tylko przy włączonej bramce), bo UI pokazuje tę liczbę
    # także w trybie demo z rozbrojoną bramką — inaczej nie dałoby się porównać „przed/po".
    top_score = max((h.get("rerank_score", 0.0) for h in hits), default=0.0)
    emit(event="stage", stage="retrieval", status="done", ms=_ms(t),
         top_score=top_score, hits=hits)

    # GUARD #2 — BRAMKA TRAFNOŚCI: jeśli nic trafnego nie wróciło (najwyższy score rerankera
    # < próg), odmawiamy BEZ pytania modelu. Odcina pytania spoza tematyki korpusu (obliczenia,
    # wiedza ogólna, inne regiony) niezależnie od tego, co „chciałby" powiedzieć LLM.
    # Czego NIE odcina: pytań NA TEMAT, na które korpus nie ma odpowiedzi — patrz guard #3.
    if gate:
        passed = top_score >= config.RELEVANCE_GATE_MIN
        emit(event="stage", stage="gate", status="done", passed=passed,
             top_score=top_score, threshold=config.RELEVANCE_GATE_MIN)
        if not passed:
            # Etapy, które NIGDY się nie wykonają, muszą to powiedzieć wprost — inaczej panel
            # przebiegu wisiałby w „w toku" i użytkownik czekałby na coś, co nie nadejdzie.
            emit(event="stage", stage="generation", status="not_run")
            emit(event="stage", stage="verify", status="not_run")
            return {"answer": REFUSAL_TEXT, "sources": hits, "blocked_by": "gate"}
    else:
        emit(event="stage", stage="gate", status="skipped")

    context = build_context(hits)
    emit(event="stage", stage="generation", status="start")
    t = time.perf_counter()
    text = call_ollama(build_prompt(question, hits), system=system)
    emit(event="stage", stage="generation", status="done", ms=_ms(t))

    # GUARD #3 — WERYFIKACJA UGRUNTOWANIA: odpowiedź już powstała, więc dopiero TERAZ widać
    # błędy generacji (błędna atrybucja podmiotu, oferta podana jako rozstrzygnięcie). Sędzia
    # (inny model niż generator) ocenia każdą tezę; za mało popartych -> odmowa zamiast publikacji.
    if verify and not refused(text):
        from src.generate.verify import grounded_fraction
        emit(event="stage", stage="verify", status="start")
        t = time.perf_counter()
        frac, verdicts = grounded_fraction(context, text)
        emit(event="stage", stage="verify", status="done", ms=_ms(t),
             grounded_fraction=frac, verdicts=verdicts)
        if frac < config.VERIFY_MIN_SUPPORTED:
            # Zachowujemy odrzuconą odpowiedź i werdykty — bez tego guard byłby czarną skrzynką
            # (nie dałoby się sprawdzić, czy ubił halucynację, czy trafną parafrazę).
            return {"answer": REFUSAL_TEXT, "sources": hits, "blocked_by": "verify",
                    "rejected_answer": text, "grounding": verdicts}
        return {"answer": text, "sources": hits, "grounding": verdicts}

    # `skipped` = wyłączone przełącznikiem; `not_run` = model odmówił sam (guard #1), więc nie
    # ma czego weryfikować. Dwa różne powody, dwa różne komunikaty w UI.
    emit(event="stage", stage="verify", status="skipped" if not verify else "not_run")
    return {"answer": text, "sources": hits}


def refused(text: str) -> bool:
    """Czy model sam odmówił (guard #1 = instrukcja w promptcie). Jedno miejsce na tę regułę,
    żeby harness ewaluacyjny i runtime nie rozjechały się w definicji odmowy."""
    return REFUSAL_TEXT.lower().rstrip(".") in text.lower()


def _print(question: str, result: dict) -> None:
    print(f"\nPYTANIE: {question}\n")
    print(result["answer"])
    if result.get("blocked_by"):
        # Pokazujemy, KTÓRY guard zablokował i dlaczego — inaczej odmowa jest nieodróżnialna od
        # „model nic nie wie", a użytkownik nie wie, czy szukać dalej.
        print(f"\n[zablokowane przez guard: {result['blocked_by']}]")
        for v in result.get("grounding") or []:
            if not v["grounded"]:
                print(f"  NIEPOPARTA TEZA: {v['claim'][:110]}\n    sędzia: {v['reason'][:150]}")
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
