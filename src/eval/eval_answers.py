"""
src/eval/eval_answers.py
Etap 4/5 / checkpoint #8: ewaluacja JAKOŚCI ODPOWIEDZI (nie tylko retrievalu).

Mierzymy:
- REFUSAL accuracy „przed/po" (pytania spoza korpusu): stary prompt (V1), wzmocniony (V2) i
  V2 + BRAMKA TRAFNOŚCI (guard na score rerankera). Sprawdza anty-halucynację.
- FALSE-REFUSAL — czy bramka/twardszy prompt nie zaczęły błędnie odmawiać na pytaniach z korpusu.
- FAITHFULNESS (per-teza) i ANSWER RELEVANCE — sędzią jest INNY model (Gemma 2), nie generator
  (Qwen), żeby zmniejszyć cyrkularność (skorelowane błędy generatora i sędziego). Wynik nadal
  traktujemy jako SYGNAŁ + ręczny sanity-check.

Uruchomienie (wolne — dużo wywołań LLM; wymaga modelu-sędziego w Ollamie):
    python src/eval/eval_answers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.eval.metrics import mean
from src.generate.answer import SYSTEM_PROMPT, answer, build_context, call_ollama
from src.ingest.chunk import split_sentences

REFUSAL = "nie znajduję odpowiedzi w źródłach"

# Poprzednia (słabsza) wersja promptu — do porównania „przed/po".
OLD_PROMPT = (
    "Jesteś skrupulatnym analitykiem. Odpowiadasz WYŁĄCZNIE na podstawie fragmentów podanych "
    "w sekcji KONTEKST. Zasady:\n"
    '1. Jeśli w KONTEKŚCIE nie ma odpowiedzi, napisz dokładnie: "Nie znajduję odpowiedzi w źródłach." '
    "i niczego nie zmyślaj ani nie zgaduj.\n"
    "2. Po każdej tezie podawaj numer źródła w nawiasach kwadratowych, np. [1], [2].\n"
    "3. Odpowiadaj zwięźle i rzeczowo, po polsku."
)

_JUDGE_FAITH = ("Jesteś surowym weryfikatorem faktów. Oceniasz, czy TWIERDZENIE wynika WPROST "
                "z KONTEKSTU. Odpowiadasz jednym słowem: TAK albo NIE.")
_JUDGE_REL = ("Oceniasz, czy ODPOWIEDŹ odnosi się do PYTANIA (jest na temat). "
              "Odpowiadasz jednym słowem: TAK albo NIE.")


def refused(text: str) -> bool:
    return REFUSAL in text.lower()


def to_claims(answer_text: str) -> list[str]:
    txt = re.sub(r"\[\d+\]", "", answer_text)          # usuń markery cytowań [n]
    return [s for s in split_sentences(txt) if len(s) > 15]


def judge_faithful(context: str, claim: str) -> bool:
    p = f"KONTEKST:\n{context}\n\nTWIERDZENIE: {claim}\n\nCzy twierdzenie wynika wprost z kontekstu? TAK/NIE."
    return call_ollama(p, system=_JUDGE_FAITH, model=config.JUDGE_MODEL).strip().upper().startswith("TAK")


def judge_relevant(question: str, ans: str) -> bool:
    p = f"PYTANIE: {question}\n\nODPOWIEDŹ: {ans}\n\nCzy odpowiedź jest na temat pytania? TAK/NIE."
    return call_ollama(p, system=_JUDGE_REL, model=config.JUDGE_MODEL).strip().upper().startswith("TAK")


def main() -> None:
    gold = [json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8")]
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]

    # --- 1) REFUSAL: prompt V1/V2 (bez bramki) + V2 z bramką ---
    print(f"=== REFUSAL accuracy na {len(ooc)} pytaniach spoza korpusu ===")
    for label, sysp in [("V1 (bez bramki)", OLD_PROMPT), ("V2 (bez bramki)", SYSTEM_PROMPT)]:
        leaks = [g["question"] for g in ooc if not refused(answer(g["question"], system=sysp, gate=False)["answer"])]
        print(f"  {label:20} refusal={(len(ooc)-len(leaks))/len(ooc):.2f}   przecieki: {leaks or '—'}")
    leaks_g = [g["question"] for g in ooc if not refused(answer(g["question"], gate=True)["answer"])]
    print(f"  {'V2 + BRAMKA':20} refusal={(len(ooc)-len(leaks_g))/len(ooc):.2f}   przecieki: {leaks_g or '—'}")

    # --- 2) Jakość odpowiedzi na pytaniach Z KORPUSU (ścieżka produkcyjna: V2 + bramka) ---
    print(f"\n=== Jakość odpowiedzi (produkcja) na {len(in_corpus)} pytaniach z korpusu; sędzia = {config.JUDGE_MODEL} ===")
    faiths, rels, false_refusals = [], [], []
    for g in in_corpus:
        res = answer(g["question"])
        a = res["answer"]
        if refused(a):
            false_refusals.append(g["id"])
            continue
        rels.append(1.0 if judge_relevant(g["question"], a) else 0.0)
        claims = to_claims(a)
        if claims:
            ctx = build_context(res["sources"])
            faiths.append(mean([1.0 if judge_faithful(ctx, c) else 0.0 for c in claims]))

    print(f"  answered (nie odmówił błędnie): {len(in_corpus) - len(false_refusals)}/{len(in_corpus)}")
    print(f"  FALSE-REFUSAL (koszt bramki/promptu): {false_refusals or '—'}")
    print(f"  faithfulness (śr. udział tez popartych kontekstem): {mean(faiths):.2f}")
    print(f"  answer relevance (śr. na temat):                    {mean(rels):.2f}")
    print(f"\n  (Sędzia = {config.JUDGE_MODEL} != generator {config.LLM_MODEL} -> mniejsza cyrkularność; nadal SYGNAŁ + sanity-check.)")


if __name__ == "__main__":
    main()
