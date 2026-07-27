"""
src/eval/eval_answers.py
Etap 4/5 / checkpoint #8: ewaluacja JAKOŚCI ODPOWIEDZI (nie tylko retrievalu).

Mierzymy:
- REFUSAL accuracy „przed/po" — na pytaniach spoza korpusu, dla starego (V1) i wzmocnionego (V2)
  promptu. Sprawdza anty-halucynację (czy model odmawia zamiast zmyślać).
- FALSE-REFUSAL — czy wzmocniony prompt nie zaczął błędnie odmawiać na PYTANIACH Z KORPUSU
  (to koszt twardszej instrukcji — trzeba go zmierzyć).
- FAITHFULNESS (per-teza) i ANSWER RELEVANCE — sędzią jest lokalny Qwen.
  UWAGA CYRKULARNOŚĆ: sędzia = model generujący. Łagodzimy oceną per-teza; traktujemy jako
  SYGNAŁ, nie prawdę absolutną (patrz ręczny sanity-check w wydruku).

Uruchomienie (wolne — dużo wywołań LLM):
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
    return call_ollama(p, system=_JUDGE_FAITH).strip().upper().startswith("TAK")


def judge_relevant(question: str, ans: str) -> bool:
    p = f"PYTANIE: {question}\n\nODPOWIEDŹ: {ans}\n\nCzy odpowiedź jest na temat pytania? TAK/NIE."
    return call_ollama(p, system=_JUDGE_REL).strip().upper().startswith("TAK")


def main() -> None:
    gold = [json.loads(l) for l in config.GOLD_PATH.open(encoding="utf-8")]
    in_corpus = [g for g in gold if g["relevant_chunk_ids"]]
    ooc = [g for g in gold if not g["relevant_chunk_ids"]]

    # --- 1) REFUSAL przed/po (pytania spoza korpusu) ---
    print("=== REFUSAL accuracy na pytaniach spoza korpusu (przed/po wzmocnieniu promptu) ===")
    for label, sysp in [("V1 (stary)", OLD_PROMPT), ("V2 (wzmocniony)", SYSTEM_PROMPT)]:
        leaks = [g["question"] for g in ooc if not refused(answer(g["question"], system=sysp)["answer"])]
        acc = (len(ooc) - len(leaks)) / len(ooc)
        print(f"  {label:16} refusal={acc:.2f}   przecieki: {leaks if leaks else '—'}")

    # --- 2) Jakość odpowiedzi na pytaniach Z KORPUSU (prompt V2) ---
    print("\n=== Jakość odpowiedzi (V2) na pytaniach z korpusu ===")
    faiths, rels, false_refusals = [], [], []
    for g in in_corpus:
        res = answer(g["question"])
        a = res["answer"]
        if refused(a):
            false_refusals.append(g["id"])          # twardszy prompt zaczął błędnie odmawiać?
            continue
        rels.append(1.0 if judge_relevant(g["question"], a) else 0.0)
        claims = to_claims(a)
        if claims:
            ctx = build_context(res["sources"])
            faiths.append(mean([1.0 if judge_faithful(ctx, c) else 0.0 for c in claims]))

    print(f"  answered (nie odmówił błędnie): {len(in_corpus) - len(false_refusals)}/{len(in_corpus)}")
    print(f"  FALSE-REFUSAL (koszt twardszego promptu): {false_refusals if false_refusals else '—'}")
    print(f"  faithfulness (śr. udział tez popartych kontekstem): {mean(faiths):.2f}")
    print(f"  answer relevance (śr. na temat):                    {mean(rels):.2f}")
    print("\n  (Sędzia = lokalny Qwen -> cyrkularność; traktować jako SYGNAŁ + sanity-check ręczny.)")


if __name__ == "__main__":
    main()
