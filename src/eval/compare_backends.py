"""
src/eval/compare_backends.py
Etap 7 / checkpoint #9 cz. 2: czy zmiana backendu psuje JAKOŚĆ, a nie tylko przyspiesza?

DLACZEGO ten skrypt musi istnieć obok benchmarku:
Przełączając Ollama -> vLLM zmieniamy DWIE rzeczy naraz: silnik serwowania ORAZ kwantyzację
(GGUF Q4_K_M -> AWQ 4-bit). Kwantyzacja to inne zaokrąglenia wag, czyli inny model numerycznie —
szybszy silnik z gorszymi odpowiedziami byłby regresją, tylko ładnie opakowaną. Mierzymy więc
to samo, czego pilnuje produkcja: odmowy (guard #1), cytowania i UGRUNTOWANIE tez (sędzia
guarda #3, Gemma).

DLACZEGO CZTERY FAZY, a nie jeden przebieg:
Na 12 GB VRAM vLLM (~8 GB), stack retrievalowy (BGE-M3 + cross-encoder) i sędzia Gemma 9B
(~5,4 GB) NIE mieszczą się razem — sprawdzone: przy działającym vLLM zapytanie pada na
`CUBLAS_STATUS_ALLOC_FAILED`. Dlatego rozdzielamy w czasie to, czego nie da się rozdzielić
w pamięci:
  1. --prepare   retrieval RAZ, na wolnej karcie -> prompty i konteksty na dysk
                 (efekt uboczny: oba backendy dostają IDENTYCZNE wejście, więc różnica
                  w odpowiedziach pochodzi z modelu, a nie z innego losowego retrievalu)
  2. --generate  tylko generacja (karta zajęta przez jeden silnik), per backend
  3. --judge     tylko sędzia na Ollamie (vLLM musi być wtedy zgaszony)
  4. --report    porównanie

CELOWO BEZ BRAMKI (guard #2): bramka patrzy na score rerankera, czyli jest identyczna dla obu
backendów i zamaskowałaby to, co nas tu interesuje — czy MODEL sam odmawia. Raport pokazuje
osobno, ile pytań bramka i tak by zatrzymała.

Uruchomienie:
    python src/eval/compare_backends.py --prepare                       # wolna karta
    LLM_BACKEND=vllm   python src/eval/compare_backends.py --generate    # vLLM wstał
    LLM_BACKEND=ollama python src/eval/compare_backends.py --generate    # vLLM zgaszony
    python src/eval/compare_backends.py --judge                          # sędzia Gemma
    python src/eval/compare_backends.py --report
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate import llm as llm_mod

RESULTS = config.ROOT / "results"
PROMPTS = RESULTS / "backend_prompts.json"


def _answers_path(backend: str) -> Path:
    return RESULTS / f"backend_answers_{backend}.json"


# --- Faza 1: prompty (raz, na wolnej karcie) ----------------------------------------------

def prepare() -> None:
    from src.generate.answer import SYSTEM_PROMPT, build_context, build_prompt
    from src.retrieve.hybrid import hybrid_search

    rows = [json.loads(l) for l in config.GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows, 1):
        hits = hybrid_search(r["question"], rerank=True)
        out.append({
            "id": r["id"],
            "question": r["question"],
            "type": r["type"],
            "in_corpus": bool(r["relevant_chunk_ids"]),
            "prompt": build_prompt(r["question"], hits),
            "context": build_context(hits),
            # Zapisujemy score, żeby raport mógł powiedzieć, ile z tych pytań bramka
            # zatrzymałaby PRZED modelem — inaczej mieszalibyśmy zasługi guardów.
            "top_score": max((h.get("rerank_score", 0.0) for h in hits), default=0.0),
        })
        print(f"  [{i}/{len(rows)}] {r['id']} top_score={out[-1]['top_score']:.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(json.dumps({"system": SYSTEM_PROMPT, "rows": out},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano {len(out)} promptów -> {PROMPTS}")


# --- Faza 2: generacja (jeden backend na raz) ---------------------------------------------

def generate() -> None:
    data = json.loads(PROMPTS.read_text(encoding="utf-8"))
    backend = config.LLM_BACKEND
    model = config.VLLM_MODEL if backend == "vllm" else config.LLM_MODEL
    print(f"=== GENERACJA: backend={backend} model={model} pytań={len(data['rows'])} ===")

    out = []
    for i, r in enumerate(data["rows"], 1):
        t = time.perf_counter()
        res = llm_mod.generate(r["prompt"], system=data["system"])
        ms = round((time.perf_counter() - t) * 1000)
        out.append({**{k: r[k] for k in ("id", "question", "type", "in_corpus", "top_score")},
                    "answer": res["text"], "tokens": res["completion_tokens"], "ms": ms})
        print(f"  [{i}/{len(data['rows'])}] {r['id']} {ms} ms: {res['text'][:70]}...")

    path = _answers_path(backend)
    path.write_text(json.dumps({"backend": backend, "model": model,
                                "when": time.strftime("%Y-%m-%d %H:%M"), "rows": out},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano -> {path}")


# --- Faza 3: sędzia (Gemma na Ollamie; vLLM musi być zgaszony) ----------------------------

def judge() -> None:
    from src.generate.answer import refused
    from src.generate.verify import grounded_fraction

    contexts = {r["id"]: r["context"] for r in json.loads(PROMPTS.read_text(encoding="utf-8"))["rows"]}
    for path in sorted(RESULTS.glob("backend_answers_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n=== SĘDZIA ({config.JUDGE_MODEL}) dla {data['backend']} ===")
        for i, r in enumerate(data["rows"], 1):
            # Oceniamy tylko odpowiedzi UDZIELONE na pytania Z KORPUSU: odmowa nie ma tez do
            # sprawdzenia, a pytania spoza korpusu mierzymy odsetkiem odmów, nie ugruntowaniem.
            if not r["in_corpus"] or refused(r["answer"]):
                r["grounded_fraction"] = None
                continue
            frac, verdicts = grounded_fraction(contexts[r["id"]], r["answer"])
            r["grounded_fraction"] = frac
            r["verdicts"] = verdicts
            print(f"  [{i}/{len(data['rows'])}] {r['id']} ugruntowanie {frac:.2f}")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  zapisano werdykty -> {path}")


# --- Faza 4: raport ----------------------------------------------------------------------

def _stats(data: dict) -> dict:
    from src.generate.answer import refused

    rows = data["rows"]
    inc = [r for r in rows if r["in_corpus"]]
    ooc = [r for r in rows if not r["in_corpus"]]
    answered = [r for r in inc if not refused(r["answer"])]
    judged = [r for r in inc if r.get("grounded_fraction") is not None]
    return {
        "backend": data["backend"],
        "model": data["model"],
        # Guard #1 = sama instrukcja w promptcie (bramka wyłączona, patrz docstring).
        # Zaokrąglamy do 3 miejsc: przy n=29 dalsze cyfry to pozorna precyzja (1 pytanie = 0,034).
        "refusal_ooc": round(sum(refused(r["answer"]) for r in ooc) / len(ooc), 3) if ooc else None,
        # Odmowa na pytaniu Z KORPUSU = false-refusal, czyli koszt ostrożności.
        "false_refusal_in": round(sum(refused(r["answer"]) for r in inc) / len(inc), 3) if inc else None,
        # Cytowanie [n] to wymóg promptu — brak markera znaczy, że model przestał go respektować.
        "cited": round(sum(bool(re.search(r"\[\d+\]", r["answer"])) for r in answered) / len(answered), 3)
                 if answered else None,
        "grounded_mean": round(statistics.mean(r["grounded_fraction"] for r in judged), 3) if judged else None,
        "grounded_n": len(judged),
        "tokens_mean": round(statistics.mean(r["tokens"] for r in rows if r["tokens"]), 1),
        "ms_mean": round(statistics.mean(r["ms"] for r in rows)),
    }


def report() -> None:
    files = sorted(RESULTS.glob("backend_answers_*.json"))
    if not files:
        sys.exit("Brak results/backend_answers_*.json — najpierw --generate.")
    stats = [_stats(json.loads(f.read_text(encoding="utf-8"))) for f in files]

    print("\n=== JAKOŚĆ: Ollama (GGUF Q4_K_M) vs vLLM (AWQ) — te same prompty, bramka WYŁĄCZONA ===")
    hdr = f"  {'metryka':<34}" + "".join(f"{s['backend']:>16}" for s in stats)
    print(hdr)
    for key, label in [("refusal_ooc", "odmowa spoza korpusu (guard #1)"),
                       ("false_refusal_in", "false-refusal na korpusie"),
                       ("cited", "odpowiedzi z cytatem [n]"),
                       ("grounded_mean", "ugruntowanie (sędzia Gemma)"),
                       ("tokens_mean", "średnio tokenów odpowiedzi"),
                       ("ms_mean", "średnio ms na odpowiedź")]:
        cells = "".join(f"{str(s[key]):>16}" for s in stats)
        print(f"  {label:<34}{cells}")
    n = [s["grounded_n"] for s in stats]
    print(f"  {'(ugruntowanie liczone na n)':<34}" + "".join(f"{x:>16}" for x in n))

    gate_blocked = sum(1 for f in files for r in json.loads(f.read_text(encoding="utf-8"))["rows"]
                       if not r["in_corpus"] and r["top_score"] < config.RELEVANCE_GATE_MIN) // max(len(files), 1)
    print(f"\n  Dla kontekstu: bramka (próg {config.RELEVANCE_GATE_MIN}) zatrzymałaby {gate_blocked} "
          f"pytań spoza korpusu PRZED modelem — te liczby mierzą sam model, nie cały pipeline.")

    # Pokazujemy realne rozjazdy: te same prompty, różne odpowiedzi -> materiał do oceny ręcznej.
    if len(files) == 2:
        a, b = (json.loads(f.read_text(encoding="utf-8")) for f in files)
        from src.generate.answer import refused
        diffs = [(ra["id"], ra["question"], refused(ra["answer"]), refused(rb["answer"]))
                 for ra, rb in zip(a["rows"], b["rows"])
                 if refused(ra["answer"]) != refused(rb["answer"])]
        if diffs:
            print(f"\n  ROZJAZD ODMÓW ({a['backend']} vs {b['backend']}):")
            for i, q, ora, orb in diffs:
                print(f"    {i}: {q[:60]}  -> {a['backend']}={'odmowa' if ora else 'odpowiedź'}, "
                      f"{b['backend']}={'odmowa' if orb else 'odpowiedź'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Porównanie JAKOŚCI backendów LLM")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        prepare()
    elif args.generate:
        generate()
    elif args.judge:
        judge()
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
