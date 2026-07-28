"""
src/eval/bench_llm.py
Etap 7 / checkpoint #9 cz. 2: benchmark backendów serwowania LLM (Ollama vs vLLM).

CO MIERZYMY I DLACZEGO WŁAŚNIE TO:
Przewagą vLLM jest PRZEPUSTOWOŚĆ SERWERA pod obciążeniem (continuous batching), a nie czas
jednej odpowiedzi. Dekodowanie jest ograniczone przepustowością pamięci — na każdy token GPU
czyta całe wagi — więc gdy kilka żądań dekoduje równocześnie, wagi czyta się RAZ dla wszystkich.
Dlatego mierzymy przy ROSNĄCEJ RÓWNOLEGŁOŚCI (1, 4, 8) dwie liczby jednocześnie:
  - tokeny/s CAŁEGO serwera  -> ta powinna rosnąć z równoległością (to zysk vLLM),
  - latencja p50/p95 jednego żądania -> ta zwykle się POGARSZA (to koszt kolejkowania).
Pomiar tylko jednej z nich jest nieuczciwy w jedną albo drugą stronę.

UCZCIWOŚĆ POMIARU (bez tego liczby są bezwartościowe):
1. IDENTYCZNE prompty dla obu backendów. Budujemy je RAZ (`--prepare`) na prawdziwych chunkach
   z gold setu i zapisujemy do pliku. Efekt uboczny jest ważny: w czasie pomiaru nie działa
   BGE-M3 ani cross-encoder, więc nie zabierają VRAM-u ani GPU silnikowi, który testujemy.
2. RÓŻNE prompty w obrębie przebiegu (nie jedno pytanie n razy). Powtarzanie tego samego
   promptu wpuszcza vLLM w automatyczny cache prefiksu — dostalibyśmy fałszywą przewagę.
3. Ta sama STAŁA liczba tokenów wyjścia (`BENCH_MAX_TOKENS`) i temperatura 0 na obu silnikach.
4. ROZGRZEWKA (1 żądanie) poza pomiarem: pierwsze żądanie płaci za wciągnięcie wag do VRAM
   (Ollama ładuje leniwie), a to koszt jednorazowy, nie właściwość silnika.
5. Tokeny liczy SERWER (`eval_count` / `usage.completion_tokens`), nie my ze słów.

Uruchomienie:
    python src/eval/bench_llm.py --prepare                    # raz; wymaga Qdranta i GPU
    LLM_BACKEND=ollama python src/eval/bench_llm.py --label ollama-qwen7b-q4
    LLM_BACKEND=vllm   python src/eval/bench_llm.py --label vllm-qwen7b-awq
    python src/eval/bench_llm.py --compare                    # tabela "przed/po" z zapisanych plików
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.generate import llm as llm_mod

PROMPTS_PATH = config.ROOT / "results" / "bench_prompts.json"
RESULTS_DIR = config.ROOT / "results"


# --- Etap 0: przygotowanie promptów (raz, z retrievalem) ----------------------------------

def prepare(limit: int | None = None) -> None:
    """Buduje prompty RAG dla pytań z korpusu i zapisuje je na dysk.

    Prompty muszą powstać z PRAWDZIWEGO retrievalu, bo długość kontekstu (~5 chunków po ~1200
    znaków) decyduje o koszcie fazy prefill — sztuczny krótki prompt zmierzyłby inny problem.
    """
    from src.generate.answer import SYSTEM_PROMPT, build_prompt
    from src.retrieve.hybrid import hybrid_search

    rows = [json.loads(l) for l in config.GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Tylko pytania Z KORPUSU: pytanie spoza korpusu skończyłoby się krótką odmową, czyli
    # kilkoma tokenami wyjścia — mierzylibyśmy głównie prefill i szum.
    rows = [r for r in rows if r.get("type") != "poza-korpusem"]
    if limit:
        rows = rows[:limit]

    prompts = []
    for i, r in enumerate(rows, 1):
        hits = hybrid_search(r["question"], rerank=True)
        prompts.append({"id": r["id"], "question": r["question"], "prompt": build_prompt(r["question"], hits)})
        print(f"  [{i}/{len(rows)}] {r['id']}  ({len(prompts[-1]['prompt'])} znaków promptu)")

    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH.write_text(json.dumps(
        {"system": SYSTEM_PROMPT, "prompts": prompts}, ensure_ascii=False, indent=2), encoding="utf-8")
    avg = statistics.mean(len(p["prompt"]) for p in prompts)
    print(f"\nZapisano {len(prompts)} promptów -> {PROMPTS_PATH} (średnio {avg:.0f} znaków)")


# --- Pomiar ------------------------------------------------------------------------------

def _one_request(prompt: str, system: str, max_tokens: int) -> tuple[float, int | None]:
    t0 = time.perf_counter()
    out = llm_mod.generate(prompt, system=system, max_tokens=max_tokens)
    return time.perf_counter() - t0, out["completion_tokens"]


def run_level(prompts: list[dict], system: str, concurrency: int,
              n_requests: int, max_tokens: int) -> dict:
    """Wysyła `n_requests` żądań, utrzymując dokładnie `concurrency` w locie.

    ThreadPoolExecutor(max_workers=concurrency) modeluje N równoległych klientów: gdy jedno
    żądanie się kończy, natychmiast startuje następne — tak zachowuje się realny ruch.
    """
    tasks = [prompts[i % len(prompts)] for i in range(n_requests)]
    if n_requests > len(prompts):
        print(f"    UWAGA: {n_requests} żądań > {len(prompts)} promptów -> prompty się powtórzą "
              f"(cache prefiksu vLLM może zawyżyć wynik)")

    latencies: list[float] = []
    tokens = 0
    tokens_known = True
    errors: list[str] = []

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one_request, t["prompt"], system, max_tokens) for t in tasks]
        for f in as_completed(futures):
            try:
                lat, tok = f.result()
            except Exception as e:                      # noqa: BLE001 - chcemy zobaczyć KAŻDY błąd
                errors.append(f"{type(e).__name__}: {e}")
                continue
            latencies.append(lat)
            if tok is None:
                tokens_known = False
            else:
                tokens += tok
    wall = time.perf_counter() - t0

    ok = len(latencies)
    lat_sorted = sorted(latencies)
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "ok": ok,
        "errors": errors,
        "wall_s": round(wall, 2),
        "completion_tokens": tokens if tokens_known else None,
        # Przepustowość CAŁEGO serwera: to ta liczba ma rosnąć z równoległością.
        "tokens_per_s": round(tokens / wall, 1) if tokens_known and wall > 0 else None,
        "requests_per_s": round(ok / wall, 3) if wall > 0 else None,
        # Latencja pojedynczego żądania: ta ma prawo rosnąć — to koszt kolejkowania.
        "lat_p50_s": round(statistics.median(lat_sorted), 2) if ok else None,
        "lat_p95_s": round(lat_sorted[max(0, int(0.95 * ok) - 1)], 2) if ok else None,
        "lat_max_s": round(lat_sorted[-1], 2) if ok else None,
    }


def bench(label: str, n_requests: int, max_tokens: int, levels: tuple[int, ...]) -> dict:
    if not PROMPTS_PATH.exists():
        sys.exit(f"Brak {PROMPTS_PATH}. Najpierw: python src/eval/bench_llm.py --prepare")
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    prompts, system = data["prompts"], data["system"]

    backend = config.LLM_BACKEND
    model = config.VLLM_MODEL if backend == "vllm" else config.LLM_MODEL
    print(f"\n=== BENCHMARK: backend={backend}  model={model}  "
          f"max_tokens={max_tokens}  promptów={len(prompts)} ===")

    print("  rozgrzewka (poza pomiarem)...", end="", flush=True)
    t0 = time.perf_counter()
    _one_request(prompts[0]["prompt"], system, max_tokens)
    print(f" {time.perf_counter() - t0:.1f} s")

    results = []
    for c in levels:
        print(f"  równoległość {c}: {n_requests} żądań...", end="", flush=True)
        r = run_level(prompts, system, c, n_requests, max_tokens)
        print(f" {r['wall_s']} s, {r['tokens_per_s']} tok/s, p95 {r['lat_p95_s']} s"
              + (f", BŁĘDY: {len(r['errors'])}" if r["errors"] else ""))
        results.append(r)

    out = {
        "label": label,
        "backend": backend,
        "model": model,
        "max_tokens": max_tokens,
        "n_requests": n_requests,
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "levels": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"bench_llm_{label}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano -> {path}")
    _print_one(out)
    return out


# --- Raportowanie ------------------------------------------------------------------------

def _print_one(res: dict) -> None:
    print(f"\n{res['label']}  ({res['backend']}, {res['model']})")
    print("  równoległość | tok/s serwera | żądania/s | p50 [s] | p95 [s] | ściana [s]")
    for lv in res["levels"]:
        print(f"  {lv['concurrency']:>12} | {str(lv['tokens_per_s']):>13} | "
              f"{str(lv['requests_per_s']):>9} | {str(lv['lat_p50_s']):>7} | "
              f"{str(lv['lat_p95_s']):>7} | {str(lv['wall_s']):>10}")


def compare() -> None:
    files = sorted(RESULTS_DIR.glob("bench_llm_*.json"))
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    if not runs:
        sys.exit("Brak plików results/bench_llm_*.json — najpierw uruchom benchmark.")
    for r in runs:
        _print_one(r)

    # Baseline = Ollama (stan projektu przed etapem 7). Każdy przebieg vLLM porównujemy do niego
    # OSOBNO — przebiegów vLLM bywa kilka (różne `--util`), a milcząco wybrany jeden dawałby
    # wrażenie, że zmierzyliśmy tylko jedną konfigurację.
    base = next((r for r in runs if r["backend"] == "ollama"), None)
    rivals = [r for r in runs if r["backend"] == "vllm"]
    if not (base and rivals):
        return
    for rival in rivals:
        print(f"\n=== PRZED/PO: {base['label']} -> {rival['label']} ===")
        print("  równoległość | tok/s przed | tok/s po | zmiana | p95 przed | p95 po")
        for a, b in zip(base["levels"], rival["levels"]):
            if a["tokens_per_s"] and b["tokens_per_s"]:
                ratio = b["tokens_per_s"] / a["tokens_per_s"]
                change = f"x{ratio:.2f}" + ("" if ratio >= 1 else "  (GORZEJ)")
            else:
                change = "n/d"
            print(f"  {a['concurrency']:>12} | {str(a['tokens_per_s']):>11} | {str(b['tokens_per_s']):>8} | "
                  f"{change:>16} | {str(a['lat_p95_s']):>9} | {str(b['lat_p95_s']):>6}")
    print("\n  Czytanie tabeli: tok/s to przepustowość CAŁEGO serwera (rośnie z równoległością,")
    print("  jeśli silnik batchuje), p95 to koszt dla pojedynczego użytkownika (zwykle rośnie).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark backendów LLM (Ollama vs vLLM)")
    ap.add_argument("--prepare", action="store_true", help="zbuduj prompty z gold setu (wymaga Qdranta+GPU)")
    ap.add_argument("--limit", type=int, help="tylko N pytań przy --prepare")
    ap.add_argument("--compare", action="store_true", help="tabela przed/po z zapisanych wyników")
    ap.add_argument("--label", default=None, help="nazwa przebiegu (do nazwy pliku wyniku)")
    ap.add_argument("--requests", type=int, default=8, help="ile żądań na poziom równoległości")
    ap.add_argument("--max-tokens", type=int, default=config.BENCH_MAX_TOKENS)
    ap.add_argument("--levels", type=int, nargs="+", default=list(config.BENCH_CONCURRENCY))
    args = ap.parse_args()

    if args.prepare:
        prepare(args.limit)
    elif args.compare:
        compare()
    else:
        label = args.label or config.LLM_BACKEND
        bench(label, args.requests, args.max_tokens, tuple(args.levels))


if __name__ == "__main__":
    main()
