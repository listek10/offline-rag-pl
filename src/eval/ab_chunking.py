"""
src/eval/ab_chunking.py
Etap 8 / checkpointy #2 i #12: A/B strategii chunkingu — DOWÓD zamiast szacowania.

PROBLEM, który to rozwiązuje: rozmiar chunka (1200 znaków, zakładka 180) był dobrany
z rozsądku, nie z pomiaru — a to najczęstsze pytanie na rozmowie o RAG. Tu porównujemy
warianty na tym samym gold secie i tym samym torze retrievalu.

DLACZEGO PRAWDA PRZEZ CYTAT (`--truth evidence`): gold set wskazuje trafne chunki przez
`chunk_id` = `{źródło}__{numer}`. Po zmianie rozmiaru chunka numeracja przesuwa się w całym
korpusie, więc te identyfikatory przestają cokolwiek wskazywać — recall policzyłby się
„poprawnie" i mierzyłby szum. Dlatego każde pytanie ma `evidence`: krótki DOSŁOWNY cytat
z odpowiedzią. Chunk jest trafny, gdy ten cytat zawiera — definicja niezależna od podziału.

CO IZOLUJEMY: zmieniamy rozmiar, trzymając UDZIAŁ zakładki stały (15%). Gdyby zakładka
została w wartościach bezwzględnych, wariant 600-znakowy miałby 30% zakładki, a 2000-znakowy
9% — i nie wiedzielibyśmy, co właściwie zmierzyliśmy. `CHUNK_MIN_CHARS` zostaje 250 dla
wszystkich: to absolutny próg użyteczności fragmentu, nie wielkość względna.

KAŻDY WARIANT W OSOBNYM PROCESIE I OSOBNEJ KOLEKCJI: moduły retrievalu trzymają indeks BM25
i klienta Qdranta w singletonach, więc dwa warianty w jednym procesie zaczytałyby sobie dane.
Osobna kolekcja (`..._c600`) chroni też indeks bazowy przed nadpisaniem.

Uruchomienie (wymaga Qdranta; ~5-8 min na wariant, bo przelicza embeddingi):
    python src/eval/ab_chunking.py                      # 600 / 1200 / 2000
    python src/eval/ab_chunking.py --tylko c600 c1200
    python src/eval/ab_chunking.py --raport             # tabela z zapisanych wyników
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

RESULTS = config.ROOT / "results"
# Zakładka = 15% rozmiaru w KAŻDYM wariancie (patrz docstring).
WARIANTY = {
    "c600":  {"target": 600,  "overlap": 90},
    "c1200": {"target": 1200, "overlap": 180},   # obecny baseline
    "c2000": {"target": 2000, "overlap": 300},
}
CONFIGS = ["dense", "hybrid+rr"]   # najsłabsza i produkcyjna: widać i sufit, i realny wynik


def sciezki(nazwa: str) -> dict:
    return {
        "chunks": config.PROCESSED_DIR / f"chunks_{nazwa}.jsonl",
        "kolekcja": f"{config.QDRANT_COLLECTION}_{nazwa}",
        "wynik": RESULTS / f"ab_chunking_{nazwa}.json",
    }


def uruchom(krok: str, args: list[str], env: dict) -> None:
    print(f"    -> {krok}...", end="", flush=True)
    t = time.perf_counter()
    p = subprocess.run([sys.executable, *args], env={**os.environ, **env},
                       cwd=config.ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        print(" BŁĄD")
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        sys.exit(f"[STOP] krok '{krok}' zakończył się kodem {p.returncode}")
    print(f" {time.perf_counter() - t:.0f} s")


def przelicz(nazwa: str, spec: dict) -> dict:
    s = sciezki(nazwa)
    env = {
        "CHUNK_TARGET_CHARS": str(spec["target"]),
        "CHUNK_OVERLAP_CHARS": str(spec["overlap"]),
        "CHUNKS_PATH": str(s["chunks"]),
        "QDRANT_COLLECTION": s["kolekcja"],
    }
    print(f"\n=== WARIANT {nazwa}: target={spec['target']} zn., zakładka={spec['overlap']} zn. "
          f"({spec['overlap'] / spec['target']:.0%}) ===")
    uruchom("chunking", ["src/ingest/chunk.py"], env)
    uruchom("indeks (embeddingi + Qdrant)", ["src/index/build_index.py"], env)
    uruchom("ewaluacja", ["src/eval/run_eval.py", "--truth", "evidence", "--skip-refusal",
                          "--configs", *CONFIGS, "--json", str(s["wynik"])], env)

    wynik = json.loads(s["wynik"].read_text(encoding="utf-8"))
    chunki = [json.loads(l) for l in s["chunks"].open(encoding="utf-8") if l.strip()]
    wynik["n_chunks"] = len(chunki)
    wynik["avg_chars"] = round(sum(c["n_chars"] for c in chunki) / len(chunki))
    # Ile chunków zawiera cytat-dowód: pokazuje, ile „biletów" daje ten podział. Bez tej
    # liczby nie da się uczciwie zinterpretować różnic w recall (patrz metrics.py).
    wynik["avg_relevant"] = round(
        sum(r["n_relevant"] for r in wynik["per_question"][CONFIGS[0]])
        / len(wynik["per_question"][CONFIGS[0]]), 2)
    wynik["niemozliwe"] = [r["id"] for r in wynik["per_question"][CONFIGS[0]] if r["n_relevant"] == 0]
    s["wynik"].write_text(json.dumps(wynik, ensure_ascii=False, indent=2), encoding="utf-8")
    return wynik


def raport(wyniki: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("=== A/B CHUNKINGU — prawda przez cytat-dowód, 29 pytań z korpusu ===")
    print(f"{'wariant':>8} {'chunków':>8} {'śr. zn.':>8} {'trafnych':>9} "
          + "".join(f"{m:>10}" for m in ("R@1", "R@3", "R@5", "R@10", "MRR")))
    for cfg in CONFIGS:
        print(f"  [{cfg}]")
        for nazwa, w in wyniki.items():
            a = w["aggregates"][cfg]
            print(f"{nazwa:>8} {w['n_chunks']:>8} {w['avg_chars']:>8} {w['avg_relevant']:>9} "
                  f"{a['recall@1']:>10.3f}{a['recall@3']:>10.3f}{a['recall@5']:>10.3f}"
                  f"{a['recall@10']:>10.3f}{a['MRR']:>10.3f}")
    for nazwa, w in wyniki.items():
        if w["niemozliwe"]:
            print(f"  UWAGA {nazwa}: cytat nie mieści się w żadnym chunku dla {w['niemozliwe']} "
                  f"-> te pytania są przy tym podziale NIEMOŻLIWE do trafienia")
    print('\n  „trafnych" = średnia liczba chunków zawierających cytat-dowód. Rośnie, gdy podział')
    print("  duplikuje tekst w zakładce — czyli część różnicy w recall bierze się z redundancji,")
    print("  nie z lepszego dopasowania. Dlatego udział zakładki trzymamy stały (15%).")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B strategii chunkingu (etap 8)")
    ap.add_argument("--tylko", nargs="+", choices=list(WARIANTY), help="policz wybrane warianty")
    ap.add_argument("--raport", action="store_true", help="tylko tabela z zapisanych wyników")
    args = ap.parse_args()

    nazwy = args.tylko or list(WARIANTY)
    if args.raport:
        wyniki = {n: json.loads(sciezki(n)["wynik"].read_text(encoding="utf-8"))
                  for n in nazwy if sciezki(n)["wynik"].exists()}
        if not wyniki:
            sys.exit("Brak zapisanych wyników — uruchom bez --raport.")
    else:
        wyniki = {n: przelicz(n, WARIANTY[n]) for n in nazwy}
    raport(wyniki)


if __name__ == "__main__":
    main()
