"""
src/serve/vllm_serve.py
Etap 7 / checkpoint #9 cz. 2: uruchamia lokalny serwer vLLM z parametrami Z CONFIGU.

DLACZEGO to osobny launcher, a nie polecenie w README:
- Parametry, które MUSZĄ się zgadzać z klientem (`VLLM_MODEL`, port z `VLLM_URL`, okno kontekstu
  `VLLM_MAX_MODEL_LEN` = `LLM_NUM_CTX`), mają jedno źródło prawdy. Rozjazd okna kontekstu między
  serwerem a benchmarkiem unieważniłby porównanie i nikt by tego nie zauważył.
- vLLM żyje w OSOBNYM ŚRODOWISKU CONDA (`vllm-server`). Powód: przyciąga własną wersję torcha
  (u nas 2.11.0+cu130), a nasz stack retrievalowy stoi na torchu cu128 zbudowanym pod Blackwella
  (sm_120) — instalacja w jednym env zdegradowałaby BGE-M3 i cross-encoder. vLLM wystawia HTTP
  zgodny z OpenAI, więc rozdzielenie środowisk nic nie kosztuje architektonicznie.
- Runtime offline: wymuszamy `HF_HUB_OFFLINE=1`, więc serwer nie sięgnie po sieć (wagi są już
  w cache HF). To twarda zasada projektu, a nie deklaracja w dokumentacji.

Uruchomienie (osobny terminal, karta musi być wolna):
    python src/serve/vllm_serve.py
    python src/serve/vllm_serve.py --util 0.55      # gdy coś jeszcze trzyma VRAM
    python src/serve/vllm_serve.py --eager          # mniej VRAM (bez grafów CUDA), wolniej
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

# Interpreter z osobnego env `vllm-server` (patrz docstring). Nadpisywalny środowiskiem,
# żeby ścieżka do cudzej instalacji condy nie była wbita w kod na sztywno.
VLLM_PYTHON = os.environ.get(
    "VLLM_ENV_PYTHON", str(Path.home() / "miniconda3" / "envs" / "vllm-server" / "bin" / "python")
)


def _free_vram_mib() -> tuple[int, int, list[str]]:
    """Ile VRAM-u jest wolne i KTO trzyma resztę. vLLM prealokuje pulę KV-cache na starcie,
    więc konflikt o pamięć objawia się jako CUDA OOM w chwili startu — lepiej ostrzec wcześniej."""
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().split(", ")
    used, total = int(q[0]), int(q[1])
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return total - used, total, [a for a in apps if a]


def main() -> None:
    ap = argparse.ArgumentParser(description="Start lokalnego serwera vLLM (offline)")
    ap.add_argument("--util", type=float, default=config.VLLM_GPU_MEMORY_UTILIZATION,
                    help="ułamek CAŁEGO VRAM-u prealokowany przez vLLM (wagi + pula KV-cache)")
    ap.add_argument("--eager", action="store_true",
                    help="bez grafów CUDA: mniej VRAM-u, niższa wydajność")
    ap.add_argument("--dry-run", action="store_true", help="tylko wypisz polecenie")
    args = ap.parse_args()

    port = urlparse(config.VLLM_URL).port or 8000
    free, total, apps = _free_vram_mib()
    need = int(args.util * total)
    print(f"GPU: wolne {free} / {total} MiB; vLLM zażąda ~{need} MiB (util {args.util})")
    for a in apps:
        print(f"  zajmuje: {a}")
    if need > free:
        print(f"\nUWAGA: budżet vLLM ({need} MiB) > wolny VRAM ({free} MiB) => spodziewaj się "
              f"CUDA OOM.\n  Zwolnij kartę (zatrzymaj serwer UI, `ollama stop <model>`) albo "
              f"obniż --util.")

    cmd = [
        VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", config.VLLM_MODEL,
        "--port", str(port),
        # Okno kontekstu ustawia się RAZ, przy starcie serwera (nie per żądanie jak num_ctx
        # w Ollamie) — bo od niego zależy rozmiar bloków KV-cache w prealokowanej puli.
        "--max-model-len", str(config.VLLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(args.util),
        # 127.0.0.1, nie 0.0.0.0: model lokalny, nie wystawiamy go w sieci (jak SERVE_HOST).
        "--host", config.SERVE_HOST,
    ]
    if args.eager:
        cmd.append("--enforce-eager")

    print("\n$ " + " ".join(cmd))
    if args.dry_run:
        return
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",          # twarda zasada: runtime bez sieci
        # FlashInfer KOMPILUJE kernel samplingu JIT-em przy starcie i wymaga do tego `nvcc`
        # (pełnego CUDA toolkitu). Koła pip dają tylko runtime CUDA, więc start kończy się
        # `Could not find nvcc`. Wracamy na sampler PyTorcha: my generujemy zachłannie
        # (temperature=0), czyli sampler nie ma nic do losowania i nic nie tracimy.
        # WNIOSEK OGÓLNY (istotny dla środowiska niejawnego): kernele kompilowane JIT-em na
        # maszynie docelowej to zależność od toolchainu, którego w air-gapped zwykle nie ma.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
