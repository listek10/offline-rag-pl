"""
Centralna konfiguracja projektu — WSZYSTKIE parametry w JEDNYM miejscu.

DLACZEGO tak: chunk size, overlap, top_k, nazwy modeli i ścieżki nie mogą być rozsiane
po kodzie. Jak rekruter zapyta "jaki masz chunk size i czemu?", chcę wskazać jedno miejsce,
a nie polować po plikach. Zmiana parametru w eksperymencie (etap 5: optymalizacja z dowodem)
ma być jednolinijkowa.
"""

from pathlib import Path

# --- Ścieżki ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"   # wynik etapu 2 (jeden chunk = jedna linia JSON)
GOLD_PATH = ROOT / "gold_set" / "gold.jsonl"   # zbiór ewaluacyjny (etap 4)

# --- Chunking (etap 2) ---
# Mierzymy w ZNAKACH jako tani proxy długości. Dokładny pomiar w tokenach wymagałby
# tokenizera modelu embeddingowego — dodamy, jeśli ewaluacja pokaże, że warto.
# ~1200 znaków ≈ ~500 tokenów dla polskiego (polski jest "token-cięższy" niż angielski).
CHUNK_TARGET_CHARS = 1200      # docelowy rozmiar chunka
CHUNK_OVERLAP_CHARS = 180      # ~15% zakładki — by myśl na styku dwóch chunków nie wypadła
CHUNK_MIN_CHARS = 250          # krótsze "ochłapy" doklejamy do poprzedniego, nie emitujemy osobno

# --- Embeddingi (etap 2: indeks) ---
# BGE-M3: wielojęzyczny (dobry polski), 1024 wymiary, normalizuje wektory -> używamy cosine.
# DLACZEGO wielojęzyczny: model EN-only nie był trenowany na polskim, źle radzi sobie z
# fleksją i rozbija polskie słowa tokenizerem na szum. Wektor = punkt w przestrzeni, gdzie
# bliskość geometryczna = podobieństwo znaczeniowe.
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_DEVICE = "cuda"          # mamy RTX 5070 Ti; "cpu" jako fallback
EMBED_BATCH_SIZE = 32
EMBED_NORMALIZE = True         # znormalizowane wektory -> cosine == dot

# --- Qdrant (etap 2: indeks) — self-hosted w Dockerze ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# Nazwa kolekcji opisowa: korpus + model, żeby dało się trzymać kilka wariantów obok siebie.
QDRANT_COLLECTION = "osw_reports_bge_m3"
QDRANT_DISTANCE = "cosine"     # patrz EMBED_NORMALIZE; standard dla tekstu

# --- Retrieval (etap 3) ---
TOP_K = 5                      # ile chunków pobieramy do promptu (dobierzemy pomiarem w etapie 5)

# --- Generacja: lokalny LLM przez Ollama (etap 3) ---
# Qwen2.5-7B-Instruct, kwantyzacja Q4 (~4,7 GB VRAM) — mieści się w 12 GB obok BGE-M3.
LLM_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0          # niska temperatura -> mniej "kreatywności"/zmyślania (ważne w RAG)
LLM_NUM_CTX = 4096             # okno kontekstu; musi pomieścić prompt + pobrane chunki

# --- Reranking (etap 5 / checkpoint #6) ---
# Cross-encoder wielojęzyczny (naturalna para do BGE-M3). Ocenia (pytanie, chunk) RAZEM.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20         # ile kandydatów z etapu 1 trafia do rerankera = SUFIT jakości (recall@N)
