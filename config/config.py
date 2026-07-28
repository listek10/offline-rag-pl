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

# --- Hybryda: BM25 (leksykalne) + RRF (etap 5) ---
BM25_K1 = 1.5                  # nasycenie częstością termu w BM25 (standard)
BM25_B = 0.75                  # siła normalizacji długości dokumentu w BM25 (standard)
RRF_K = 60                     # stała Reciprocal Rank Fusion; łączymy rankingi po POZYCJI, nie po score

# --- Bramka trafności (guard #2, etap 5) ---
# Jeśli najwyższy score rerankera < progu -> kontekst nietrafny -> wymuszamy odmowę (bez pytania LLM).
# Próg SKALIBROWANY NA HELD-OUCIE (`gold_set/heldout_gate.jsonl`, nigdy nieraportowanym jako metryka),
# a nie na gold secie — inaczej dopasowywalibyśmy hiperparametr do zbioru testowego.
# Dane z kalibracji: OOC `daleki` ≤0.0015, OOC `sąsiedni-obszar` ≤0.1533, najsłabsze pytanie
# Z KORPUSU 0.7376 -> 0.35 daje ~0.2 marginesu w obie strony przy false-refusal = 0.
# ŚWIADOME OGRANICZENIE: hard negatives („na temat, ale odpowiedzi nie ma") mają score do 0.98,
# więc ta bramka ich NIE łapie — to sufit sygnału (relevance != answerability). Stąd guard #3 niżej.
RELEVANCE_GATE_MIN = 0.35

# --- Weryfikacja ugruntowania po generacji (guard #3, etap 5) ---
# Sprawdzamy KAŻDĄ tezę odpowiedzi przeciw kontekstowi modelem-sędzią; jeśli udział tez popartych
# < progu -> odmawiamy zamiast publikować. Atakuje failure mode niewidoczny dla bramki: błędną
# atrybucję podmiotu (np. liczba Norwegii przypisana Danii) i zamianę oferty/planu na fakt.
VERIFY_GROUNDING = True
VERIFY_MIN_SUPPORTED = 1.0     # 1.0 = wszystkie tezy muszą być poparte (kalibrowane na held-oucie)

# --- Sędzia faithfulness (etap 5) ---
# INNY model niż generator (Qwen) -> mniej skorelowane błędy = mniejsza cyrkularność oceny.
JUDGE_MODEL = "gemma2:9b"

# --- Warstwa serwująca / UI (etap 6) ---
# 127.0.0.1, nie 0.0.0.0: aplikacja jest lokalna i nie ma się wystawiać w sieci.
SERVE_HOST = "127.0.0.1"
SERVE_PORT = 8000
# Rozgrzewka przy starcie: jedno zapytanie na śmieci wciąga BGE-M3 i cross-encoder do VRAM,
# żeby PIERWSZE pytanie użytkownika było szybkie (w CLI ten koszt płaci się przy każdym
# uruchomieniu). Wyłączamy w testach — nie chcemy tam GPU.
SERVE_WARMUP = True
