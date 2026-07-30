"""
Centralna konfiguracja projektu — WSZYSTKIE parametry w JEDNYM miejscu.

DLACZEGO tak: chunk size, overlap, top_k, nazwy modeli i ścieżki nie mogą być rozsiane
po kodzie. Jak rekruter zapyta "jaki masz chunk size i czemu?", chcę wskazać jedno miejsce,
a nie polować po plikach. Zmiana parametru w eksperymencie (etap 5: optymalizacja z dowodem)
ma być jednolinijkowa.
"""

import os
from pathlib import Path

# --- Ścieżki ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# Ścieżka do chunków nadpisywalna środowiskiem — pod A/B chunkingu (etap 8) każdy wariant ma
# WŁASNY plik i WŁASNĄ kolekcję, żeby nie nadpisywać jedynego indeksu i móc powtórzyć wynik:
#     CHUNKS_PATH=data/processed/chunks_c600.jsonl QDRANT_COLLECTION=osw_reports_bge_m3_c600 ...
CHUNKS_PATH = Path(os.environ.get("CHUNKS_PATH", PROCESSED_DIR / "chunks.jsonl"))
GOLD_PATH = ROOT / "gold_set" / "gold.jsonl"   # zbiór ewaluacyjny (etap 4)

# --- Chunking (etap 2) ---
# Mierzymy w ZNAKACH jako tani proxy długości. Dokładny pomiar w tokenach wymagałby
# tokenizera modelu embeddingowego — dodamy, jeśli ewaluacja pokaże, że warto.
# ~1200 znaków ≈ ~500 tokenów dla polskiego (polski jest "token-cięższy" niż angielski).
# Rozmiar i zakładka są nadpisywalne środowiskiem WYŁĄCZNIE pod eksperyment A/B (etap 8) —
# domyślne wartości zostają tutaj, więc zasada „konfiguracja w jednym miejscu" jest zachowana.
CHUNK_TARGET_CHARS = int(os.environ.get("CHUNK_TARGET_CHARS", 1200))   # docelowy rozmiar chunka
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", 180))  # ~15% zakładki — by myśl na styku dwóch chunków nie wypadła
CHUNK_MIN_CHARS = 250          # krótsze "ochłapy" doklejamy do poprzedniego, nie emitujemy osobno

# --- Reguła awaryjna dla dokumentów WYPUNKTOWANYCH (etap 14) ---
# PROBLEM (zmierzony na 100-stronicowym raporcie pisanym wypunktowaniami): tekst bez kropek
# wraca z `split_sentences` jako jedno "zdanie" długości 2867 zn., a chunk sięgał 5722 zn.
# Dwa progi, bo dziura miała dwie warstwy — szczegóły w `src/ingest/chunk.py`.
CHUNK_MAX_SENTENCE_CHARS = int(os.environ.get("CHUNK_MAX_SENTENCE_CHARS", 1800))  # 1,5 × target
# Kap zakładki: najdłuższe JEDNO zdanie, które wolno przenieść do następnego chunka.
# 600 zn. dobrane POMIAREM (siatka limit × kap): ogon ginie na obu korpusach przy koszcie
# 38 z 1509 ruszonych chunków bazowych. Kap 180 (= sam budżet zakładki) likwidowałby zakładkę
# na normalnej prozie (−9% chunków), czyli unieważniał decyzję o 15% zakładki z etapu 8.
CHUNK_CARRY_MAX_CHARS = int(os.environ.get("CHUNK_CARRY_MAX_CHARS", 600))

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
# Nadpisywalna środowiskiem — patrz CHUNKS_PATH (A/B chunkingu trzyma warianty w osobnych kolekcjach).
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "osw_reports_bge_m3")
QDRANT_DISTANCE = "cosine"     # patrz EMBED_NORMALIZE; standard dla tekstu

# --- Retrieval (etap 3) ---
TOP_K = 5                      # ile chunków pobieramy do promptu (dobierzemy pomiarem w etapie 5)

# --- Generacja: lokalny LLM przez Ollama (etap 3) ---
# Qwen2.5-7B-Instruct, kwantyzacja Q4 (~4,7 GB VRAM) — mieści się w 12 GB obok BGE-M3.
LLM_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0          # niska temperatura -> mniej "kreatywności"/zmyślania (ważne w RAG)
LLM_NUM_CTX = 4096             # okno kontekstu; musi pomieścić prompt + pobrane chunki

# --- Backend serwowania LLM (etap 7 / checkpoint #9 cz. 2) ---
# Przełącznik silnika: Ollama (dev, prosty) vs vLLM (produkcyjny throughput). Domyślnie
# "ollama", żeby zmiana była ADDYTYWNA — bez ustawienia env cały projekt działa jak dotąd.
# Nadpisywalny środowiskiem, bo benchmark odpala oba backendy pod rząd i nie może
# edytować configu między przebiegami:
#     LLM_BACKEND=vllm python src/eval/bench_llm.py
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")   # "ollama" | "vllm"

# vLLM wystawia API ZGODNE Z OPENAI (/v1/chat/completions) — wołamy je własnym urllib,
# bez biblioteki `openai` (zasada: cienka własna warstwa, mniej zależności).
# Port 8001, bo 8000/8100 zajmuje warstwa serwująca UI.
VLLM_URL = "http://localhost:8001/v1"
# GGUF (format Ollamy) w vLLM jest eksperymentalny i wolny -> kwantyzacja AWQ.
# UWAGA: AWQ != Q4_K_M numerycznie, więc po przełączeniu backendu trzeba PONOWNIE zmierzyć
# jakość na gold secie, nie tylko szybkość (zmieniamy silnik ORAZ kwantyzację).
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
# Musi równać się LLM_NUM_CTX, inaczej benchmark porównywałby różne okna kontekstu.
VLLM_MAX_MODEL_LEN = 4096
# vLLM PREALOKUJE ten ułamek CAŁEGO VRAM-u na starcie (domyślnie 0.9) i trzyma go do końca
# procesu — to cena PagedAttention (pula bloków KV-cache z góry, bez fragmentacji).
# 0.9 x 12,2 GB = 11 GB -> nie zostaje miejsca na BGE-M3 + reranker (~2,5 GB) => OOM.
# 0.7 -> ~8,5 GB budżetu; po wagach AWQ (~5,5 GB) zostaje ~3 GB puli KV, a KV Qwena 2.5-7B
# to ~56 KB/token (28 warstw x 4 głowy KV GQA x 128 x 2 x fp16) = ~229 MB na sekwencję przy
# 4096 -> ~13 sekwencji naraz. Niżej (0.5) pula spada do ~2-3 sekwencji i continuous batching
# nie ma czego batchować — czyli sam odbieram vLLM-owi jego przewagę.
VLLM_GPU_MEMORY_UTILIZATION = 0.7

# Sztywny limit tokenów wyjścia dla BENCHMARKU (nie dla produkcji, gdzie model kończy sam).
# Bez niego oba silniki generowałyby różną liczbę tokenów i porównanie tokenów/s byłoby fikcją.
BENCH_MAX_TOKENS = 256
BENCH_CONCURRENCY = (1, 4, 8)   # vLLM wygrywa przepustowością POD OBCIĄŻENIEM, nie przy 1

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
# Port z możliwością nadpisania środowiskiem: 8000 to popularny domyślny, więc realnie
# bywa zajęty przez inny projekt (u mnie już był). Domyślna wartość zostaje TUTAJ, więc
# zasada „konfiguracja w jednym miejscu" jest zachowana — env tylko nadpisuje na czas uruchomienia:
#     SERVE_PORT=8100 python src/serve/api.py
SERVE_PORT = int(os.environ.get("SERVE_PORT", 8000))
# Rozgrzewka przy starcie: jedno zapytanie na śmieci wciąga BGE-M3 i cross-encoder do VRAM,
# żeby PIERWSZE pytanie użytkownika było szybkie (w CLI ten koszt płaci się przy każdym
# uruchomieniu). Wyłączamy w testach — nie chcemy tam GPU.
SERVE_WARMUP = True

# --- Wgrywanie PDF-u ad hoc (etap 2 UI) ---
# OSOBNA kolekcja, kasowana przy starcie I przy zamknięciu procesu. Nazwa jest inna niż
# QDRANT_COLLECTION i to jest cała istota tej decyzji: `build_index.py` robi `delete_collection`
# przy każdym buildzie, więc indeksowanie obcego dokumentu do kolekcji zmierzonej skasowałoby
# ją i przebudowało razem z tym dokumentem — liczby z README przestałyby być odtwarzalne.
# Indeks, na którym mierzę, musi być ZAMROŻONY (ta sama zasada, co held-out do kalibracji progu).
UPLOAD_COLLECTION = "demo_upload"
# Limity NIE są zabezpieczeniem (serwer stoi na 127.0.0.1, użytkownik jest jeden) — chronią demo
# przed zamrożeniem: 2000-stronicowy tom to ~minuta parsowania i liczenia embeddingów.
UPLOAD_MAX_BYTES = 25 * 1024 * 1024
UPLOAD_MAX_PAGES = 300
# Wykrywanie skanu: PDF bez warstwy tekstowej to obraz — PyMuPDF zwraca z niego kilkanaście
# śmieciowych znaków na stronę (numeracja, artefakty), a nie zero. Dlatego próg nie może być
# zerem. OCR-u świadomie nie mamy w pipeline, więc taki dokument odrzucamy z wyjaśnieniem.
UPLOAD_MIN_CHARS_PER_PAGE = 100
