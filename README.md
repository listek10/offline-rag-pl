# offline-rag-pl — offline'owy RAG na polskich raportach analitycznych

[![testy](https://github.com/listek10/offline-rag-pl/actions/workflows/tests.yml/badge.svg)](https://github.com/listek10/offline-rag-pl/actions/workflows/tests.yml)

*Badge pokazuje testy kontraktowe (obecnie 82) na czystej maszynie bez GPU. Jakość odpowiedzi mierzy
gold set lokalnie — patrz „Uruchomienie", punkt A vs B.*

System **Retrieval-Augmented Generation** działający **w pełni offline**: odpowiada na pytania
na podstawie polskich raportów analitycznych (OSW/PISM), **cytuje źródło** i ma **zmierzoną
jakość** (gold set + metryki + poprawa „przed/po"). Projekt portfolio pod stanowisko AI/LLMOps.

> „RAG na źródłach otwartych, ale architektura jest identyczna dla korpusu niejawnego — całość
> offline, model open-source serwowany lokalnie, dane nie opuszczają systemu. Jakość mam
> zmierzoną: gold set, recall@k, i poprawę po zmianie strategii."

---

## Architektura

```mermaid
flowchart LR
  subgraph Ingest["Ingest (offline)"]
    PDF["PDF OSW/PISM"] --> P["parse_pdf.py<br/>czyszczenie: kolumny, stopki,<br/>miękki dywiz U+00AD"]
    P --> C["chunk.py<br/>cięcie po zdaniach + overlap"]
  end
  C --> E["BGE-M3<br/>embeddingi 1024D"]
  E --> Q[("Qdrant<br/>cosine + HNSW")]
  C --> B["BM25<br/>indeks leksykalny"]

  subgraph Query["Zapytanie (offline)"]
    U["pytanie"] --> D["dense: BGE-M3 + Qdrant"]
    U --> BM["BM25"]
    D --> R["RRF fusion"]
    BM --> R
    R --> RR["cross-encoder rerank<br/>bge-reranker-v2-m3"]
    RR --> G{"bramka trafności<br/>score ≥ próg?"}
    G -- "nie" --> REF["Odmowa:<br/>„nie znajduję w źródłach”"]
    G -- "tak" --> PR["prompt + kontekst"]
    PR --> LLM["Qwen2.5-7B (Ollama)<br/>temp 0, cytowanie [n]"]
    LLM --> A["odpowiedź z cytatem"]
  end

  Q -.-> D
  B -.-> BM
```

Dwustopniowy retrieval (szybki bi-encoder → dokładny cross-encoder), hybryda gęste+leksykalne
(RRF), bramka anty-halucynacyjna i wymuszone cytowanie. Orkiestracja to cienka własna warstwa
(bez ciężkich frameworków).

---

## Wyniki (zmierzone na gold secie: 36 par, w tym 7 spoza korpusu)

### Retrieval — poprawa „przed/po" (29 pytań z korpusu)
| konfiguracja | recall@1 | recall@5 | MRR |
|---|--:|--:|--:|
| dense (baseline) | 0.45 | 0.76 | 0.62 |
| + reranking (cross-encoder) | 0.79 | 0.97 | 0.86 |
| **+ hybryda BM25 (RRF)** | **0.83** | **0.97** | **0.89** |

*Reranking podniósł recall@1 (trafny fragment był w puli, ale nie na górze); hybryda podniosła
sufit recall@10 z 0.93 do 0.97 (BM25 dorzucił trafienia leksykalne, których dense nie wyciągał).*

### Chunking — A/B trzech rozmiarów (a nie „dobrane z rozsądku")

Rozmiar chunka to najczęstsze pytanie o RAG, więc nie zostaje przy uzasadnieniu — jest zmierzony.
Trzy warianty, **stały udział zakładki 15%**, ten sam gold set i ten sam tor retrievalu
(`src/eval/ab_chunking.py`; każdy wariant w osobnej kolekcji Qdranta i osobnym procesie):

| wariant | chunków | śr. znaków | trafnych* | R@1 | R@3 | R@5 | R@10 | MRR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| *dense (sam retrieval gęsty)* | | | | | | | | |
| 600 zn. | 4001 | 524 | 1.62 | **0.655** | 0.759 | 0.828 | 0.897 | **0.726** |
| 1200 zn. (baseline) | 1509 | 1092 | 1.21 | 0.448 | 0.724 | 0.759 | 0.931 | 0.613 |
| 2000 zn. | 835 | 1888 | 1.21 | 0.517 | 0.759 | 0.931 | 0.966 | 0.676 |
| *hybrid+rerank (ścieżka produkcyjna)* | | | | | | | | |
| 600 zn. | 4001 | 524 | 1.62 | 0.793 | 0.862 | 0.862 | 0.897 | 0.833 |
| 1200 zn. (baseline) | 1509 | 1092 | 1.21 | 0.793 | 0.931 | 0.966 | 0.966 | 0.863 |
| **2000 zn.** | 835 | 1888 | 1.21 | 0.793 | **0.966** | **0.966** | **1.000** | **0.879** |

\* *„trafnych" = średnia liczba chunków zawierających cytat-dowód. Rośnie, gdy podział duplikuje
tekst w zakładce, więc część różnicy w recall może brać się z redundancji, nie z dopasowania —
kolumna jest w tabeli właśnie po to, żeby to było widać. 1200 i 2000 mają identyczne 1.21, więc
ich porównanie nie jest tym efektem zaburzone.*

**Trzy odczyty:**
1. **W ścieżce produkcyjnej 2000 zn. domyka sufit: R@10 = 1.000** — po raz pierwszy żadne pytanie
   nie wypada z top-10, w tym `g21` (pojęciowe), które wypadało nawet w hybrid+rerank.
2. **600 zn. mierzalnie szkodzi** (R@5 0.862 vs 0.966 = 3 pytania), mimo że dostało *więcej*
   „biletów" (1.62 vs 1.21) — traci `g00` i `g10`, których większe chunki znajdują. Cytaty się
   w nich mieszczą, więc to nie artefakt metryki, a brak kontekstu w krótkim fragmencie.
3. **Bez rerankingu wnioski się odwracają:** małe chunki celują lepiej wysoko (dense R@1 0.655 vs
   0.448), bo krótszy fragment ma ciaśniejszy temat, ale obniżają sufit. Dwuetapowy retrieval
   przesuwa wagę na sufit, bo **reranker przestawia, ale nie dobiera** — nie wymyśli kandydata,
   którego nie dostał.

**Dlaczego mimo tego 1200 zostaje wartością domyślną** — 2000 zn. przy `top_k=5` daje prompt
~9440 znaków ≈ **3630 tokenów**, a `LLM_NUM_CTX=4096` musi pomieścić jeszcze instrukcję
systemową, pytanie i odpowiedź: nie zmieści się. Przy wyrównanym budżecie promptu przewaga znika:

| przy ~2100-2200 tokenach promptu | recall |
|---|--:|
| 1200 zn., `top_k=5` | 0.966 |
| 2000 zn., `top_k=3` | 0.966 |

**Uczciwie o sile dowodu:** n=29, więc jedno pytanie = 3,4 pkt proc. Przewaga 2000 nad 1200 to
**1-2 pytania** — poziom szumu, nie dowód; strata 600 to 3 pytania, czyli już sygnał. Wniosek,
który się broni: *w zakresie 1200-2000 zn. rozmiar nie jest dźwignią jakości — dźwignią były
reranking i hybryda; poniżej ~600 zn. jakość spada.* Podniesienie `num_ctx` do 8192 kosztowałoby
KV-cache ~460 MB na sekwencję (56 KB/token — patrz sekcja o serwowaniu), więc to decyzja
o VRAM, nie o chunkingu.

**Jak mierzona jest prawda:** recall po `chunk_id` przestaje mieć sens, gdy zmienia się rozmiar
chunka (numeracja się przesuwa). Dlatego każde pytanie z korpusu ma w gold secie `evidence` —
krótki **dosłowny cytat** z odpowiedzią; trafiony chunk to taki, który go zawiera. Definicja jest
niezależna od podziału i **surowsza** (chunk „na temat", ale bez odpowiedzi, nie liczy się).
Na baseline kosztuje ~2 pkt MRR (0.886 → 0.863) przy niezmienionym recall@5 — te dwie definicje
dają różne liczby bezwzględne, więc nie wolno ich mieszać w jednym porównaniu.

```bash
python src/eval/ab_chunking.py                 # 3 warianty: chunking -> indeks -> ewaluacja
python src/eval/run_eval.py --truth evidence   # ewaluacja z prawdą przez cytat-dowód
```

### Jakość odpowiedzi
| metryka | wynik |
|---|--:|
| refusal — pytania spoza korpusu (gold set, n=7) | 1.00 (0.71 → 0.86 prompt → 1.00 +bramka) |
| refusal — **held-out spoza korpusu (n=17), pełne 3 guardy** | **1.00** (guard #1+#2: 0.94 — bramka nie dodaje nic) |
| false-refusal na held-oucie z korpusu (n=13), 3 guardy | **0.15** (2/13, oba przez guard #3) |
| false-refusal na pytaniach z korpusu — bramka (#2) | **0** |
| false-refusal dodane przez weryfikację (guard #3) | **2/29 (7%)** — świadomy koszt |
| faithfulness (sędzia: Gemma 2, niezależny od generatora) | 0.96 |
| answer relevance | 0.97 |

### Kalibracja progu bramki na HELD-OUCIE — i dlaczego 1.00 było optymistyczne

Próg bramki (`RELEVANCE_GATE_MIN`) był początkowo dobrany na **tym samym gold secie**, na którym
raportowałem `refusal` — to dopasowanie hiperparametru do zbioru testowego. Zbudowałem więc osobny,
**nigdy nieraportowany** zbiór `gold_set/heldout_gate.jsonl` (30 pytań), w którym pytania spoza
korpusu to w większości **hard negatives**: „na temat, ale bez odpowiedzi w naszych 5 raportach".

| top-score rerankera | zakres |
|---|--:|
| pytania z korpusu (n=13) | 0.7376 … 0.9998 |
| spoza korpusu: `daleki` (arytmetyka, wiedza ogólna, n=4) | 0.0000 … 0.0015 |
| spoza korpusu: `sąsiedni-obszar` (n=6) | 0.0014 … 0.1533 |
| spoza korpusu: **`bliski-temat` (hard negatives, n=7)** | 0.0220 … **0.8827** |

> **Rewizja zbioru (28.07): jedno pytanie było błędnie oznaczone.** `h18` („Ile wynoszą wydatki
> obronne Danii jako procent PKB?") miało etykietę `out_of_corpus`, ale korpus na nie **odpowiada** —
> `Polarny-kwartet` chunk #97: „**Dania** (…) zbyt małe wydatki obronne (5,3 mld dolarów, **tj. 1,3%
> PKB w 2021 r.**)". Po przeniesieniu do `in_corpus` maksimum hard negatives spada z 0.9806 (h18) na
> **0.8827** (h15, „liczba okrętów podwodnych Norwegii" — tej danej w korpusie faktycznie nie ma,
> sprawdzone). Liczby niżej są policzone **po** tej poprawce. Diagnoza się nie zmienia: rozkłady
> nadal się nakładają, `relevance ≠ answerability` obowiązuje. Zmieniła się jedna liczba, nie wniosek —
> i to jest właśnie powód, dla którego trzymam surowe dane w repo, a nie tylko wynik w prezentacji.

**Rozkłady się nakładają** (przerwa −0.145) → nie istnieje próg z zerem przecieków i zerem błędnych
odmów. Front kompromisu: próg 0.1 → przecieki 0.35 / FR 0 · 0.735 → 0.06 / 0 · 0.985 → 0 / **FR 0.38**.

End-to-end na held-oucie (17 pytań spoza korpusu + 13 z korpusu), **zmierzone po poprawce etykiety**:

| konfiguracja | refusal spoza korpusu | false-refusal na pytaniach z korpusu |
|---|--:|--:|
| sam prompt (guard #1) | 0.94 | 0.00 |
| + bramka 0.35 (guard #2) | 0.94 | 0.00 |
| + weryfikacja tez (guard #3) | **1.00** | **0.15** (h03, h05) |

**Bramka nie dodaje tu NIC** (0.94 → 0.94): jedyny przeciek, `h19`, ma score wyżej niż najsłabsze
pytanie z korpusu, więc żaden próg go nie usunie bez strat. Domyka go dopiero guard #3, płacąc
15% błędnych odmów na held-oucie — tu widać cały koszt tej decyzji, nie tylko jej zysk.

**Wniosek architektoniczny:** cross-encoder mierzy **trafność tematyczną**, nie **obecność
odpowiedzi** (`relevance ≠ answerability`). To sufit tego sygnału, nie zła wartość progu — błąd
powstaje w warstwie **generacji**, więc kontrolę trzeba postawić właśnie tam.

```bash
python src/eval/calibrate_gate.py --e2e   # rozkłady, przemiatanie progu, odmowa end-to-end
```

### Guard #3 — weryfikacja ugruntowania po generacji (zmierzona)
Skoro błąd powstaje w warstwie generacji, kontrolę postawiłem właśnie tam: po wygenerowaniu
odpowiedzi **inny model-sędzia (Gemma)** sprawdza KAŻDĄ tezę przeciw kontekstowi; za mało popartych
→ odmowa (`src/generate/verify.py`).
- **Wartość:** na held-oucie domyka to, czego bramka nie potrafi — refusal spoza korpusu
  **0.94 → 1.00** (przeciek `h19`, „lista ofert w przetargu HX podana jako rozstrzygnięcie",
  ma score wyższy niż najsłabsze poprawne pytanie, więc progiem go nie usuniesz).
- **Koszt (zmierzony, nie założony):** **0.15 false-refusal** na held-oucie (2/13) i **7%** (2/29) na
  gold secie — `VERIFY_MIN_SUPPORTED=1.0` wymaga poparcia WSZYSTKICH tez.
- **Guard #3 jest STOCHASTYCZNY i to widać w danych, nie tylko w zastrzeżeniu:** `h19` przeciekł
  w pomiarze z 28.07, a w powtórce został złapany; `h18` odwrotnie. Pojedynczy przebieg nie jest
  dowodem na zachowanie tego guardu wobec KONKRETNEGO pytania — tylko rozkład na całym zbiorze.
- **Residual failure mode (najważniejsze ograniczenie, zreprodukowane na żywo):** pytanie „wydatki
  obronne **Danii** jako % PKB" jest w korpusie (1,3% PKB, chunk #97), ale pipeline odpowiada
  **1,75% PKB** — liczbą **Norwegii** z chunka #98 — i **wszystkie trzy guardy to przepuszczają**.
  Poprawny fragment był w kontekście na pozycji [3] (score 0.962). Sędzia uzasadnił TAK słowami
  *„zwiększenie budżetu resortu obrony **Norwegii**"* — czyli **sam nazwał właściwy podmiot i mimo to
  zagłosował za tezą o Danii**. Wniosek: weryfikacja per-teza łapie brak poparcia, ale nie łapie
  konsekwentnie **podmiany podmiotu**, gdy w kontekście stoją obok siebie dwie liczby tego samego typu.
- **Trade-off (kontekst wywiadowczy):** fałszywa odmowa = ponowne zapytanie analityka; pewna
  halucynacja = błędna decyzja. Dlatego surowy próg (1.0) jest uzasadniony.

### Podmiana podmiotu — pomiar i WYNIK NEGATYWNY próby czwartego guardu

Najpierw trzeba było naprawić metrykę, bo istniejące liczyły tę awarię jako sukces (patrz „Znane
ograniczenia"). Powstał zbiór **7 pytań-pułapek** (`gold_set/entity_traps.jsonl`, cytaty walidowane
programowo w korpusie) na dwóch klastrach, w których wartości są rozróżnialne tylko przez podmiot:
polarny (Kanada 26 mld/**1,3%**, Dania 5,3 mld/**1,3%**, Norwegia 8,4 mld/1,75% — procent Kanady
i Danii jest identyczny) oraz bałtycki, gdzie trzy kwoty stoją w **jednym zdaniu** (Estonia 2,316 /
Łotwa 1,548 / Litwa 2,048 mld).

Pomiar **parowany** — jedna generacja, warianty guardów na tych samych odpowiedziach, 3 powtórzenia
dla stochastycznego sędziego (`src/eval/eval_traps.py`):

| wariant | poprawna | niejedn. | **PODMIANA** | odmowa |
|---|--:|--:|--:|--:|
| bez guardów post-generacyjnych | 9 | 3 | **3** | 6 |
| guard #3 (sędzia Gemma) | 9 | 3 | **0** | 9 |
| guard #4 — przypisanie wielkości (kandydat) | 9 | 3 | **3** | 6 |
| guard #3 + #4 | 9 | 3 | **0** | 9 |

**Wniosek: guardu #4 NIE włączyłem.** Guard #3 domyka podmianę 3/3 przy zerowym koszcie, a kandydat
abstynuje wszędzie — zero zysku. Zostaje w repo (`src/generate/attribution.py`, 11 testów na
nieskróconych fragmentach korpusu) jako udokumentowana próba: **negatywny wynik z liczbami jest
wart więcej niż guard dodany na wyczucie.**

Dwie rzeczy, których nauczyła ta próba, ważniejsze niż sam guard:
1. **Reguła „najbliższa wcześniejsza wzmianka" w obrębie chunka jest aktywnie szkodliwa.**
   Przypisywała 1,75% PKB *Rosji*, bo dwa zdania przed liczbą stoi „…Globus III w Vardø przy
   granicy z **Rosją**" — i blokowała **poprawną** odpowiedź o Norwegii. W raportach
   geopolitycznych przypadkowych nazw państw jest pełno.
2. **Testy na fragmencie „prawdziwym, ale skróconym" są fałszywym dowodem.** Mój skrót usunął
   właśnie zdanie o Rosji, więc testy były zielone na łatwiejszej rzeczywistości niż prawdziwa;
   błąd wyszedł dopiero w pomiarze end-to-end. Testy przepisane na fragmenty nieskrócone.

---

## Stack
- **Python 3.11**, orkiestracja = cienka własna warstwa (bez LangChain)
- **Ollama** + **Qwen2.5-7B** (kwantyzacja Q4, GPU) — LLM serwowany lokalnie (domyślny backend)
- **vLLM 0.26** + **Qwen2.5-7B-Instruct-AWQ** — alternatywny backend pod przepustowość
  (×3,4 tok/s przy 4+ równoległych żądaniach; patrz sekcja „Serwowanie LLM")
- **BGE-M3** (sentence-transformers) — embeddingi wielojęzyczne (1024D)
- **Qdrant** (Docker) — baza wektorowa (metryka cosine, indeks HNSW)
- **bge-reranker-v2-m3** — cross-encoder do rerankingu
- **BM25** (własna implementacja Okapi) + **RRF** — hybryda leksykalno-semantyczna
- **Gemma 2 9B** — niezależny model-sędzia do faithfulness (mniejsza cyrkularność)

---

## Uruchomienie

Zależności są rozbite na dwa pliki, bo odpowiadają na dwa różne pytania:
**„czy repo jest spójne?"** (sekundy, bez GPU) i **„czy system działa?"** (GPU + usługi).

### A. Weryfikacja kontraktów — bez GPU, bez Qdranta, bez Ollamy

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # 4 paczki bezpośrednie, 61 MB, ~5 s (bez cache)
pytest                                   # -> 82 passed in 0,6 s
```

Działa, bo ciężkie importy (`sentence_transformers`, `qdrant_client`, `CrossEncoder`) są
**leniwe** — siedzą w środku funkcji jako singletony, nie na poziomie modułu — a retrieval i LLM
w testach są podmienione atrapami. Zmierzone w czystym venvie: `import torch` →
`ModuleNotFoundError`, a mimo to **82 passed**. Bez `node` (opcjonalna zależność dev dla
`tests/check_app_js.js`): **81 passed, 1 skipped**. Pierwszy przebieg CI: **82 passed w 2,13 s**.

**Co to udowadnia, a czego nie.** `pytest` dowodzi *kontraktów*: kształtu odpowiedzi API,
kolejności zdarzeń, chunkingu, metryk, BM25/RRF, routingu backendów. Nie dowodzi, że RAG
odpowiada dobrze — ani nawet że w ogóle wstaje. Testy podstawiają `hybrid_search` atrapą
zwracającą ręcznie napisany słownik, więc **dryf kluczy między atrapą a prawdziwą funkcją
przeszedłby niezauważony**. Tym, co pilnuje całej ścieżki, jest ewaluacja na gold secie
(punkt B.6): prawdziwy retrieval, prawdziwy model, liczby.

### B. Pełny system

```bash
# 1. Środowisko
conda create -n offline-rag-pl python=3.11 && conda activate offline-rag-pl

# 2. Zależności — KOLEJNOŚĆ MA ZNACZENIE (patrz ramka niżej)
pip install torch --index-url https://download.pytorch.org/whl/cu128   # najpierw torch z indeksu CUDA
pip install -r requirements.txt                                        # potem reszta

# 3. Baza wektorowa (Qdrant w Dockerze, dane w named volume)
docker run -d --name qdrant-osw -p 6333:6333 -v qdrant_osw_storage:/qdrant/storage qdrant/qdrant

# 4. Modele LLM (raz — wymaga internetu; potem offline)
ollama pull qwen2.5:7b        # generator
ollama pull gemma2:9b         # sędzia do ewaluacji faithfulness

# 5. Pipeline: PDF -> tekst -> chunki -> indeks -> odpowiedź
python src/ingest/parse_pdf.py data/raw/<raport>.pdf data/processed/
python src/ingest/chunk.py
python src/index/build_index.py
python src/generate/answer.py "Które państwa tworzą arktyczny kwartet NATO?"

# 6. Ewaluacja
python src/eval/run_eval.py       # retrieval: recall@k, MRR (dense vs hybryda vs +rerank)
python src/eval/eval_answers.py   # odpowiedzi: refusal, faithfulness, relevance
```
> **Offline w runtime:** po pobraniu modeli ustaw `HF_HUB_OFFLINE=1` — całość działa bez sieci.

> **Dlaczego torch nie jest w `requirements.txt`, a instalacja i tak wymaga kolejności.**
> `sentence-transformers` **wymaga** torcha, więc pip ściągnie go tranzytywnie — tylko że z PyPI,
> a ten build nie ma kerneli pod RTX 5070 Ti (Blackwell, `sm_120`). Awaria jest podstępna:
> instalacja przechodzi, a `no kernel image is available for execution` wybucha dopiero przy
> pierwszym `.encode()`. Instalując torcha **pierwszy** z indeksu `cu128`, sprawiamy, że pip
> widzi wymaganie jako spełnione i go nie rusza — sprawdzone `pip install --dry-run` w gotowym
> środowisku: *„by doinstalował: NIC"*. Odrzucona alternatywa: `--extra-index-url` w pliku —
> ta flaga nie jest przypisana do paczki, obowiązuje cały resolving (pip mógłby wziąć z indeksu
> PyTorcha dowolną inną zależność), a do tego przybiłaby plik do jednej architektury GPU.
> **vLLM (etap 7) nie należy do żadnego z tych plików** — ciągnie własnego torcha `cu130`,
> który zdegradowałby BGE-M3, więc żyje w osobnym env conda (sekcja „Serwowanie LLM").

**Zweryfikowane wersje** (2026-07-29, Python 3.11): `torch 2.11.0+cu128`,
`sentence-transformers 5.6.1`, `transformers 5.14.1`, `qdrant-client 1.18.0`, `pymupdf 1.28.0`,
`fastapi 0.140.13`, `uvicorn 0.51.0`, `numpy 2.4.6` — dev: `pytest 9.1.1`, `httpx 0.28.1`.
W plikach są **dolne granice** (`>=`), nie sztywne piny: chcemy dostawać łatki bezpieczeństwa,
a nie odmrażać ręcznie 20 wersji po pół roku. Cena tej decyzji jest realna i już widoczna —
`starlette >= 1.3` ostrzega, że `httpx` w `TestClient` jest **deprecated** (docelowo `httpx2`),
czyli kiedyś sam upgrade tranzytywny wywali `tests/test_api.py`. Wtedy: podmiana na `httpx2`.

---

## UI — lokalny panel analityka

```bash
python src/serve/api.py                    # potem: http://127.0.0.1:8000
SERVE_PORT=8100 python src/serve/api.py    # gdy 8000 jest zajęte
```

Jeden ekran: po lewej odpowiedź z **klikalnymi cytatami** `[n]`, po prawej **panel diagnostyczny**
— etapy pipeline'u z czasami, wynik bramki (`score` vs próg), werdykty sędziego per teza i źródła
z pełnym tekstem chunka. Przełączniki `bramka` / `weryfikacja` / `top_k` pozwalają pokazać
**„przed/po" na żywo**: to samo pytanie z rozbrojonym guardem przecieka halucynacją.

`/api/health` sprawdza **generator i sędziego OSOBNO**, bo to dwie różne usługi: generator idzie
za `LLM_BACKEND` (Ollama albo vLLM), a sędzia guarda #3 stoi zawsze w Ollamie. Zlanie ich w jedną
kropkę „Ollama" znaczyłoby, że przy `LLM_BACKEND=vllm` panel świeci zielono, choć generator leży.
Brak sędziego **nie blokuje** przycisku — bez guarda #3 da się pytać; brak generatora blokuje.

### Zmierzone czasy etapów (RTX 5070 Ti, Qwen2.5-7B Q4 + Gemma 2 9B jako sędzia)

| scenariusz | retrieval | generacja | weryfikacja | **total** | źródła na ekranie |
|---|--:|--:|--:|--:|--:|
| pytanie z korpusu | 923 ms | 3665 ms | 9049 ms | **13,6 s** | **+0,98 s** |
| pytanie odcięte bramką | 648 ms | nie uruchomiono | nie uruchomiono | **0,65 s** | +0,71 s |
| pytanie trudne (dłuższa odpowiedź) | 683 ms | 6268 ms | 13665 ms | **20,6 s** | +0,74 s |

Trzy rzeczy, które te liczby pokazują:
- **Guard #3 kosztuje 2,2-2,5× tyle, co sama generacja** — weryfikacja to osobne wywołanie sędziego
  na każdą tezę. To najdroższa warstwa w systemie i teraz mam to policzone, nie oszacowane.
- **Bramka jest też optymalizacją opóźnienia:** odmowa w 0,65 s zamiast 13,6 s, bo LLM nie jest
  w ogóle pytany.
- **Źródła są na ekranie po ~1 s, odpowiedź po 13-20 s.** Dlatego wysyłamy je w zdarzeniu
  `retrieval`, a nie razem z odpowiedzią — użytkownik czyta fragmenty, gdy model jeszcze pracuje.

### Decyzje projektowe warte odnotowania

- **FastAPI + statyczny HTML/vanilla JS** — dwie zależności, bez npm i build-stepu (Streamlit
  wciąga ~30 paczek tranzytywnych i przeładowuje skrypt przy każdej interakcji, co grozi
  ładowaniem BGE-M3 za każdym kliknięciem).
- **`POST` + NDJSON, nie SSE** — `EventSource` nie obsługuje POST, więc pytanie musiałoby jechać
  w query stringu, a stamtąd do logów dostępowych i historii przeglądarki.
- **Nie strumieniujemy tokenów odpowiedzi, tylko etapy** — guard #3 ocenia odpowiedź *po*
  generacji i może ją odrzucić; streaming pokazałby halucynację przed werdyktem sędziego.
- **Modele ładowane raz, przy starcie** (rozgrzewka ~10 s) — nie przy pierwszym pytaniu.
- **Trzy źródła odmowy rozróżnione w UI** — guard #1 (model sam), #2 (bramka), #3 (weryfikacja)
  znaczą różne rzeczy; zlanie ich w jedno „nie znajduję odpowiedzi" gubi informację, czy szukać dalej.
- **Wyjście LLM-a traktowane jak tekst niezaufany** — wstawiane przez `textContent`/węzły DOM,
  nigdy `innerHTML` (w kontekście bywają znaczniki z PDF-a).
- **Kształt trafienia w jednym miejscu (`src/retrieve/hit.py`), a atrapy testowe budowane tą samą
  fabryką** — bo testy podmieniają retrieval, więc ręcznie przepisany słownik pozwoliłby na dryf
  kontraktu przy zielonym suicie. Dryf byłby CICHY i kosztowny: bramka czyta `rerank_score`
  z domyślną zerówką, czyli utrata tego pola nie daje `KeyError`, tylko `top_score = 0.0` →
  odmowa na **każde** pytanie. Argumenty fabryki są tylko po nazwie, a atrapa `hybrid_search`
  powstaje przez `create_autospec` z prawdziwej funkcji — zmiana kontraktu daje `TypeError`
  w teście, nie ciszę. Sprawdzone eksperymentalnie: oba dryfy wprowadzone celowo → suite czerwony.

---

## Serwowanie LLM: Ollama vs vLLM (zmierzone)

Brief zakładał „Ollama na start, vLLM docelowo". Backend jest **jedną zmienną**
(`config.LLM_BACKEND`, nadpisywalną środowiskiem), a nie zmianą w pętli RAG — `src/generate/llm.py`
to cienki dyspozytor na `urllib` (bez biblioteki `openai`, bez klienta Ollamy).

```bash
# vLLM żyje w OSOBNYM env conda: przyciąga własny torch (2.11+cu130), a stack retrievalowy
# stoi na cu128 zbudowanym pod Blackwella (sm_120) — wspólna instalacja zdegradowałaby BGE-M3.
conda create -n vllm-server python=3.11 && conda activate vllm-server && pip install vllm==0.26.0
python src/serve/vllm_serve.py                 # parametry z configu; HF_HUB_OFFLINE=1

python src/eval/bench_llm.py --prepare                                # prompty raz, na wolnej karcie
LLM_BACKEND=vllm python src/eval/bench_llm.py --label vllm-awq-util070
python src/eval/bench_llm.py --compare
```

### DOWÓD — przepustowość (Qwen2.5-7B, prompty RAG ~2 400 tokenów, 256 tok. wyjścia, temp 0)

| równoległość | Ollama Q4 [tok/s] | vLLM AWQ [tok/s] | zmiana | p95 Ollama [s] | p95 vLLM [s] |
|---|---|---|---|---|---|
| 1 | 29,7 | 36,9 | ×1,24 | 3,74 | 2,16 |
| 4 | 38,3 | 128,2 | **×3,35** | 9,38 | 1,14 |
| 8 | 37,6 | 127,3 | **×3,39** | 12,39 | 1,29 |

Ollama saturuje się na ~38 tok/s (z 4 na 8 klientów zysk zerowy, p95 rośnie o trzecią część) —
vLLM utrzymuje ~128 tok/s przy p95 **8× niższym**. Mechanizm: dekodowanie jest ograniczone
przepustowością pamięci (na każdy token GPU czyta całe wagi), więc continuous batching czyta wagi
raz dla wielu żądań. Przy równoległości 1 przewaga jest niewielka (×1,24) — i to jest oczekiwane:
vLLM optymalizuje przepustowość serwera, nie czas jednego zapytania.

### Cena: prealokacja VRAM (PagedAttention) — i realne ograniczenie tej maszyny

vLLM zajmuje swój budżet na starcie i trzyma do końca procesu. Rozkład z logu przy `--util 0.7`
na 12 GB (widoczne 11,5 GiB):

```
wagi AWQ 5,29 GiB + szczyt aktywacji 1,04 + non-torch 0,11 + grafy CUDA 0,38 = 6,82 GiB
pula KV-cache: 1,61 GiB = 30 048 tokenów -> MAKS. RÓWNOLEGŁOŚĆ 7,34x przy oknie 4 096
```
1,61 GiB / 30 048 tokenów = **56 KB/token**, dokładnie tyle, ile daje arytmetyka Qwena 2.5-7B
(28 warstw × 4 głowy KV z GQA × 128 × 2 (K,V) × 2 B) — stąd wiadomo, ile sekwencji się zmieści,
zanim cokolwiek uruchomimy.

- Zejście `--util` 0,70 → 0,62 **nic nie zmieniło** (135–139 vs 127–128 tok/s): pula KV była i tak
  większa, niż potrzebują 8 żądania. Poniżej ~0,62 serwer nie wstaje — sam narzut to 6,82 GiB.
- **vLLM i stack retrievalowy nie mieszczą się razem na 12 GB.** Sprawdzone: przy działającym vLLM
  serwer UI dostaje CUDA OOM przy ładowaniu cross-encodera, a zapytanie pada na
  `CUBLAS_STATUS_ALLOC_FAILED`. Ollama tego problemu nie ma, bo ładuje leniwie i **zwalnia** kartę
  po bezczynności (`keep_alive`).
- **Wniosek architektoniczny:** na tym sprzęcie Ollama zostaje pod interaktywne UI (grzecznie dzieli
  GPU z BGE-M3), a vLLM ma sens tam, gdzie liczy się przepustowość i można rozdzielić etapy w
  czasie — czyli w ewaluacji wsadowej (prompty przygotowane wcześniej, potem sam LLM). Na
  dedykowanym GPU produkcyjnym ten kompromis znika.

### DOWÓD — jakość po zmianie kwantyzacji Q4_K_M → AWQ

Zmiana backendu zmienia DWIE rzeczy: silnik i kwantyzację (inne zaokrąglenia wag = inny model
numerycznie). Szybszy silnik z gorszymi odpowiedziami byłby regresją, więc jakość zmierzono na tych
samych promptach (`src/eval/compare_backends.py`, bramka wyłączona — mierzymy sam model):

| metryka | Ollama Q4_K_M | vLLM AWQ |
|---|---|---|
| ugruntowanie tez (sędzia Gemma, n=29) | 0,943 | 0,953 |
| odpowiedzi z cytatem `[n]` | 0,966 | 0,966 |
| false-refusal na pytaniach z korpusu | 0,0 | 0,0 |
| odmowa spoza korpusu (guard #1, sam prompt) | 0,857 | **0,714** |

Wierność i cytowanie bez zmian. Jedyna regresja: AWQ przepuścił **oba** pytania arytmetyczne
(„√144", „15 × 8"), Q4_K_M jedno — ale ich score rerankera to 0,004 i 0,010, więc **bramka (0,35)
zatrzymuje je przed modelem** i odmowa end-to-end zostaje 1,00. Praktyczny argument za obronnością
warstwową: pogorszenie jednego guardu nie przebiło się na wynik pipeline'u.

### Pułapka offline warta zapamiętania

Pierwszy start vLLM padł na `Could not find nvcc` — FlashInfer **kompiluje kernel samplingu JIT-em**
przy starcie i wymaga pełnego CUDA toolkitu, którego koła pip nie zawierają. Rozwiązanie:
`VLLM_USE_FLASHINFER_SAMPLER=0` (generujemy zachłannie, `temperature=0`, więc nic nie tracimy).
Wniosek ogólniejszy dla środowiska air-gapped: kernele kompilowane na maszynie docelowej to
zależność od toolchainu, którego tam zwykle nie ma.

---

## Struktura repo
```
config/config.py        # WSZYSTKIE parametry (chunk size, top_k, modele, progi) w jednym miejscu
src/ingest/             # parse_pdf.py (czyszczenie PDF) + chunk.py (chunking)
src/index/build_index.py# embeddingi -> Qdrant
src/retrieve/           # search.py (dense), bm25.py, hybrid.py (RRF), rerank.py,
                        # hit.py (kanoniczny kształt trafienia — jedno źródło nazw pól)
src/generate/answer.py  # pętla RAG: retrieval -> bramka -> prompt -> LLM -> cytat (+ on_event dla UI)
src/generate/llm.py     # dyspozytor backendów LLM (Ollama /api/generate vs vLLM /v1/chat/completions)
src/generate/verify.py  # guard #3: weryfikacja ugruntowania tez po generacji (sędzia Gemma)
src/serve/              # api.py (3 endpointy), stream.py (mostek callback->iterator), static/ (UI)
src/serve/vllm_serve.py # start serwera vLLM z parametrami z configu (osobne env conda, offline)
src/eval/               # metrics.py, run_eval.py (retrieval), eval_answers.py (odpowiedzi),
                        # calibrate_gate.py (kalibracja progu bramki na held-oucie),
                        # bench_llm.py (przepustowość backendów), compare_backends.py (jakość),
                        # ab_chunking.py (A/B rozmiaru chunka: chunking -> indeks -> ewaluacja)
gold_set/gold.jsonl     # zbiór ewaluacyjny (pytanie -> odpowiedź -> chunk źródłowy -> `evidence`:
                        # dosłowny cytat z odpowiedzią, prawda niezależna od podziału na chunki)
gold_set/heldout_gate.jsonl # held-out do kalibracji progu — NIGDY nieraportowany jako metryka
docs/specs/, docs/plans/# spec i plan implementacji UI (decyzje + odrzucone alternatywy)
tests/                  # chunking, metryki, BM25/RRF, mostek strumienia, kontrakt API, logika app.js
```

---

## Znane ograniczenia (świadome, zmierzone)
- **Kolejność kolumn** w niektórych dwukolumnowych PDF-ach bywa mieszana — wpływ na jakość mały
  po dodaniu hybrydy (zweryfikowane pomiarem).
- **Retrieval pojęciowy** (pytania „dlaczego/jak") słabszy niż faktograficzny (recall@5 ~0.89).
- **Cytowanie per-teza** na modelu 7B bywa nieprecyzyjne (grupuje źródła).
- **Metryki LLM-sędziego** to sygnał, nie prawda absolutna (cyrkularność mimo innego modelu) —
  zalecany ręczny sanity-check; wartości zależą od sędziego (faithfulness 0.88 Qwen vs 0.96 Gemma).
- **Gold set n=29** (z korpusu) — liczby wciąż nieco szumne. Próg bramki jest już kalibrowany na
  **osobnym held-oucie**, nie na gold secie (patrz sekcja o kalibracji).
- **Residual refusal:** przed bramką prosta arytmetyka przeciekała (model liczył); bramka to domyka.
  Na kwantyzacji AWQ przecieka o jedno pytanie więcej niż na Q4_K_M — bramka nadal to domyka.
- **vLLM nie zmieści się obok retrievalu na 12 GB VRAM** (zmierzone: CUDA OOM). Domyślnym backendem
  aplikacji zostaje Ollama; vLLM jest przełączalną opcją pod obciążenia wsadowe i dedykowane GPU.
- **Hard negatives** („na temat, ale odpowiedzi nie ma") — bramka ich nie łapie (`relevance ≠
  answerability`); dołożony guard #3 (weryfikacja ugruntowania po generacji) podnosi refusal na
  held-oucie **0.94 → 1.00**, ale kosztem **0.15 false-refusal** (2/13) tam i ~7% (2/29) na gold
  secie. System nie jest tu idealny — każdą warstwę mierzę osobno.
- **Podmiana podmiotu** — na pytanie o wydatki obronne **Danii** generator podaje **1,75% PKB**,
  czyli liczbę **Norwegii** (dosłownie to samo zdanie co dla pytania o Norwegię), choć poprawne
  1,3% jest w kontekście. Zmierzone osobnym harnessem (niżej): guard #3 **łapie to 3/3** przy
  tym brzmieniu pytania, ale w demo z innym brzmieniem przepuścił, a sędzia uzasadnił TAK
  zdaniem o Norwegii. Czyli awaria jest **zależna od brzmienia pytania**, a nie domknięta —
  i to jest najpoważniejsze znane ograniczenie tej architektury.
- **Metryki jakości odpowiedzi były na tę awarię ŚLEPE** (naprawione): `faithfulness` liczyło
  podmianę jako wierną (sędzia głosował TAK), `false-refusal` jako poprawną (system odpowiedział),
  `relevance` jako na temat. Pole `gold["answer"]` leżało w danych i **nie było czytane przez
  żaden skrypt ewaluacyjny**. Dopiero `src/eval/metrics.py::classify_trap_answer` sprawdza
  deterministycznie, czy w odpowiedzi stoi WARTOŚĆ właściwego podmiotu.
- **Temperatura 0 nie daje powtarzalności** (zmierzone): to samo pytanie odpowiadało poprawnie
  3/3, po przerwie odmawiało 3/3, przy identycznym retrievalu (top_score 0,9964, te same 5
  chunków). Wykluczone pomiarem: wpływ flagi `verify` na generację, zimne przeładowanie modelu,
  kolejność żądań. Konsekwencja praktyczna: **porównania „przed/po" muszą być parowane** —
  generacja raz, warianty guardów oceniane na tej samej odpowiedzi (`eval_traps.py --prepare`
  / `--compare`), inaczej różnica miesza efekt guardu z tym szumem.
- **Formy przymiotnikowe powodują fałszywe odmowy** (zmierzone): korpus mówi „**łotewskie**
  1,548 mld", pytanie mówi „**Łotwy**" — model odmawia, choć chunk z odpowiedzią stoi na
  **pozycji 1** (rr=0,931). Retrieval bez winy. To lustrzane odbicie podmiany podmiotu: ten sam
  model raz przypisuje za dużo, raz za mało.
- **Jakość etykiet zbioru ewaluacyjnego jest częścią systemu:** jedno pytanie held-outu (`h18`) było
  błędnie oznaczone jako „spoza korpusu", co zawyżało raportowaną przerwę rozkładów (−0.243 zamiast
  −0.145). Znalazłem to dopiero na demo z prawdziwymi modelami — stąd zasada trzymania surowych
  score'ów w repo, bo bez nich taka pomyłka jest niewykrywalna.
