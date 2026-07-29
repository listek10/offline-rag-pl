# Spec: UI dla offline-rag-pl (panel analityka + diagnostyka)

Data: 2026-07-28 · Branch: `ui-panel` (worktree `/home/maciek/Pulpit/offline-rag-pl-ui`)
Baza: `91f0ccf` (`main` po merge PR #7 — trzy guardy w komplecie)

---

## 1. Problem i cel

System RAG działa end-to-end i ma zmierzoną jakość, ale jedynym interfejsem jest CLI
(`python src/generate/answer.py "pytanie"`). To ogranicza projekt na dwa sposoby:

1. **Nie da się go pokazać.** Cała praca nad trzema guardami, kalibracją progu na held-oucie
   i dowodami „przed/po" jest niewidoczna dla kogokolwiek, kto nie czyta kodu.
2. **Nie da się go używać.** Analityk nie ma jak jednym kliknięciem sprawdzić, czy cytowany
   fragment naprawdę mówi to, co model twierdzi.

**Cel:** jeden ekran, na którym widać jednocześnie **odpowiedź** (widok analityka) i **jak system
do niej doszedł** (panel diagnostyczny). Diagnostyka jest domyślnie widoczna — w projekcie
portfolio niewidoczna praca to praca, która się nie liczy.

**Czym to UI NIE jest:** zamiennikiem CLI. Ingest, ewaluacja i kalibracja zostają w terminalu,
gdzie widać, co się dzieje. UI służy do zadawania pytań i oglądania, jak system myśli.

---

## 2. Decyzje architektoniczne (z odrzuconymi alternatywami)

### D1. FastAPI + statyczny HTML/CSS/vanilla JS

**Wybrane:** dwie nowe zależności (`fastapi`, `uvicorn`), jeden statyczny plik HTML, zero npm,
zero build-stepu. Prawdziwa granica HTTP API (`POST /api/ask`) i długo żyjący proces.

**Odrzucone — Streamlit:** ~30 zależności tranzytywnych (tornado, pandas, pyarrow, protobuf) oraz
model wykonania polegający na **przeładowaniu całego skryptu przy każdej interakcji** — bez
`@st.cache_resource` BGE-M3 i reranker ładowałyby się od nowa przy każdym kliknięciu. Łamie zasadę
z CLAUDE.md („mniej zależności = lepiej pod środowisko o wysokich wymaganiach bezpieczeństwa").

**Odrzucone — samo `http.server` (stdlib):** zero nowych zależności, ale routing i walidację JSON
trzeba napisać ręcznie, bez OpenAPI. Gorsza infrastruktura własnej roboty.

**Zysk niezależny od wyglądu:** proces serwera ładuje modele **raz przy starcie**, a nie przy każdym
zapytaniu jak CLI.

### D2. `POST` + NDJSON, a nie SSE

`EventSource` (natywne SSE w przeglądarce) **nie obsługuje POST** — wymusiłby przekazanie pytania
w query stringu, czyli **treść zapytania w logach dostępowych uvicorna i w historii przeglądarki**.
Dla systemu, który ma być architekturą pod korpus niejawny, to dyskwalifikuje.

Cena: ~15 linii ręcznego parsera NDJSON w JS zamiast 5 linii `EventSource`. Świadomy wybór.

### D3. NIE strumieniujemy tokenów odpowiedzi — strumieniujemy etapy

Guard #3 ocenia odpowiedź **po jej wygenerowaniu** i potrafi ją odrzucić. Strumieniowanie tokenów
pokazałoby użytkownikowi halucynację, zanim sędzia wyda werdykt — czyli zniszczyłoby cel guardu.
Strumieniujemy więc **postęp etapów pipeline'u** (retrieval → bramka → generacja → weryfikacja),
z czasami. Użytkownik nie patrzy w martwy ekran, a odpowiedź pojawia się dopiero po przejściu guardów.

### D4. Mostek callback → iterator (wątek + kolejka)

`answer()` jest synchroniczna i blokująca (urllib + torch), a zdarzenia potrzebne są **ze środka**
jej wykonania.

**Wybrane:** `answer(..., on_event=cb)`; `answer()` leci w wątku roboczym, callback wrzuca słowniki
do `queue.Queue`, generator odsysa kolejkę i yielduje linie NDJSON aż do sentinela.
Sedno: `StreamingResponse` musi z czegoś **ciągnąć** (iterator), a callback **wypycha**; ktoś musi
blokować po drugiej stronie kolejki. Gdyby `answer()` leciała w tym samym wątku co generator, nic
nie dałoby się wyemitować, dopóki nie skończy — czyli dokładnie to, czego unikamy.

**Odrzucone — przepisanie pipeline'u w warstwie serwującej** (wołanie `hybrid_search`, bramki,
`call_ollama`, `grounded_fraction` wprost w `api.py`): powstałyby **dwie definicje pipeline'u**,
które się rozjadą. Gorzej — UI pokazywałoby zachowanie inne niż mierzy `eval_answers.py`, i liczby
przestałyby opisywać to, co widzi użytkownik. CLAUDE.md zabrania tego wprost.

**Odrzucone — przepisanie na async** (urllib→httpx, torch i tak do executora): duża zmiana, zero
zysku dla lokalnej aplikacji jednoosobowej.

### D5. Rozgrzewka przy starcie, nie leniwe ładowanie

W `lifespan` FastAPI: jedno zapytanie na śmieci (`hybrid_search("rozgrzewka", rerank=True)`) wciąga
BGE-M3 i reranker do VRAM, plus ping do Ollamy. Serwer wstaje ~15–30 s, ale **pierwsze pytanie
użytkownika jest szybkie** — a to jest pytanie, które oglądać będzie rekruter. Health zwraca
`warm:false`, UI trzyma przycisk zablokowany z komunikatem „rozgrzewanie modeli…".

---

## 3. Kontrakt API

### `GET /` → `static/index.html`

### `GET /api/health`

```json
{ "qdrant": {"ok": true, "collection": "osw_reports_bge_m3", "points": 1509},
  "ollama": {"ok": true, "models": ["qwen2.5:7b", "gemma2:9b"]},
  "config": {"top_k": 5, "gate_min": 0.35, "verify_min_supported": 1.0,
             "llm": "qwen2.5:7b", "judge": "gemma2:9b", "embed": "BAAI/bge-m3"},
  "warm": true }
```

Progi i nazwy modeli **czytane z `config.py`**, nigdy wpisane w HTML — inaczej UI kłamałoby po
zmianie progu. Konfiguracja jedzie razem z healthem (jeden strzał przy załadowaniu strony), żeby
nie mnożyć endpointów.

### `POST /api/ask` → `application/x-ndjson` (strumień)

Wejście (walidacja Pydantic → złe wejście daje 422 bez naszego kodu):

```json
{"question": "…", "top_k": 5, "gate": true, "verify": true}
```

- `question`: 1–500 znaków po `strip()`
- `top_k`: 1–20
- `gate`, `verify`: bool

Wyjście — jedna linia = jedno zdarzenie JSON:

```
{"event":"stage","stage":"retrieval","status":"start"}
{"event":"stage","stage":"retrieval","status":"done","ms":412,"top_score":0.87,
 "hits":[{"n":1,"source":"Polarny-kwartet","chunk_index":37,"score":0.87,"text":"…"}]}
{"event":"stage","stage":"gate","status":"done","passed":true,"top_score":0.87,"threshold":0.35}
{"event":"stage","stage":"generation","status":"start"}
{"event":"stage","stage":"generation","status":"done","ms":8100}
{"event":"stage","stage":"verify","status":"done","ms":11200,"grounded_fraction":1.0,
 "verdicts":[{"claim":"…","grounded":true,"reason":"…"}]}
{"event":"final","answer":"…","blocked_by":null,"self_refused":false,
 "rejected_answer":null,"grounding":[…],"total_ms":19700}
```

`self_refused` jest osobnym polem, bo guard #1 **nie ustawia `blocked_by`** — model odmawia sam,
z instrukcji w promptcie, i `answer()` zwraca zwykły tekst. Rozpoznajemy to helperem `refused()`
(patrz tabela trzech źródeł odmowy niżej).

Zdarzenie błędu (zamiast `final`):

```
{"event":"error","where":"ollama","message":"Nie mogę połączyć się z Ollamą (localhost:11434). Uruchom: ollama serve"}
```

**Dwie decyzje w kontrakcie:**

1. **Źródła jadą w zdarzeniu `retrieval`, nie w `final`.** UI renderuje panel źródeł po ~400 ms
   i użytkownik czyta fragmenty, gdy model jeszcze generuje. Zamiast 8 sekund spinnera — materiał
   do czytania. To jest cel strumieniowania etapów.
2. **Wyłączony guard emituje `status:"skipped"`**, nie znika po cichu. Na ekranie musi być widać,
   że demo leci z rozbrojonym zabezpieczeniem.

### Trzy źródła odmowy — UI musi je rozróżniać

Odmowa może przyjść z trzech różnych warstw i **`blocked_by` nie wystarcza**, bo guard #1 go nie
ustawia. `answer()` zwraca:

| źródło | co widać w wyniku | co pokazuje UI |
|---|---|---|
| **guard #1** (instrukcja w promptcie — model sam odmówił) | `answer` = tekst odmowy, `blocked_by` **brak** | „model sam odmówił — nie znalazł odpowiedzi w kontekście (guard #1: prompt)" |
| **guard #2** (bramka trafności) | `blocked_by:"gate"`, brak `grounding` | „bramka trafności: top-score 0.21 < próg 0.35 — LLM nie był w ogóle pytany" |
| **guard #3** (weryfikacja ugruntowania) | `blocked_by:"verify"` + `rejected_answer` + `grounding` | odrzucona odpowiedź + niepoparte tezy z uzasadnieniem sędziego |

Rozróżnienie jest istotne, bo znaczą **zupełnie różne rzeczy**: guard #2 mówi „w korpusie nie ma
nic na ten temat", guard #1 „temat jest, ale model nie widzi odpowiedzi", guard #3 „odpowiedź
powstała, ale nie trzyma się źródeł". Zlanie ich w jedno „nie znajduję odpowiedzi" jest tym samym
błędem, którego uniknięcie było celem panelu diagnostycznego. Detekcja guardu #1 idzie przez
istniejący helper `refused()` z `answer.py` — jedno miejsce na tę regułę, żeby UI i harness
ewaluacyjny nie rozjechały się w definicji odmowy.

**Etapy nieuruchomione:** gdy bramka zablokuje, generacja i weryfikacja **nigdy się nie wykonają**.
Emitujemy dla nich `status:"not_run"`, żeby panel przebiegu pokazał je jako pominięte, a nie wisiał
w nieskończonym „w toku". `status` przyjmuje więc: `start` · `done` · `skipped` (guard wyłączony
przełącznikiem) · `not_run` (wcześniejszy guard przerwał pipeline).

### Zmiana w `src/generate/answer.py` — addytywna

```python
def answer(question, top_k=None, system=SYSTEM_PROMPT, gate=True, verify=None,
           on_event=None):                      # +1 kwarg, domyślnie None
    emit = on_event or (lambda **_: None)       # brak callbacka -> no-op
```

Plus ~8 jednolinijkowych `emit(...)` na granicach etapów. **Zero zmian w logice.** CLI i harness
ewaluacyjny działają bez zmian, bo domyślnie `on_event=None`.

---

## 4. Ekran

```
┌───────────────────────────────────────────────────────────────────────┐
│  offline-rag-pl              ● Qdrant   ● Ollama   1509 chunków       │
├───────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐  ┌────────────┐  │
│  │ Które państwa tworzą arktyczny kwartet NATO?    │  │  Zapytaj   │  │
│  └─────────────────────────────────────────────────┘  └────────────┘  │
│   bramka [✓]    weryfikacja [✓]    top_k [5]                          │
├─────────────────────────────────────────┬─────────────────────────────┤
│  ODPOWIEDŹ                              │  PRZEBIEG                   │
│                                         │  ✓ retrieval        412 ms  │
│  Arktyczny kwartet NATO tworzą Norwegia,│  ✓ bramka   0.87 ≥ 0.35     │
│  Dania, Islandia i Kanada [1], a od     │  ✓ generacja        8.1 s   │
│  2023 r. dołączyły do nich [2].         │  ✓ weryfikacja     11.2 s   │
│                     ↑ klikalne          │      3/3 tezy poparte       │
│                                         ├─────────────────────────────┤
│  ─── albo, przy odmowie: ───            │  ŹRÓDŁA                     │
│                                         │  [1] Polarny-kwartet #37    │
│  ⛔ Nie znajduję odpowiedzi w źródłach.  │      score 0.87       ▸     │
│     Zablokowane przez: weryfikacja      │  [2] Formaty-wspolpracy #12 │
│     Odrzucona odpowiedź:  ▸ pokaż       │      score 0.71       ▸     │
│     Niepoparta teza: „Dania 1,75% PKB"  │  [3] …                      │
│       sędzia: w źródle ta liczba        │                             │
│       dotyczy Norwegii, nie Danii       │                             │
└─────────────────────────────────────────┴─────────────────────────────┘
```

Jeden ekran, bez zakładek i nawigacji. Na wąskim ekranie prawa kolumna schodzi pod odpowiedź.

### Zakres — siedem funkcji

1. **Pole pytania** — Enter wysyła; przycisk blokowany na czas pracy (zapytanie ~20 s, trzeba
   uniemożliwić podwójne wysłanie).
2. **Odpowiedź z klikalnymi cytatami** — `[1]` w tekście podświetla i rozwija fragment `[1]`
   w panelu źródeł. Sedno weryfikowalności: analityk jednym kliknięciem sprawdza, czy model nie ściemnia.
3. **Panel źródeł** — dokument + `#chunk_index` + score rerankera + rozwijany **pełny tekst chunka**
   (to, co dokładnie zobaczył model, nie skrót).
4. **Panel przebiegu** — etapy z czasami i **liczbami bramki** (`0.87 ≥ 0.35`), nie samym „przeszło".
5. **Ekran odmowy, który tłumaczy się z siebie** — który guard, dlaczego, a przy guardzie #3
   dodatkowo odrzucona odpowiedź i niepoparte tezy z uzasadnieniem sędziego. Najmocniejszy element
   demo: widać halucynację i widać, jak system ją zabija.
6. **Przełączniki `bramka` / `weryfikacja` / `top_k`** — `answer()` już przyjmuje te parametry.
   To samo pytanie z wyłączonym guardem pokazuje przeciek na żywo; dowód „przed/po" z README
   staje się interaktywny.
7. **Health w nagłówku** — Qdrant, Ollama, liczba chunków, nazwy modeli, aktualne progi. Czerwona
   kropka **przed** zadaniem pytania jest lepsza niż błąd po 20 sekundach czekania.

### Świadomie wycięte (YAGNI)

| wycięte | dlaczego |
|---|---|
| Historia rozmowy / pytania wieloturowe | System jest single-turn, gold set też. Multi-turn wymaga przepisywania zapytań („a co z Danią?" → pełne pytanie) i **unieważnia całą ewaluację**. Osobny checkpoint, nie dodatek do UI. |
| Logowanie / użytkownicy | Serwer wiąże się z `127.0.0.1`. Autentykacja w aplikacji lokalnej to teatr bezpieczeństwa. |
| Zakładka z metrykami ewaluacji | Liczby musiałyby być świeże, więc UI zaczęłoby udawać dashboard. README robi to lepiej i uczciwiej. |
| Przełącznik ciemnego motywu | Motyw idzie za `prefers-color-scheme`. Przełącznik to stan trzymany bez powodu. |

### Dwie decyzje frontendowe niezerowe

- **Rozwijanie źródeł na natywnym `<details>`**, nie na stanie w JS. Mniej kodu, działa bez skryptów.
- **Odpowiedź modelu wstawiana przez `textContent`, nie `innerHTML`.** Wyjście LLM-a traktujemy jak
  tekst niezaufany — model może wypluć `<script>`, bo w kontekście były znaczniki z PDF-a. Cytaty
  `[n]` linkujemy budując węzły DOM po zescape'owaniu, nie sklejając HTML ze stringów.

---

## 5. Obsługa błędów

| sytuacja | zachowanie |
|---|---|
| Qdrant nie odpowiada | czerwona kropka **przed** zapytaniem + zdarzenie `error` z podpowiedzią `docker start qdrant-osw` |
| Ollama nie odpowiada | `error` z podpowiedzią `ollama serve` |
| Timeout LLM (180 s w `call_ollama`) | `error`: „przekroczono limit 180 s" |
| Nieoczekiwany wyjątek | łapany w generatorze; do przeglądarki typ + komunikat, **pełny traceback tylko do logu serwera** |

Nigdy surowy traceback w UI, nigdy ciche zawieszenie.

**Uczciwe ograniczenie:** po zamknięciu karty w trakcie zapytania wątek roboczy dobiega do końca —
nie da się anulować `torch`/`urllib` w pół drogi.

---

## 6. Testy — bez GPU i bez Ollamy

- `tests/test_serve_stream.py` — mostek `stream.py` na fałszywej funkcji blokującej: kolejność
  zdarzeń, obsługa sentinela, propagacja wyjątku. Czysty test jednostkowy, milisekundy.
- `tests/test_api.py` — `TestClient` z podmienioną `answer()`: linie parsują się jako JSON,
  `final` jest ostatni, ścieżka odmowy nosi `blocked_by` i `rejected_answer`, `gate=false` emituje
  `skipped`, złe wejście daje 422.

**Świadomy non-goal:** żaden test nie dotyka Ollamy ani Qdranta. Test wymagający GPU i dwóch usług
nie jest uruchamiany, a test, którego się nie uruchamia, nie jest testem. Ścieżka z prawdziwymi
modelami sprawdzana ręcznie przy demo.

---

## 7. Pliki (etap 1)

```
src/serve/api.py            # FastAPI: 3 endpointy                        ~150 linii
src/serve/stream.py         # mostek callback → iterator (kolejka+wątek)   ~40 linii
src/serve/static/index.html # struktura                                    ~80 linii
src/serve/static/app.js     # bez frameworka, bez build-stepu             ~150 linii
src/serve/static/style.css  # CSS grid, prefers-color-scheme              ~150 linii
src/generate/answer.py      # + opcjonalny kwarg on_event (addytywnie)
config/config.py            # + SERVE_HOST/SERVE_PORT (blok na końcu pliku)
requirements.txt            # + fastapi, uvicorn (+ httpx tylko do testów)
tests/test_serve_stream.py
tests/test_api.py
```

`stream.py` jest osobno celowo: **nie wie nic ani o FastAPI, ani o RAG-u**. Zamienia „funkcję
blokującą wołającą callback" w „iterator zdarzeń", więc jest testowalny fałszywką, a `api.py`
zostaje czytelny.

---

## 8. Etap 2 — wgrywanie PDF-u do izolowanej, ulotnej kolekcji

Osobny commit **po** etapie 1. Powód rozdzielenia: etap 2 wymaga sparametryzowania warstwy
retrievalu korpusem, czyli pracy backendowej. Zmieszane z UI dałoby jedną wielką zmianę, w której
nie odróżnisz błędu w interfejsie od błędu w refaktorze retrievalu.

### Co to udowadnia (i czego NIE udowadnia)

**Nie** udowadnia, że „model generalizuje na niewidzianych danych" — model nie był trenowany
na żadnym dokumencie z korpusu; jego wagi są zamrożone, a wiedza wchodzi przez prompt w czasie
zapytania. Udowadnia coś mocniejszego: **cały pipeline** (parsowanie → chunking → embedding →
indeks → hybrydowy retrieval → trzy guardy → cytowanie) działa na dokumencie nigdy wcześniej
niewidzianym, **bez dotykania modelu**. To ekonomiczny argument za RAG: nowa wiedza kosztuje
sekundy indeksowania, nie godziny treningu.

### Wymóg izolacji — i dlaczego jest twardy

`src/index/build_index.py:67` robi `client.delete_collection(name)` przy każdym buildzie. Obecny
pipeline ingestu podłączony pod przycisk „wgraj PDF" **skasowałby zmierzoną kolekcję i przebudował
ją razem z obcym dokumentem** — liczby z README (recall@1 0.83, MRR 0.89) przestałyby być
odtwarzalne na tym indeksie. To nie ryzyko niespójnego stanu, to zanieczyszczenie zbioru
ewaluacyjnego. Ta sama zasada, która kazała zbudować held-out do kalibracji progu: **indeks,
na którym mierzę, musi być zamrożony.**

Sprzyjający fakt: `chunk.py:157` nadaje id jako `{source}__{i}`, więc id są **stabilne per
dokument** — dodanie pliku nie przesuwa id istniejących chunków i gold set się nie rozjeżdża.

### Projekt etapu 2

- **Osobna kolekcja Qdranta** (`demo_upload`), kasowana przy zamknięciu procesu **i przy starcie**
  — ten drugi zamiatacz sprawia, że `kill -9` nie zostawia sieroty. Ulotność jest **strukturalna,
  nie dyscyplinarna**: dokument demo fizycznie nie przeżywa procesu, więc nie da się przypadkiem
  zaśmiecić korpusu ewaluacyjnego.
  *Odrzucona alternatywa:* embeddingi w pamięci + cosine brute-force (300 chunków × 1024 wymiary
  to ~1,2 MB; przy tej skali dokładne przeszukanie bije HNSW, który jest aproksymacją potrzebną
  dopiero przy dużym korpusie). Odrzucone, bo osłabia demo — pokazujemy, że działa *cała* ścieżka
  razem z indeksem wektorowym, a nie ścieżka obok niej; druga implementacja retrievalu to też
  drugie miejsce do rozjechania się z pierwszym.
- **Plik nie dotyka dysku** — z uploadu prosto do PyMuPDF jako strumień bajtów
  (`fitz.open(stream=…, filetype="pdf")`), bez `data/raw/`, bez `/tmp`.
- **Przełącznik korpusu w UI:** `OSW — 5 raportów, 1509 chunków (zmierzony)` / `dokument ad hoc`.
- **Wykrywanie skanu:** po ekstrakcji liczymy **średnią liczbę znaków na stronę**; poniżej ~100 →
  odrzucamy z komunikatem „PDF bez warstwy tekstowej (prawdopodobnie skan) — wymagałby OCR,
  którego świadomie nie mamy w pipeline". Progu nie da się ustawić na zero, bo skan zwykle ma
  kilkanaście śmieciowych znaków na stronę (numeracja, artefakty).
- **Limity:** 25 MB / 300 stron — nie dla bezpieczeństwa, a żeby 2000-stronicowy tom nie zamroził
  demo na minutę.
- **Ostrzeżenie w UI przy aktywnym korpusie demo:** „próg bramki 0.35 skalibrowany na korpusie
  OSW, nie na tym dokumencie". Guardy nie przenoszą swoich gwarancji na obcy dokument i nie
  będziemy udawać, że przenoszą.

### Do zrobienia w backendzie (etap 2)

- `search.py:76` czyta `config.QDRANT_COLLECTION` na sztywno → parametr nazwy kolekcji.
- `get_bm25()` (`bm25.py:81`) to singleton bez klucza korpusu → cache kluczowany korpusem.
- `_get_chunks()` (`hybrid.py:31`) — to samo.
- `extract_pages(Path)` i `chunk_file(Path)` → warianty „na bajtach/tekście". Logika czyszczenia
  i chunkingu jest już w czystych funkcjach (`find_running_lines`, `clean_text`, `split_sentences`,
  `pack_chunks`), więc to opakowania, nie duplikacja — kod wychodzi z tego **bardziej testowalny**.

---

## 9. Znane ograniczenia tego UI

- Jedno zapytanie w danym momencie (aplikacja jednoosobowa, lokalna). Bez kolejki, bez sesji.
- Zapytanie trwa ~20 s przy włączonych wszystkich guardach (weryfikacja to dodatkowe wywołania
  sędziego per teza). To koszt guardu #3, nie wada UI — i UI go pokazuje liczbą.
- Ollama, Qdrant i GPU są dzielone z innymi procesami; równoległa ewaluacja w terminalu
  zauważalnie zwolni UI.
- Cytowanie per-teza na modelu 7B bywa nieprecyzyjne (grupuje źródła) — UI to uwidacznia,
  nie naprawia.

---

## 10. Kolejność realizacji

1. ✅ **Etap 1 — ZREALIZOWANY 2026-07-28** (PR #8, `ui-panel` zmergowany do `main`):
   `stream.py` → `api.py` → statyki → testy → ręczne demo. Plan z odhaczonymi krokami
   i odnotowanymi odstępstwami: `docs/plans/2026-07-28-ui-panel-etap1.md`.
2. ✅ **Etap 2 — ZREALIZOWANY 2026-07-29** (gałąź `ui-etap2-upload-pdf`): parametryzacja
   retrievalu korpusem (`src/retrieve/corpus.py`) → ingest z bajtów (`extract_pages_bytes`,
   `chunk_text`, `src/ingest/upload.py`) → ulotna kolekcja (`src/index/ad_hoc.py`) →
   przełącznik korpusu i upload w UI. Projekt z sekcji 8 zrealizowany bez odstępstw
   projektowych; szczegóły i liczby: `NOTATKI_REKRUTER.md` (etap 12) oraz README.

**Odstępstwa i uzupełnienia względem sekcji 8 (do wiedzy przy czytaniu specu):**
- **Transport pliku:** spec nie rozstrzygał formy uploadu. Wybrane: **surowe bajty w ciele POST-a**
  (`?name=` na etykietę źródła), NIE `multipart/form-data` — Starlette trzyma `UploadFile`
  w `SpooledTemporaryFile(max_size=1 MB)`, więc multipart **zapisałby PDF >1 MB na dysk**, łamiąc
  wymóg „plik nie dotyka dysku". Efekt uboczny: zero nowych zależności (`python-multipart`).
- **Dodany bezpiecznik:** kod **odmawia pracy**, gdy `UPLOAD_COLLECTION == QDRANT_COLLECTION`
  (nazwa kolekcji zmierzonej jest nadpisywalna środowiskiem od etapu 8, więc zbieżność nazw jest
  osiągalna, a jej skutkiem byłby cicho przebudowany indeks zmierzony).
- **Rozdzielone powody odmowy:** limit stron ma własny typ (`PdfTooLong`) i własny komunikat —
  pierwszy przebieg na żywo sklejał go z „nie udało się odczytać PDF-a", co prowadzi użytkownika
  w złą stronę. Znalezione dopiero na prawdziwym dokumencie, przy zielonych testach.
- **Wynik uboczny do backlogu:** chunking zakłada prozę analityczną i na dokumencie pisanym
  wypunktowaniami daje chunki do 5614 znaków (3% >2000 zn.), co podnosi koszt guardu #3 do 264 s.
  Zapisane jako pozycja 1 w `TODO.md`.

Po każdym etapie: wpis do `NOTATKI_REKRUTER.md`
(PROBLEM → CO ZROBIŁEM/alternatywy → DOWÓD → JAK UJĄĆ NA ROZMOWIE) i aktualizacja README.
