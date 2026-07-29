# TODO — co dalej (stan: 2026-07-29)

Jedno źródło prawdy o tym, co zostało. Wcześniej ta lista żyła w rozmowie i w `PROJECT_BRIEF.md`,
który jest w `.gitignore` — czyli nie istniała dla nikogo, kto otwiera repo. Trzymam ją tutaj,
w repo, a brief tylko na nią wskazuje: dwie kopie jednej prawdy zawsze się rozjeżdżają.

**Stan bazowy:** wszystkie 5 etapów z briefu + 8 ponad plan zamknięte, 12/12 checkpointów
z `CLAUDE.md` pokrytych, 157 testów zielonych bez GPU (+9 pominiętych, bo wymagają `pymupdf`
/ `qdrant-client`; w env projektu 166), CI na GitHub Actions zielone.
Liczby: `README.md`. Historia decyzji: `NOTATKI_REKRUTER.md`.

Kolejność = rekomendacja, nie zależności. Każda pozycja ma **jak zmierzyć**, bo w tym projekcie
zmiana bez pomiaru nie jest zrobiona.

---

## 1. Chunking na dokumentach listowych  ⬜ NOWE (wynik uboczny etapu 12)

**Problem (zmierzony 29.07 na wgranym raporcie `IV ROK WOJNY`, 100 stron):** `split_sentences`
tnie po interpunkcji, a raport pisany **wypunktowaniami** nie ma kropek — więc `pack_chunks` nie
ma gdzie przerwać. Rozkład: korpus bazowy max **2548** zn., 0% chunków >2000 zn.; wgrany dokument
max **5614** zn., **3% chunków >2000 zn.** przy podobnej średniej (1151 vs 1092 — średnia kłamie).

**Dlaczego to boli, a nie jest kosmetyką:** na wielkim chunku pytanie ogólne dało generację 43,5 s
i **weryfikację 264 s** (vs 26,7 s na korpusie bazowym), a guard #3 odrzucił odpowiedź
(`grounded_fraction` 0,0). Koszt guardu #3 rośnie z liczbą tez × wielkością kontekstu.

**Czego NIE robić od razu:** zmiana `split_sentences`/`pack_chunks` dotyka korpusu bazowego, więc
unieważnia zmierzone liczby (recall@1 0,83, MRR 0,89, A/B chunkingu z etapu 8) do czasu ponownego
pomiaru. Kandydat na najtańszy wariant: twardy limit długości „zdania" (dzielenie po `\n`/`•`
dopiero, gdy fragment przekracza np. 1,5× target) — czyli reguła awaryjna, nie nowa strategia.

**Jak zmierzyć:** `src/eval/ab_chunking.py` istnieje i mierzy warianty w osobnych kolekcjach;
warunek wejścia to **brak regresji** na gold secie (`run_eval.py`) plus rozkład długości chunków
przed/po na dokumencie listowym.

**Definicja zrobienia:** liczba „przed/po" dla obu korpusów (bazowy bez regresji, listowy z max
chunkiem w granicach targetu) albo wynik negatywny z nazwaną ceną.

---

## 2. Zbiór pułapek `n=7` → uczciwe `n`  ⬜ (etap 11 podniósł wagę tej pozycji)

**Po co:** wynik negatywny guardu #4 (etap 10) opiera się na 7 pytaniach, z czego podmiana
zaszła realnie w **jednym**. Mogę więc powiedzieć „nie zadziałał tam, gdzie mogłem zmierzyć",
a nie „nie działa". To jedyna droga, żeby ten guard dostał uczciwą drugą szansę.

**Drugi powód, mocniejszy (etap 11):** decyzja o prompcie sędziego stoi na JEDNYM pytaniu z
prawdziwą podmianą (`t01`) i JEDNYM z fałszywą odmową (`t06`). Zmierzona wymiana (fałszywa odmowa
9/9 → 0/9 blokad kosztem podmiany 18/18 → 10/18) jest wyraźna, ale opiera się na dwóch pytaniach.
Przy większym `n` może się okazać, że da się znaleźć brzmienie reguły, które kupuje jedno bez
drugiego — dziś nie mam podstawy, żeby to twierdzić ani temu zaprzeczyć.

**Czego dokładnie potrzeba:** pułapek typu **bałtyckiego** — wielkość i podmiot w JEDNYM zdaniu
(tam guard ma podstawę do werdyktu). Typ polarny (anafora „wyniosły *one* 8,4 mld") sprawia,
że guard abstynuje, więc dokładanie takich pytań nic nie rozstrzygnie.
Warunek powrotu guardu jest zapisany w `src/generate/attribution.py` (nagłówek statusu).

**Wymóg jakości danych:** każdy cytat walidowany programowo — musi występować w korpusie znak
w znak (tak jak `evidence` w gold secie). Skrypt sprawdzający był użyty przy budowie zbioru.

**Definicja zrobienia:** `eval_traps.py --compare` na powiększonym zbiorze; guard #4 albo
pokazuje zysk przy zerowym koszcie (wtedy wraca do rozmowy o włączeniu), albo zostaje odrzucony
z mocniejszym n.

---

## 3. Kontrakt payloadu w Qdrancie  ⬜ najmniej pilne

**Problem:** `build_index.py` zapisuje payload (`chunk_id`, `source`, `chunk_index`, `text`),
a `search.py` go czyta — dwa niezależne opisy jednego kształtu. Dokładnie ta klasa dryfu, którą
zamknęliśmy dla trafień przez `src/retrieve/hit.py` (commit `177c36e`), tylko po stronie indeksu.

**Dlaczego nie zrobione od razu:** domknięcie sensownie wymaga przebudowy indeksu, a ta unieważnia
zmierzone liczby do czasu ponownej ewaluacji. Koszt większy od ryzyka.

**Definicja zrobienia:** jedna fabryka payloadu używana przez zapis i odczyt + `run_eval.py`
z liczbami identycznymi jak przed zmianą.

---

## Zamknięte (dla kontekstu, żeby nie wracać)

- **Dziura fail-open w guardzie #3 (odpowiedzi bez ani jednej tezy) — ZAMKNIĘTA z pomiarem**,
  etap 13 (29.07). Poprawka dwuczęściowa, bo dziura miała dwie warstwy: tezą jest fragment z choć
  jednym znakiem alfanumerycznym (zamiast >15 znaków — nagła wielkość IDZIE do sędziego), a zero
  tez znaczy teraz „po usunięciu cytowań brak treści" i blokuje (fail-closed, z uzasadnieniem
  w `verdicts`). Kluczowa obserwacja z danych: przeciek `"120 [1]"` i POPRAWNA odpowiedź
  `"65% [1]"` (`g13`) mają identyczny kształt, więc „brak tez → blokuj" kupowało przecieki za
  pewny false-refusal. **DOWÓD:** przecieki 0/12 → 12/12 blokad; false-refusale 0/3 (`g13`,
  prawdziwa odpowiedź modelu) i 0/18 (6 wzorcowych odpowiedzi gold setu ≤15 znaków); zasięg
  zmiany udowodniony BEZ LLM-a — 744 z 779 zapisanych odpowiedzi ma identyczny zbiór tez, więc
  zmierzone liczby zachowały ważność bez ponownej generacji. Harness: `src/eval/ab_claims.py`
  (`--struktura` / `--sedzia` / `--wzorce`), wynik: `results/ab_claims.json`. Testy 140 → 157
  bez GPU (166 w env projektu). Granica dowodu (zapisana w README): obowiązuje na zmierzonym
  zbiorze odpowiedzi, nie „na zawsze".

- Reprodukowalność: podział `requirements.txt` / `requirements-dev.txt` — `697280a`
- Kontrakt trafienia w jednym miejscu (`hit.py`), dryf atrapy → `TypeError` — `177c36e`
- CI na GitHub Actions, 82 testy bez GPU — `97d366a`
- Metryka widząca podmianę podmiotu + zbiór pułapek + baseline — `bc697e2`
- Guard #4: zaprojektowany, zmierzony, **ODRZUCONY** (wynik negatywny) — `48c46f1`
- Fałszywe odmowy od form przymiotnikowych: **DWA wyniki negatywne** (prompt generatora
  i prompt sędziego), harness `src/eval/ab_prompt.py` + 9 testów — etap 11. Zysk realny na 1 z 2
  pytań, ale w produkcji zerowy (sędzia blokuje 9/9), a reguła u sędziego rozbraja detekcję
  prawdziwej podmiany (18/18 → 10/18). Nic nie weszło do runtime; liczby w README i notatniku.
- Oznaczenie stanu w planie UI i specu — `d60e05a`
- **UI etap 2: wgrywanie PDF-u do ulotnej kolekcji** — etap 12 (29.07). Korpus jako parametr
  retrievalu (`src/retrieve/corpus.py`, cache'e kluczowane, nadal leniwe), ingest z bajtów
  (`fitz.open(stream=…)`, bez multiparta — `UploadFile` zapisuje >1 MB na dysk), osobna kolekcja
  `demo_upload` z zamiataczem przy starcie i zamknięciu + odmowa przy zbieżności nazw kolekcji,
  jedna fabryka punktów Qdranta dla ingestu wsadowego i uploadu. **DOWÓD:** 100 stron → 364 chunki
  w 17,6 s, odpowiedź z cytatem w 26,8 s, kolekcja zmierzona 1509 punktów bez zmian, `run_eval.py`
  po demo identyczny (0,828 / 0,966 / 0,886). Testy 91 → 140 bez GPU, `check_app_js.js` 13 → 31.
- Usunięte trzy zmergowane gałęzie zdalne (SHA w historii `main`, odtwarzalne)
