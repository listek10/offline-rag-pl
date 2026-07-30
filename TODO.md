# TODO — co dalej (stan: 2026-07-30)

Jedno źródło prawdy o tym, co zostało. Wcześniej ta lista żyła w rozmowie i w `PROJECT_BRIEF.md`,
który jest w `.gitignore` — czyli nie istniała dla nikogo, kto otwiera repo. Trzymam ją tutaj,
w repo, a brief tylko na nią wskazuje: dwie kopie jednej prawdy zawsze się rozjeżdżają.

**Stan bazowy:** wszystkie 5 etapów z briefu + 10 ponad plan zamknięte, 12/12 checkpointów
z `CLAUDE.md` pokrytych, 178 testów zielonych bez GPU (+9 pominiętych, bo wymagają `pymupdf`
/ `qdrant-client`; w env projektu 187), CI na GitHub Actions zielone.
Liczby: `README.md`. Historia decyzji: `NOTATKI_REKRUTER.md`.

Kolejność = rekomendacja, nie zależności. Każda pozycja ma **jak zmierzyć**, bo w tym projekcie
zmiana bez pomiaru nie jest zrobiona.

---

## 1. Zbiór pułapek `n=7` → uczciwe `n`  ⬜ OSTATNIA POZYCJA (etap 11 podniósł jej wagę)

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

## Zamknięte (dla kontekstu, żeby nie wracać)

- **Kontrakt payloadu w Qdrancie — ZAMKNIĘTY z pomiarem**, etap 15 (30.07). Ta pozycja mówiła
  o DWÓCH niezależnych opisach kształtu; było ich **trzy**: zapis (`index_records`), odczyt z bazy
  (`search.py`) i składanie trafienia z rekordu (`hybrid.py` — trafienia z samego BM25 nie
  przechodzą przez Qdranta). Ta trzecia jest najgroźniejsza, bo jej rozjechanie się z odczytem
  z bazy **nie daje wyjątku**, tylko trafienie z metadanymi innego dokumentu — czyli zły cytat
  wyglądający na poprawną odpowiedź. Rozwiązanie: `src/index/payload.py` (`make_payload`
  z argumentami tylko po nazwie, `payload_from_record`, `hit_from_payload`, `hit_from_record`);
  zapis i odczyt w JEDNYM module celowo, żeby rename łamał obie strony naraz. `make_hit()` jest
  teraz wołane wyłącznie z tej fabryki. Test `test_ad_hoc_index` zostawiony z LITERALNYM payloadem
  (pin formatu w bazie — asercja przez stałą z modułu podążałaby za kodem). Dodatkowo narzędzie na
  DANE, a nie na kod: `src/index/verify_payload.py` (kod wyjścia ≠ 0 przy niezgodności) rozróżnia
  payload różny od fabryki, sierotę w kolekcji i chunk brakujący po przerwanym buildzie — czego
  żaden test w `pytest` nie zobaczy, bo tam retrieval jest atrapą. **DOWÓD:** payload w bazie vs
  fabryka **1497/1497 znak w znak** (`results/payload_contract.json`), zero sierot; po przebudowie
  nową ścieżką zapisu ponownie 1497/1497; `run_eval.py` przed i po **identyczny BIT W BIT** —
  4 konfiguracje × 29 pytań × każde pole + identyczna lista odmów (0,828 / 0,966 / 0,886,
  odmowa 1,000). Testy 178 → **187** w env projektu. Uwaga metodologiczna: tutaj „identyczny" to
  **wymóg, nie sukces** — payload jest ten sam, więc ten sam tekst → ten sam wektor → ten sam
  ranking; każda różnica znaczyłaby błąd w refaktorze. Obawa z tej pozycji („przebudowa unieważnia
  zmierzone liczby") okazała się nieaktualna: przebudowa to ~3 min, a liczby są odtwarzalne.

- **Ogon rozkładu długości chunków na dokumentach wypunktowanych — ZAMKNIĘTY z pomiarem**,
  etap 14 (30.07). Poprawka dwuczęściowa, bo **pomiar wywrócił diagnozę zapisaną w tej pozycji**:
  sam limit długości „zdania" zostawiał chunk 2417 zn. Druga, prawdziwa przyczyna siedziała w pętli
  zakładki `pack_chunks` — brała **zawsze co najmniej jedno** zdanie bez względu na długość, więc
  moloch wchodził do nowego chunka i doklejało się do niego następne zdanie (2867 + 2854 = 5722).
  Ten sam mechanizm dawał 2548 zn. w korpusie bazowym, czyli nie była to usterka dokumentów
  listowych, a niezamierzone zachowanie zakładki niewidoczne na prozie (zdania ~190 zn.).
  Naprawa: `split_overlong()` (limit 1800 zn. = 1,5 × target, cięcie na `•`/`- `, w ostateczności
  po spacji) + `carry_max` = 600 zn. w `pack_chunks`; oba progi w `config`, nadpisywalne
  środowiskiem, więc stan „przed" odtwarza jedna zmienna. Reguła awaryjna jest **osobną funkcją
  wołaną tylko z ingestu** — `split_sentences` jest współdzielone z guardem #3, więc zmiana w niej
  unieważniłaby pomiary z etapu 13. Wybór progów z siatki limit × kap (kap 180 odrzucony: likwiduje
  zakładkę na prozie, −9% chunków). **DOWÓD:** korpus bazowy max 2548 → 1621 zn., >2000: 1 → 0
  chunków, 96,7% chunków identycznych znak w znak, żaden cytat `evidence` z gold setu w ruszonym
  chunku, wszystkie 40 `chunk_id` wskazują ten sam tekst; dokument listowy max 5722 → 1736 zn.,
  >2000: 0,62% → 0,00%. Ewaluacja: `hybrid+rr` bez zmian w ANI JEDNYM polu (0,793/0,931/0,966/
  0,966/MRR 0,863), dense tylko `g02` z pozycji 7 na 8 (MRR −0,0006); tryb domyślny odtworzył
  nagłówek README (0,828 / 0,966 / 0,886, odmowa 1,000). Harness: `src/eval/ab_ogon.py`
  (`--rozklad` / `--zasieg`), wyniki: `results/ab_ogon.json`, `results/ab_ogon_listowy.json`,
  `results/ab_chunking_ogon.json`. Testy 166 → 178 w env projektu (169 bez GPU).
  Granica dowodu (w README): liczby z sesji na żywo z etapu 12 (max 5614, 3%) pochodzą z ulotnego
  dokumentu 100-stronicowego i **nie odtwarzają się** — plik w `data/raw_deferred/` to inny
  egzemplarz (518 stron). Klasa defektu ta sama, wartości bezwzględne inne.

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
