# TODO — co dalej (stan: 2026-07-29)

Jedno źródło prawdy o tym, co zostało. Wcześniej ta lista żyła w rozmowie i w `PROJECT_BRIEF.md`,
który jest w `.gitignore` — czyli nie istniała dla nikogo, kto otwiera repo. Trzymam ją tutaj,
w repo, a brief tylko na nią wskazuje: dwie kopie jednej prawdy zawsze się rozjeżdżają.

**Stan bazowy:** wszystkie 5 etapów z briefu + 6 ponad plan zamknięte, 12/12 checkpointów
z `CLAUDE.md` pokrytych, 91 testów zielonych bez GPU, CI na GitHub Actions zielone.
Liczby: `README.md`. Historia decyzji: `NOTATKI_REKRUTER.md`.

Kolejność = rekomendacja, nie zależności. Każda pozycja ma **jak zmierzyć**, bo w tym projekcie
zmiana bez pomiaru nie jest zrobiona.

---

## 1. UI etap 2 — wgrywanie PDF-u do ulotnej kolekcji  ⬜ REKOMENDOWANE NASTĘPNE

**Wartość:** największa demonstracyjna — rekruter przynosi własny PDF i widzi, że cały pipeline
działa na dokumencie nigdy niewidzianym, bez dotykania modelu.

**Projekt jest gotowy:** `docs/specs/2026-07-28-ui-panel-design.md` **sekcja 8** (osobna kolekcja
`demo_upload` kasowana przy starcie i zamknięciu, plik nie dotyka dysku — `fitz.open(stream=…)`,
wykrywanie skanu po znakach/stronę, limity 25 MB / 300 stron, ostrzeżenie w UI że próg bramki
nie jest kalibrowany na obcym dokumencie). Sekcja 10 ma uwagę do wznowienia.

**Praca backendowa:** parametryzacja kolekcji w `search.py`, cache `get_bm25()` i `_get_chunks()`
kluczowany korpusem, warianty `extract_pages`/`chunk_file` na bajtach.

**Dwie pułapki, obie zapisane w specu:**
- `build_index.py` robi `delete_collection` przy każdym buildzie → podpięcie go pod przycisk
  „wgraj PDF" **skasowałoby zmierzoną kolekcję**; indeks, na którym mierzę, musi być zamrożony;
- `_get_chunks()`/`get_bm25()` to leniwe singletony, na których stoi testowalność bez GPU
  (patrz `requirements-dev.txt`) — kluczowanie korpusem **nie może** zamienić ich na eager.

**Definicja zrobienia:** obcy PDF → odpowiedź z cytatem, kolekcja bazowa nietknięta
(`run_eval.py` daje te same liczby po demo), testy kontraktowe bez GPU.

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

## 3. Dziura fail-open w guardzie #3 — odpowiedzi bez ani jednej tezy  ⬜ (nowe, z etapu 11)

**Problem (zmierzony 29.07, etap 11):** `grounded_fraction()` zwraca 1.0, gdy odpowiedź nie ma
ani jednej tezy — świadomy fail-open („nie ma czego podważać"). Ale `to_claims()` usuwa markery
`[n]` i odsiewa fragmenty ≤15 znaków, więc wyjścia `"[1]"` i `"444 [5]"` mają **zero tez** i
przechodzą przez **wszystkie trzy guardy** do użytkownika. Zmierzone na 9 wyjściach po stronie
pułapek i 3 przeciekach spoza korpusu. Dotyczy produkcji, nie tylko odrzuconego wariantu promptu.

**Dlaczego to nie jest jednolinijkowa poprawka:** „brak tez → blokuj" tyka też krótkich,
POPRAWNYCH odpowiedzi. Granica jest w `to_claims()` (próg 15 znaków po usunięciu cytowań), więc
zmiana wymaga sprawdzenia, co dziś przechodzi jako teza, a co nie.

**Jak zmierzyć:** `ab_prompt.py` ma już zapisane odpowiedzi z obu stron błędu — nowa reguła daje
się ocenić na nich bez generacji (pomiar parowany, zero GPU po stronie generatora). Do tego
`eval_answers.py` na false-refusal krótkich odpowiedzi z korpusu (np. „ok. 3 mld euro [2]").

**Definicja zrobienia:** liczba „przed/po" dla obu stron + brak nowych false-refusali na gold
secie, albo wynik negatywny z nazwaną ceną.

---

## 4. Kontrakt payloadu w Qdrancie  ⬜ najmniej pilne

**Problem:** `build_index.py` zapisuje payload (`chunk_id`, `source`, `chunk_index`, `text`),
a `search.py` go czyta — dwa niezależne opisy jednego kształtu. Dokładnie ta klasa dryfu, którą
zamknęliśmy dla trafień przez `src/retrieve/hit.py` (commit `177c36e`), tylko po stronie indeksu.

**Dlaczego nie zrobione od razu:** domknięcie sensownie wymaga przebudowy indeksu, a ta unieważnia
zmierzone liczby do czasu ponownej ewaluacji. Koszt większy od ryzyka.

**Definicja zrobienia:** jedna fabryka payloadu używana przez zapis i odczyt + `run_eval.py`
z liczbami identycznymi jak przed zmianą.

---

## Zamknięte (dla kontekstu, żeby nie wracać)

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
- Usunięte trzy zmergowane gałęzie zdalne (SHA w historii `main`, odtwarzalne)
