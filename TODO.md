# TODO — co dalej (stan: 2026-07-29)

Jedno źródło prawdy o tym, co zostało. Wcześniej ta lista żyła w rozmowie i w `PROJECT_BRIEF.md`,
który jest w `.gitignore` — czyli nie istniała dla nikogo, kto otwiera repo. Trzymam ją tutaj,
w repo, a brief tylko na nią wskazuje: dwie kopie jednej prawdy zawsze się rozjeżdżają.

**Stan bazowy:** wszystkie 5 etapów z briefu + 5 ponad plan zamknięte, 12/12 checkpointów
z `CLAUDE.md` pokrytych, 82 testy zielone bez GPU, CI na GitHub Actions zielone.
Liczby: `README.md`. Historia decyzji: `NOTATKI_REKRUTER.md`.

Kolejność = rekomendacja, nie zależności. Każda pozycja ma **jak zmierzyć**, bo w tym projekcie
zmiana bez pomiaru nie jest zrobiona.

---

## 1. Fałszywe odmowy od form przymiotnikowych  ⬜ REKOMENDOWANE NASTĘPNE

**Problem (zmierzony 29.07, etap 10):** korpus mówi „**łotewskie** 1,548 mld", pytanie mówi
„**Łotwy**" → model odmawia, choć chunk z odpowiedzią stoi na **pozycji 1** (rerank 0,931).
`blocked_by=None`, czyli odmawia sam generator (guard #1), a nie bramka. Retrieval jest bez winy.
Dotyczy `t05` i `t06` w `gold_set/entity_traps.jsonl`.

To **lustrzane odbicie** podmiany podmiotu: ten sam model raz przypisuje wartość za szeroko
(liczba Norwegii → Dania), a raz odmawia przypisania, choć fakt jest wprost w kontekście.

**Hipoteza:** instrukcja systemowa w `answer.py` (`SYSTEM_PROMPT`) nie mówi modelowi, że forma
przymiotnikowa („łotewskie wydatki") nazywa ten sam podmiot co rzeczownik z pytania („Łotwy”).
Prompt jest surowy anty-halucynacyjnie i model rozstrzyga wątpliwość na „nie znajduję".

**Jak zmierzyć (narzędzia GOTOWE, nic nie trzeba pisać):**
```bash
python src/eval/eval_traps.py --prepare              # generacja RAZ, obecny prompt
python src/eval/eval_traps.py --compare --repeats 3  # klasy: poprawna / podmiana / odmowa
python src/eval/eval_answers.py                     # KONTROLA drugiej strony błędu
```
**Obie strony trzeba raportować**, bo rozluźnienie promptu może podnieść przecieki:
- zysk: `t05`/`t06` przechodzą z `odmowa` do `poprawna`,
- koszt: `refusal` na pytaniach spoza korpusu (gold set n=7 + held-out n=17) nie może spaść.

**Uwaga metodologiczna (nie pomiń):** pomiar musi być **parowany** — temperatura 0 nie daje
powtarzalności (zmierzone, etap 10). Przy zmianie promptu generacja i tak się zmienia, więc
parowanie robimy po stronie guardów, a samą zmianę promptu mierzymy na **wielu powtórzeniach**
i raportujemy rozkład, nie jedną liczbę.

**Definicja zrobienia:** liczba „przed/po" dla obu stron błędu w README + wpis w notatniku.
Wynik negatywny (prompt nie pomaga albo koszt za duży) jest **pełnowartościowym zakończeniem**.

---

## 2. UI etap 2 — wgrywanie PDF-u do ulotnej kolekcji  ⬜

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

## 3. Zbiór pułapek `n=7` → uczciwe `n`  ⬜

**Po co:** wynik negatywny guardu #4 (etap 10) opiera się na 7 pytaniach, z czego podmiana
zaszła realnie w **jednym**. Mogę więc powiedzieć „nie zadziałał tam, gdzie mogłem zmierzyć",
a nie „nie działa". To jedyna droga, żeby ten guard dostał uczciwą drugą szansę.

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
- Oznaczenie stanu w planie UI i specu — `d60e05a`
- Usunięte trzy zmergowane gałęzie zdalne (SHA w historii `main`, odtwarzalne)
