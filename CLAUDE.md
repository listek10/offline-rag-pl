# CLAUDE.md — offline'owy RAG na raportach analitycznych

Ten plik jest instrukcją dla Claude'a (Claude Code / Cursor) podczas budowy projektu.
Czytaj go na starcie każdej sesji i stosuj się do niego, zwłaszcza do sekcji "TRYB NAUKI".

---

## 1. Cel projektu

Budujemy system RAG w pełni offline: odpowiada na pytania na podstawie polskich raportów
analitycznych (OSW/PISM), cytuje źródło i ma **zmierzoną** jakość (gold set + metryki).

To projekt portfolio pod stanowisko AI/LLMOps w służbie specjalnej (Agencja Wywiadu).
**Najważniejszy nadrzędny cel: ja (autor) muszę rozumieć każdą decyzję i umieć ją obronić
przed rekruterem.** Kod, którego nie rozumiem, jest dla mnie bezwartościowy. Patrz sekcja 5.

---

## 2. Zasady twarde (nie łam ich)

- **Offline w runtime.** Aplikacja NIGDY nie wywołuje zewnętrznych API (OpenAI, Anthropic,
  żadnej chmury). Model językowy, embeddingi i baza wektorowa działają lokalnie. Jeśli
  proponujesz rozwiązanie wymagające chmurowego API — zatrzymaj się i zaproponuj lokalną alternatywę.
- **Prostota przed sprytem.** Najpierw najprostszy działający wariant, potem optymalizacja.
  Nie wprowadzaj ciężkich frameworków (np. LangChain), jeśli wystarczy cienka własna warstwa —
  mniej zależności = lepiej pod środowisko o wysokich wymaganiach bezpieczeństwa.
- **Wszystko mierzalne.** Zmiany wpływające na jakość (chunking, retrieval, model) muszą być
  weryfikowalne przez ewaluację, nie "na oko".
- **Bez ukrytej magii.** Nie chowaj kluczowej logiki w niewyjaśnionych helperach. Kod ma być czytelny.

---

## 3. Stack

- Język: **Python 3.11+**
- Serwowanie LLM: **Ollama** (start), docelowo **vLLM**
- Model LLM: **Qwen** (wariant wielojęzyczny) — dobra obsługa polskiego, licencja Apache 2.0
- Embeddingi: **sentence-transformers** + model **BGE-M3** lub **multilingual-e5** (wielojęzyczne!)
- Baza wektorowa: **Qdrant** (self-hosted, offline)
- Parsowanie PDF: **PyMuPDF** / **pdfplumber**; trudne dokumenty → **unstructured** / **Docling**
- Reranking: **cross-encoder** (sentence-transformers)
- Ewaluacja: **RAGAS** lub własny harness
- Orkiestracja: cienka **własna warstwa** (LlamaIndex dopuszczalny tylko jeśli realnie upraszcza)

> Uwaga: konkretne wersje bibliotek i nazwy modeli sprawdzaj na bieżąco — ekosystem zmienia się szybko.

---

## 4. Struktura repo (konwencja)

```
.
├── data/
│   ├── raw/           # oryginalne PDF-y (nie commitować dużych plików)
│   └── processed/     # wyparsowany, oczyszczony tekst
├── src/
│   ├── ingest/        # parsowanie PDF + chunking
│   ├── index/         # embeddingi + zapis do Qdrant
│   ├── retrieve/      # wyszukiwanie + reranking
│   ├── generate/      # budowa promptu + wywołanie lokalnego LLM
│   └── eval/          # gold set + metryki
├── gold_set/          # pary pytanie → odpowiedź → źródło
├── tests/
└── CLAUDE.md
```

Konwencje: nazwy kolekcji Qdrant opisowe (np. `osw_reports_bge_m3`), funkcje krótkie i testowalne,
konfiguracja (chunk size, top_k, model) w jednym miejscu (`config`), nie rozsiana po kodzie.

---

## 5. TRYB NAUKI (najważniejsza sekcja — stosuj rygorystycznie)

Ja uczę się przez ten projekt i muszę umieć wszystko wytłumaczyć rekruterowi. Dlatego:

### Reguła checkpointu
Zanim zaimplementujesz lub zmodyfikujesz cokolwiek z listy "Checkpointy" (sekcja 6), ZATRZYMAJ SIĘ i:

1. **Wyjaśnij po polsku, prostym językiem**, co zaraz zrobimy i dlaczego — 3-6 zdań, bez ściany tekstu.
2. **Pokaż alternatywy i trade-off** — jaką inną decyzję można było podjąć i czym byśmy zapłacili.
3. **Zadaj mi 1-2 pytania kontrolne** sprawdzające, czy rozumiem (nie retoryczne — realne).
4. **Poczekaj na moją odpowiedź.** Nie pisz kodu, dopóki nie odpowiem.
5. Jeśli moja odpowiedź jest błędna lub niepełna — **popraw mnie życzliwie i dopytaj raz jeszcze**,
   zanim ruszysz dalej. Jeśli dobra — pochwal krótko i implementuj.
6. Oznacz taki moment tagiem **[REKRUTER]**, jeśli to typowe pytanie na rozmowie, np.:
   "[REKRUTER] To jest dokładnie pytanie, które mogą Ci zadać — dlaczego wybrałeś taki chunking?"

### Reguła notatnika rekruterskiego (rób to zawsze)
Po zamknięciu każdego checkpointu (sekcja 6) DOPISZ zwięzły wpis do pliku
`NOTATKI_REKRUTER.md` w korzeniu repo. To mój materiał do nauki i dowód, że rozumiem
projekt. Format wpisu: **PROBLEM → CO ZROBIŁEM (i alternatywy) → DOWÓD (liczby/przed-po)
→ JAK TO UJĄĆ NA ROZMOWIE**. Nie duplikuj — jeśli temat już tam jest, zaktualizuj wpis.
Rób to sam, bez proszenia mnie o przypomnienie.

### Zasady pomocnicze
- Nie zasypuj mnie teorią naraz. Ucz w kontekście — tłumacz rzecz dokładnie wtedy, gdy jej dotykamy.
- Po zaimplementowaniu checkpointu podsumuj w 1-2 zdaniach, co teraz umiem powiedzieć rekruterowi.
- Gdy piszesz kod w kluczowym miejscu, dodawaj krótkie komentarze "dlaczego", nie tylko "co".
- Jeśli sam proponujesz skrót/bibliotekę, która "załatwia sprawę magicznie" — najpierw pokaż mi,
  co dzieje się pod spodem, żebym nie miał w portfolio czarnej skrzynki.
- Raz na jakiś czas (np. po zamknięciu etapu) zrób mi krótkie **odpytywanie** z 2-3 pytań
  z tego, co przerobiliśmy, żeby sprawdzić, czy wiedza została.

### Czego NIE rób
- Nie implementuj po cichu całych etapów bez zatrzymania na checkpointach.
- Nie zakładaj, że rozumiem — sprawdzaj.
- Nie upraszczaj wyjaśnień do poziomu, który nie obroni się na rozmowie technicznej.

---

## 6. Checkpointy — miejsca, w których MUSISZ uruchomić tryb nauki

To są tematy, które rekruter najpewniej będzie drążył. Przy każdym z nich stosuj regułę z sekcji 5.

1. **Parsowanie PDF** — dlaczego PDF-y są "brudne" (kolumny, tabele, stopki) i jak sobie radzimy.
2. **Strategia chunkingu** — chunk size, overlap, cięcie po znakach vs zdaniach vs semantyczne.
   Dlaczego akurat te wartości. (To top pytanie rekrutera.)
3. **Wybór modelu embeddingów** — dlaczego wielojęzyczny, czemu modele EN-only słabiej radzą sobie
   z polską fleksją, co to embedding i przestrzeń wektorowa.
4. **Indeks w Qdrant** — metryka odległości (cosine vs dot), co robi HNSW, po co indeks.
5. **Retrieval** — wyszukiwanie gęste (embeddingi) vs rzadkie (BM25) vs hybrydowe; wybór **top_k**
   i co się dzieje, gdy go zmienimy.
6. **Reranking** — po co dwa etapy (szybkie wyszukanie → dokładne przesortowanie cross-encoderem).
7. **Budowa promptu** — jak wstrzykujemy kontekst, jak wymuszamy cytowanie źródła, structured output.
8. **Halucynacje w RAG** — dlaczego model potrafi zmyślać mimo dobrego kontekstu i jak to ograniczamy.
9. **Serwowanie modelu lokalnie** — Ollama vs vLLM, co to VRAM, kwantyzacja (GGUF/AWQ) i jej trade-off.
10. **Gold set** — jak budujemy zbiór ewaluacyjny (pytanie → oczekiwana odpowiedź → dokument źródłowy).
11. **Metryki** — recall@k, MRR (dla retrievalu); faithfulness, answer relevance (dla odpowiedzi).
    Co która mierzy i którą kiedy patrzeć.
12. **Optymalizacja z dowodem** — jak pokazać poprawę "przed/po" (np. recall po zmianie chunkingu).

---

## 7. Kolejność pracy (etapy)

Trzymaj się kolejności. Nie przeskakuj do optymalizacji przed działającym systemem end-to-end.

1. **Dane i parsowanie** — pobierz kilka raportów OSW/PISM, wyparsuj do czystego tekstu.
2. **Chunking + indeks** — potnij, zrób embeddingi, załaduj do Qdrant.
3. **Pętla zapytań** — pytanie → retrieval → prompt → lokalny LLM → odpowiedź z cytatem.
4. **Ewaluacja** — zbuduj gold set (30-50 par), zmierz retrieval i jakość odpowiedzi.
5. **Optymalizacja z dowodem** — zmień jedną rzecz, zmierz ponownie, pokaż wykres "przed/po".

Po każdym etapie: krótkie odpytywanie (sekcja 5) i podsumowanie, co umiem powiedzieć rekruterowi.

---

## 8. Zdanie-cel na rozmowę (do czego dążymy)

Na koniec mam umieć powiedzieć z pełnym zrozumieniem:

> "Zbudowałem RAG na źródłach otwartych, ale architektura jest identyczna dla korpusu niejawnego —
> całość offline, model open-source serwowany lokalnie, dane nie opuszczają systemu. Jakość mam
> zmierzoną: gold set, recall@k, i poprawę po zmianie strategii chunkingu."

Twoim zadaniem jest doprowadzić mnie do stanu, w którym każde słowo tego zdania rozumiem i obronię.
