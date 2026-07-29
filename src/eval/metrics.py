"""
src/eval/metrics.py
Etap 4 / checkpoint #11: metryki retrievalu (obiektywne, liczone z chunk_id).

- recall@k (hit-rate): czy PRZYNAJMNIEJ JEDEN trafny chunk jest w top-k. To najważniejsze dla
  RAG — mówi, czy właściwa informacja w ogóle trafiła do promptu w obrębie k. (Wariant "czy
  dotarł choć jeden trafny"; przy pytaniach z kilkoma poprawnymi chunkami liczymy trafienie,
  gdy złapiemy którykolwiek.)
- MRR (reciprocal rank): 1 / pozycja pierwszego trafnego chunka. Nagradza ustawienie trafnego
  WYSOKO. recall@k patrzy "czy jest w top-k", MRR — "jak blisko szczytu".

DWIE DEFINICJE PRAWDY (etap 8) — świadomie rozdzielone:
- `chunk_ids`: trafny = chunk wymieniony w gold secie. Tanie, ale WIĄŻE prawdę z konkretnym
  podziałem tekstu: po zmianie rozmiaru chunka numeracja się przesuwa i identyfikatory
  przestają cokolwiek wskazywać.
- `evidence`: trafny = chunk ZAWIERAJĄCY dosłowny cytat, w którym jest odpowiedź. Definicja
  niezależna od podziału, więc jedyna sensowna przy porównywaniu strategii chunkingu (także
  semantycznego, gdzie granice nie są funkcją liczby znaków). Jest też SUROWSZA: chunk „na
  temat", ale bez odpowiedzi, nie liczy się jako trafienie.
UWAGA: te dwie definicje dają RÓŻNE liczby bezwzględne, więc nie wolno porównywać recall
liczonego jedną z recallem liczonym drugą. Wariant bazowy raportujemy pod OBIEMA.
"""

from __future__ import annotations

import re


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 jeśli którykolwiek trafny chunk jest w top-k, inaczej 0.0 (dla jednego pytania)."""
    top = retrieved_ids[:k]
    return 1.0 if any(r in top for r in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / pozycja pierwszego trafnego chunka; 0.0 jeśli nie ma go w ogóle."""
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm(s: str) -> str:
    """Normalizuje białe znaki. Parser PDF-a zostawia miejscami dziwne odstępy (np.
    „fińsko- -estońskiego"), a dopasowanie ma być odporne na odstęp, nie na treść."""
    return re.sub(r"\s+", " ", s).strip()


def extract_values(text: str) -> set[str]:
    """Wyciąga wartości liczbowe w formie znormalizowanej (przecinek -> kropka, bez zer na końcu).

    Markery cytowań `[n]` USUWAMY najpierw: to numeracja źródeł, nie fakt. Bez tego odpowiedź
    z „[3]" udawałaby, że zawiera wartość 3 — metryka dawałaby fałszywe trafienia.
    """
    clean = re.sub(r"\[\d+\]", " ", text)
    out = set()
    for raw in re.findall(r"\d+(?:[.,]\d+)?", clean):
        s = raw.replace(",", ".")
        if "." in s:                      # „1,30" i „1,3" to ta sama wartość
            s = s.rstrip("0").rstrip(".")
        out.add(s)
    return out


def value_match(expected_answer: str, system_answer: str) -> tuple[bool | None, list[str]]:
    """Czy odpowiedź systemu zawiera WSZYSTKIE wartości z oczekiwanej odpowiedzi.

    DLACZEGO ta metryka istnieje (i dlaczego jest deterministyczna, bez sędziego): pozostałe
    metryki odpowiedzi są ŚLEPE na najgroźniejszą awarię tego systemu. Dla „wydatków obronnych
    Danii" pipeline podaje 1,75% PKB — liczbę NORWEGII — a `faithfulness` liczy to jako wierne
    (sędzia głosuje TAK, uzasadniając zdaniem o Norwegii), `false-refusal` jako poprawne (system
    odpowiedział), `relevance` jako na temat. Najpoważniejszy znany tryb awarii liczył się więc
    jako SUKCES. Pole `gold["answer"]` leżało w danych i nie było czytane przez żaden harness.

    Zwraca `(None, [])`, gdy oczekiwana odpowiedź nie zawiera liczby — takie pytanie jest POZA
    zakresem tej metryki i musi wypaść z mianownika. Zaliczenie go jako trafienia zawyżałoby
    wynik tym łatwiej, im mniej pytań liczbowych w zbiorze.

    ZNANE OGRANICZENIA (świadome, bo metryka ma być tłumaczalna, nie sprytna):
    - liczba zapisana słownie („cztery miliony") nie zostanie dopasowana -> zaniży wynik;
    - wartość obecna w odpowiedzi „przy okazji", w innym zdaniu, zaliczy się jako trafienie
      -> może zawyżyć. Dlatego raportujemy też LISTĘ braków do oględzin, a nie samą średnią.
    """
    want = extract_values(expected_answer)
    if not want:
        return None, []
    have = extract_values(system_answer)
    missing = sorted(want - have)
    return not missing, missing


def classify_trap_answer(expected_answer: str, trap_values: list[str], system_answer: str) -> str:
    """Klasyfikuje odpowiedź na pytanie-pułapkę (`gold_set/entity_traps.jsonl`).

    Zwraca jedną z: `poprawna` | `podmiana` | `niejednoznaczna` | `inna-bledna`.
    (Odmowę rozpoznaje warstwa wywołująca — `refused()` z answer.py — i nie wchodzi tu wcale.)

    DLACZEGO CZTERY KLASY, A NIE DWIE: pułapka ma więcej niż jeden interesujący wynik.
    - `podmiana` = w odpowiedzi jest wartość INNEGO podmiotu, a nie ma właściwej. To jest awaria
      h18 („wydatki Danii" -> liczba Norwegii) i jedyna klasa, która jest NIEBEZPIECZNA: system
      kłamie pewnym głosem i z cytatem.
    - `niejednoznaczna` = są OBIE wartości. Zwykle znaczy, że odpowiedź wymieniła oba państwa —
      rozwlekła, ale nie halucynacja. Wrzucenie tego do `podmiany` zawyżałoby awaryjność.
    - `inna-bledna` = żadnej znanej wartości. Błąd, ale inny gatunek (brak faktu, nie podmiana).
    Wystarczy JEDNA z oczekiwanych wartości: „1,3% PKB" bez kwoty jest poprawną odpowiedzią na
    pytanie o wydatki, a wymaganie wszystkich karałoby zwięzłość.
    """
    have = extract_values(system_answer)
    want = extract_values(expected_answer)
    has_expected = bool(want & have)
    has_trap = bool(set(trap_values) & have)
    if has_expected and has_trap:
        return "niejednoznaczna"
    if has_expected:
        return "poprawna"
    if has_trap:
        return "podmiana"
    return "inna-bledna"


def relevant_by_evidence(chunks: list[dict], evidence: str) -> list[str]:
    """Zwraca id chunków ZAWIERAJĄCYCH cytat-dowód — prawda niezależna od podziału tekstu.

    Liczy WSZYSTKIE takie chunki, nie tylko pierwszy: przy zakładce ten sam fragment bywa w
    dwóch kolejnych chunkach i oba realnie zawierają odpowiedź. Konsekwencja, którą trzeba
    znać: większa zakładka = więcej chunków z odpowiedzią = statystycznie łatwiejsze trafienie.
    Dlatego w A/B rozmiaru chunka trzymamy UDZIAŁ zakładki stały (15%), żeby nie mieszać
    dwóch efektów w jednej liczbie.
    """
    cyt = _norm(evidence)
    return [c["id"] for c in chunks if cyt in _norm(c["text"])]
