"""
src/generate/attribution.py
Kandydat na guard #4: KONTROLA PRZYPISANIA WIELKOŚCI DO PODMIOTU (anti-„podmiana podmiotu").

╔══ STATUS: ZMIERZONY DWA RAZY I **NIEWŁĄCZONY** DO PRODUKCJI — WYNIK NEGATYWNY ════════════╗
║ `answer()` NIE woła tego modułu. Zostaje w repo jako udokumentowana, przetestowana próba    ║
║ z pomiarem, bo negatywny wynik z liczbami jest wart więcej niż guard dodany „na wyczucie".  ║
║                                                                                            ║
║ POMIAR 1 (etap 10, n=7): guard abstynował na 6 z 7 pytań, więc wniosek brzmiał uczciwie     ║
║ „nie zadziałał tam, gdzie mogłem zmierzyć" — podmiana zaszła realnie w JEDNYM pytaniu.      ║
║                                                                                            ║
║ POMIAR 2 (etap 16, n=21 — zbiór zbudowany minerem `src/eval/mine_traps.py` POD TEN guard,   ║
║ tak by wielkość i podmiot stały w jednym zdaniu; 3 powtórzenia, pomiar parowany):           ║
║     bez guardów post:   poprawna 51  PODMIANA 3  odmowa 9                                  ║
║     guard #3 (sędzia):  poprawna 51  PODMIANA 0  odmowa 12  <- łapie, koszt 0               ║
║     ten guard:          poprawna 51  PODMIANA 3  odmowa 9   <- ZERO BLOKAD                  ║
║     ten guard + szersze jednostki: identycznie                                              ║
║ Audyt zasięgu (`eval_traps.py --audyt`): 19/21 pytań to REALNE pułapki (cudza liczba jest   ║
║ w kontekście), 13/21 jest w zasięgu tego guardu, **11/21 rozstrzyga** (pułapka I zasięg) —  ║
║ wobec 1 pytania w pomiarze 1. Na wszystkich 13 pytaniach w zasięgu guard wydał werdykt      ║
║ POPRAWNY („wartość należy do pytanego podmiotu") — czyli nie jest zepsuty, jest bezużyteczny║
║ na tym korpusie: nie ma czego blokować.                                                     ║
║ Przyczyna niezmieniona: jedyna realna podmiana (t01) siedzi w zdaniu z anaforą („wyniosły   ║
║ ONE 8,4 mld"), więc reguła ograniczona do zdania nie ma podstawy do werdyktu — i zgodnie    ║
║ z projektem woli nie wiedzieć, niż zgadnąć. Wcześniejsza, śmielsza wersja (szukanie wstecz  ║
║ w całym bloku) była AKTYWNIE SZKODLIWA: przypisywała 1,75% PKB *Rosji* („…Globus III        ║
║ w Vardø przy granicy z Rosją" dwa zdania wcześniej) i blokowała poprawną odpowiedź          ║
║ o Norwegii.                                                                                 ║
║ Warunek powrotu jest teraz WĘŻSZY i sprawdzalny: pułapki, w których model REALNIE podmienia ║
║ podmiot, a wielkość stoi w jednym zdaniu z nazwą. Tego korpus 5 raportów nie dostarcza —    ║
║ na 19 realnych pułapek model podmienił raz, i to w typie anaforycznym.                      ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝

PROBLEM, KTÓRY TO ROZWIĄZUJE (zmierzony, nie hipotetyczny):
na pytanie „Ile wynosiły wydatki obronne Danii w 2021 r.?" pipeline odpowiada „8,4 mld dolarów,
1,75% PKB" — czyli liczbą NORWEGII, słowo w słowo tym samym zdaniem co dla pytania o Norwegię.
Guard #3 (sędzia LLM) łapie to niekonsekwentnie, bo pytany o werdykt TAK/NIE racjonalizuje:
w logu z 28.07 sam napisał „zwiększenie budżetu resortu obrony NORWEGII" i zagłosował TAK
za tezą o Danii. Informacja była wyekstrahowana poprawnie — zawiódł krok decyzyjny.

ZASADA PROJEKTOWA: nie prosimy modelu o decyzję, którą umiemy podjąć w kodzie na podstawie
faktu obecnego w tekście. Ten guard nie woła LLM-a wcale — jest deterministyczny, darmowy
(zero latencji, zero VRAM) i w pełni wytłumaczalny: pokazuje, KTÓRA wzmianka zdecydowała.

MECHANIZM — „najbliższa wcześniejsza wzmianka".
Wartość w tekście eksperckim należy do podmiotu nazwanego najbliżej PRZED nią; dalej autor
używa anafory („wyniosły ONE 8,4 mld") albo elipsy. Dlatego dla każdego wystąpienia wartości
szukamy najbliższej poprzedzającej nazwy podmiotu i to ją uznajemy za właściciela liczby.

Dlaczego nie prościej — sprawdzone na danych, oba warianty odrzucone:
- POZIOM ZDANIA (czy zdanie z liczbą nazywa podmiot z pytania): zdanie „W 2021 r. wyniosły one
  8,4 mld dolarów, tj. 1,75% PKB" nie nazywa NIKOGO, więc ta reguła zablokowałaby także
  POPRAWNĄ odpowiedź o Norwegii. False-refusal z konstrukcji.
- POZIOM CHUNKA (czy chunk z liczbą nazywa podmiot z pytania): chunk #98 nazywa I Danię,
  I Norwegię — to zresztą powód, dla którego model je pomylił. Ta reguła nie zablokowałaby nic.
Reguła „najbliższa wcześniejsza wzmianka" trafia w OBA przypadki naraz, a na klastrze bałtyckim
radzi sobie nawet wewnątrz jednego zdania: „…Estonii… 2,316 mld…, podczas gdy łotewskie
1,548 mld, a litewskie 2,048 mld" — przed każdą liczbą stoi właściwy podmiot.

ŚWIADOME OGRANICZENIA (guard ma wąski, zadeklarowany zakres — to jego zaleta, nie wada):
1. Działa tylko dla pytań o podmiot Z LEKSYKONU i tylko dla tez zawierających LICZBĘ.
   Poza tym zakresem ABSTYNUJE (zwraca None) i decyzję zostawia guardowi #3. Guard, który
   udaje kompetencję poza swoim zakresem, jest gorszy niż guard, który mówi „nie wiem".
2. LEKSYKON zamiast lematyzatora: polska fleksja i formy przymiotnikowe („Łotwa" -> „łotewskie",
   „Dania" -> „duński") nie sprowadzają się do wspólnego prefiksu, więc formy są WYPISANE.
   Alternatywa (morfeusz/spacy-pl) to nowa ciężka zależność do audytu — ten sam trade-off,
   który już świadomie przyjęliśmy w BM25 („znana wada: brak lematyzacji").
3. Zawodzi, gdy wartość STOI PRZED swoim podmiotem albo gdy nazwa właściciela wypadła za
   granicę chunka. Dlatego brak jakiejkolwiek wzmianki = ABSTYNENCJA, nie blokada.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Leksykon form: podmiot kanoniczny -> formy (małymi literami), po których go rozpoznajemy.
# Zakres = korpus projektu (bezpieczeństwo nordycko-bałtyckie). Formy dobrane tak, żeby złapać
# odmianę rzeczownika I formę przymiotnikową, bo raporty używają obu wymiennie.
#
# ═══ FORMA JEST PREFIKSEM SŁOWA, NIE DOWOLNYM PODCIĄGIEM (naprawa zmierzona) ═══
# Pierwsza wersja szukała form jako podciągu w dowolnym miejscu słowa i było to CICHO błędne:
# „ba-DANI-a" liczyło się jako wzmianka Danii (335 z 463 trafień formy „dani" w korpusie!),
# „b-RYG-ada" jako Łotwa (85 z 89), „cy-WILNO-wojskowy" jako Litwa, „jezusa" jako USA.
# Dla guardu przypisania to nie kosmetyka: fałszywa wzmianka stojąca bliżej liczby PORYWA
# przypisanie i daje albo blokadę poprawnej odpowiedzi, albo przepuszczenie podmiany.
# Dlatego forma musi zaczynać SŁOWO (`\b`), a formy, które są prefiksem zwykłych polskich słów,
# są wypisane w pełnych odmianach: „dani" (→ danina, Daniel) rozbite na „dania/danii/danią/danię",
# „fin" (→ finansowanie, definicja) na „finów/finowie/finom/finami", „ryg" (→ brygada, rygor)
# na „ryga/rygi/rygą/rydze". Koszt: dłuższa lista i ryzyko, że jakiejś odmiany zabraknie —
# ale brak odmiany daje ABSTYNENCJĘ (guard milczy), a fałszywa wzmianka dawała WERDYKT.
# W tym kierunku wolimy się mylić, więc ten koszt jest zamierzony.
ENTITY_FORMS: dict[str, tuple[str, ...]] = {
    "Dania":      ("dania", "danii", "danią", "danię", "duń", "duns", "kopenhag"),
    "Norwegia":   ("norwe", "norwes", "oslo"),
    "Szwecja":    ("szwecj", "szwed", "sztokholm"),
    "Finlandia":  ("finlandi", "fiń", "finów", "finowie", "finom", "finami", "helsink"),
    "Islandia":   ("islandi", "islandz", "reykjav"),
    "Estonia":    ("estoni", "estoń", "eston", "tallin"),
    "Łotwa":      ("łotw", "łotew", "ryga", "rygi", "rygą", "rydze"),
    "Litwa":      ("litw", "litew", "wilno", "wilnie", "wilna"),
    "Kanada":     ("kanad", "ottaw"),
    "USA":        ("usa", "stany zjednoczone", "amerykan", "waszyngton"),
    "Rosja":      ("rosj", "rosyj", "moskw", "kreml"),
    "Niemcy":     ("niemc", "niemieck", "berlin"),
    "Polska":     ("polsk", "warszaw"),
    "Grenlandia": ("grenlandi", "grenlandz"),
}

# Skróty muszą być CAŁYM słowem: „usa" jako prefiks łapie „usamodzielnił", „usankcjonowano",
# „usatysfakcjonuje". Cyfra po formie jest dopuszczalna, bo parser PDF-a dokleja numery
# przypisów do słowa („usa110", „Danią121") — to artefakt korpusu, nie inne słowo.
WHOLE_WORD_FORMS = frozenset({"usa"})

_LETTER = "a-zA-Zа-яąćęłńóśźżĄĆĘŁŃÓŚŹŻ"


def _form_pattern(form: str) -> re.Pattern[str]:
    tail = rf"(?![{_LETTER}])" if form in WHOLE_WORD_FORMS else ""
    return re.compile(rf"\b{re.escape(form)}{tail}")


# Kompilujemy raz — `_mentions` woła się dla każdego zdania każdego chunka w kontekście.
_FORM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_form_pattern(f), ent) for ent, forms in ENTITY_FORMS.items() for f in forms
)


def _mentions(text: str) -> list[tuple[int, str]]:
    """Wszystkie wzmianki podmiotów w tekście jako (pozycja, podmiot), posortowane po pozycji."""
    low = text.lower()
    return sorted((m.start(), ent) for pat, ent in _FORM_PATTERNS for m in pat.finditer(low))


def entity_in(text: str) -> str | None:
    """Który podmiot z leksykonu jest nazwany w tekście (np. w pytaniu). None = żaden.

    Bierzemy NAJPÓŹNIEJSZĄ wzmiankę — w pytaniach typu „Ile wynosiły wydatki obronne Danii
    w 2021 r.?" podmiot stoi po rzeczowniku opisującym wielkość, a ewentualna wcześniejsza
    nazwa (np. „w NATO") jest tłem, nie tematem.

    Ta funkcja liczy wzmianki TĄ SAMĄ regułą co `_mentions` (jedno miejsce, jedna reguła):
    gdyby pytanie i kontekst rozpoznawały podmioty inaczej, guard porównywałby jabłka z gruszkami.
    """
    ms = _mentions(text)
    return ms[-1][1] if ms else None


# Wielkość = liczba + JEDNOSTKA. Rok („2021 r.") jednostki nie ma i celowo nie jest wielkością:
# to modyfikator czasu, nie przypisywana ilość. Bez tego zawężenia przypadkowy rok w odpowiedzi
# „zaliczał" trafienie w pytany podmiot i przepuszczał podmianę kwoty (wykryte pomiarem).
_UNIT = r"(?:%|proc\.|mld|mln|tys\.)"


def _quantity_positions(text: str, value: str) -> list[int]:
    """Pozycje wystąpień WIELKOŚCI (liczba + jednostka). `value` jest znormalizowane („1.75"),
    a tekst używa przecinka, więc dopuszczamy oba separatory. Osłony z obu stron, żeby „1,3"
    nie trafiało w środek „1,35" ani w „11,3"."""
    num = re.escape(value).replace(r"\.", "[.,]")
    return [m.start() for m in re.finditer(rf"(?<![\d,.]){num}(?![\d])\s*{_UNIT}", text)]


def quantities(text: str) -> set[str]:
    """Wielkości (liczba+jednostka) występujące w tekście, znormalizowane jak w `extract_values`."""
    out = set()
    for raw in re.findall(rf"(\d+(?:[.,]\d+)?)(?=\s*{_UNIT})", text):
        s = raw.replace(",", ".")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        out.add(s)
    return out


def attributed_entities(context: str, value: str) -> set[str]:
    """Komu kontekst przypisuje daną wielkość: najbliższa wcześniejsza wzmianka W TYM SAMYM ZDANIU.

    DLACZEGO TYLKO W ZDANIU — pierwotna wersja szukała wstecz w całym bloku i POMIAR JĄ OBALIŁ.
    W prawdziwym chunku przed liczbą Norwegii stoi „stacja radiolokacyjna Globus III w Vardø przy
    granicy z ROSJĄ", więc reguła „najbliższa wcześniejsza wzmianka w bloku" przypisywała 1,75%
    PKB *Rosji* i blokowała POPRAWNĄ odpowiedź o Norwegii. Przypadkowa nazwa państwa w zdaniu
    obok porywa przypisanie — a takich nazw w raportach geopolitycznych jest pełno.
    (Mój test tego nie złapał, bo używałem fragmentu SKRÓCONEGO, a skrót usunął właśnie to zdanie.
    Lekcja: fragment „prawdziwy, ale skrócony" nie jest tym samym co prawdziwy.)

    Zawężenie do zdania kosztuje zasięg: gdy zdanie z liczbą używa anafory („wyniosły ONE 8,4
    mld"), guard ABSTYNUJE, zamiast zgadywać. To świadomy wybór precyzji nad czułością —
    fałszywa odmowa na poprawnej odpowiedzi jest w tym systemie kosztem realnym, a lukę
    w czułości pokrywa guard #3.
    Wewnątrz zdania kolejność wciąż działa i to wystarcza w najtrudniejszym przypadku korpusu:
    „…Estonii… 2,316 mld…, podczas gdy łotewskie 1,548 mld, a litewskie 2,048 mld".
    """
    from src.ingest.chunk import split_sentences      # ten sam podział zdań co w guardzie #3

    out: set[str] = set()
    for block in re.split(r"\n\n(?=\[\d+\] \(źródło:)", context):
        for sent in split_sentences(block):
            mentions = _mentions(sent)
            for pos in _quantity_positions(sent, value):
                earlier = [(p, e) for p, e in mentions if p < pos]
                if earlier:
                    out.add(earlier[-1][1])
    return out


def check_attribution(question: str, answer_text: str, context: str) -> tuple[bool | None, list[dict]]:
    """Czy wartości z odpowiedzi są w kontekście przypisane podmiotowi z PYTANIA.

    Zwraca `(werdykt, szczegóły)`:
      True  — PRZYNAJMNIEJ JEDNA wartość z odpowiedzi jest w kontekście przypisana podmiotowi
              z pytania (odpowiedź zawiera żądany fakt),
      False — ŻADNA nie należy do podmiotu z pytania, a co najmniej jedna należy do KOGO
              INNEGO. To jest podmiana podmiotu.
      None  — ABSTYNENCJA: pytanie nie nazywa podmiotu z leksykonu, odpowiedź nie ma liczb
              albo żadnej z nich nie udało się w kontekście przypisać. Wtedy decyduje guard #3.

    DLACZEGO „przynajmniej jedna", a nie „wszystkie" (wadę wykrył test na realnych danych):
    odpowiedź „Estonii 2,316 mld, łotewskie 1,548 mld, litewskie 2,048 mld" ma KAŻDĄ wartość
    przypisaną poprawnie i jest dobrą odpowiedzią na pytanie o Estonię — tylko rozwlekłą.
    Kryterium „wszystkie wartości muszą należeć do pytanego" blokowałoby ją, czyli karałoby
    rozwlekłość jak halucynację. Rozróżnienie, które ma znaczenie, jest inne: czy odpowiedź
    W OGÓLE podaje wartość pytanego podmiotu. Jeśli nie podaje żadnej, a podaje czyjąś — to
    właśnie podmiana. (Ta sama logika co klasy `podmiana` vs `niejednoznaczna` w metryce.)
    `szczegóły` trafiają do panelu diagnostycznego — guard ma być audytowalny, nie wyroczny.
    """
    asked = entity_in(question)
    if asked is None:
        return None, [{"powód": "pytanie nie nazywa podmiotu z leksykonu"}]

    values = quantities(answer_text)
    if not values:
        return None, [{"powód": "odpowiedź nie zawiera wielkości (liczba + jednostka)"}]

    details, hit_asked, hit_other = [], False, False
    for v in sorted(values):
        owners = attributed_entities(context, v)
        if not owners:
            details.append({"wartość": v, "przypisana": None, "pytanie_o": asked,
                            "ocena": "brak wzmianki w zasięgu — bez oceny"})
            continue
        if len(owners) > 1:
            # WARTOŚĆ NIEROZSTRZYGAJĄCA — kontekst przypisuje ją kilku podmiotom, więc nie
            # niesie sygnału o przypisaniu i NIE GŁOSUJE. Wadę wykrył test: rok „2021" stoi
            # przy obu państwach, więc bez tej reguły przypadkowy rok w odpowiedzi „zaliczał"
            # trafienie w pytany podmiot i przepuszczał podmianę kwoty.
            details.append({"wartość": v, "przypisana": sorted(owners), "pytanie_o": asked,
                            "ocena": "wieloznaczna — nie rozstrzyga"})
            continue
        if asked in owners:
            hit_asked = True
            ocena = "zgodna"
        else:
            hit_other = True
            ocena = "należy do innego podmiotu"
        details.append({"wartość": v, "przypisana": sorted(owners), "pytanie_o": asked,
                        "ocena": ocena})

    if hit_asked:
        return True, details
    if hit_other:
        # Żadnej wartości pytanego podmiotu, a jest wartość cudza -> podmiana.
        for d in details:
            if d["ocena"] == "należy do innego podmiotu":
                d["ocena"] = "PODMIANA PODMIOTU"
        return False, details
    return None, details
