"""
src/generate/verify.py
Guard #3: POST-GENERACYJNA WERYFIKACJA UGRUNTOWANIA odpowiedzi.

DLACZEGO OSOBNY GUARD (a nie wyższy próg bramki):
Held-out pokazał, że bramka na score rerankera ma sufit — cross-encoder mierzy trafność
TEMATYCZNĄ, nie obecność odpowiedzi (`relevance != answerability`). Dowód: „wydatki obronne
Norwegii jako % PKB" (jest w korpusie) = 0.9910 vs „…Danii…" (nie ma) = 0.9806 — jeden wyraz
różnicy, ten sam fragment, żaden próg tego nie rozdzieli. Błąd nie powstaje w retrievalu, tylko
w GENERACJI (model wziął liczbę Norwegii i przypisał ją Danii), więc kontrolę stawiamy w warstwie
generacji: po wygenerowaniu odpowiedzi sprawdzamy KAŻDĄ tezę przeciw kontekstowi.

SĘDZIA = INNY MODEL niż generator (Gemma vs Qwen) — mniej skorelowanych błędów.

ŚWIADOMY KOSZT (zmierzony, nie założony):
- ~2x latencja (dodatkowe wywołania LLM na każdą tezę),
- guard sam jest stochastyczny — to samo pytanie może raz przejść, a raz nie,
- CICHY FALSE-REFUSAL na poprawnych PARAFRAZACH: gdy odpowiedź trafnie parafrazuje źródło innymi
  słowami, surowy sędzia może powiedzieć NIE i ubić dobrą odpowiedź. Użytkownik zobaczy „nie
  znajduję w źródłach" i wywnioskuje, że korpusu tego nie zawiera — a zawiera. Dlatego prompt
  sędziego JAWNIE dopuszcza parafrazę i wnioskowanie z kontekstu (patrz JUDGE_GROUNDED).

Progiem sterujemy z configu (`VERIFY_MIN_SUPPORTED`) i kalibrujemy go na held-oucie — tak samo
jak próg bramki, mierząc OBIE strony błędu (przecieki i false-refusale).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config
from src.ingest.chunk import split_sentences

# Prompt sędziego — DOBRANY EKSPERYMENTALNIE na dwóch realnych przeciekach z held-outu.
# Wersja odrzucona (łagodna, „czy twierdzenie wynika z kontekstu? TAK/NIE") przepuszczała OBA:
# dla „Danii 1,75% PKB" sędzia znajdował właściwy cytat, ale jako jego podmiot podawał… „Dania",
# czyli przepisywał podmiot Z TWIERDZENIA, nie z kontekstu (confirmation bias). Dlatego:
# 1. DOMYŚLNY werdykt to NIE — odwracamy prior, model musi uzasadnić TAK, a nie odwrotnie.
# 2. Obie realne pułapki są WYMIENIONE WPROST (inny podmiot/rok; oferta-plan podana jako fakt),
#    bo ogólne „bądź surowy" nie działało.
# 3. Wymuszamy jednozdaniowe UZASADNIENIE przed werdyktem — po pierwsze kieruje uwagę modelu na
#    podmiot ZANIM wyda ocenę, po drugie zostaje w logu, więc guard nie jest czarną skrzynką.
#
# Nazwa jest PUBLICZNA (bez podkreślenia), bo harness ewaluacyjny musi umieć porównać ten prompt
# z wariantem — dokładnie tak, jak `answer()` przyjmuje `system` dla promptu generatora. Guard,
# którego promptu nie da się podmienić w pomiarze, jest nieaudytowalny.
JUDGE_GROUNDED = """Jesteś adwersaryjnym weryfikatorem. Twoim zadaniem jest PODWAŻYĆ twierdzenie.
DOMYŚLNY werdykt to NIE. TAK wolno Ci wydać tylko wtedy, gdy w KONTEKŚCIE istnieje zdanie JAWNIE \
łączące podmiot z twierdzenia z podaną wartością.
NAJCZĘSTSZA PUŁAPKA: kontekst zawiera podobną liczbę lub nazwę, ale dotyczącą INNEGO państwa, \
instytucji albo roku. Sprawdź to jako pierwsze.
DRUGA PUŁAPKA: kontekst mówi o ofertach, kandydatach, planach lub możliwościach, a twierdzenie \
przedstawia to jako fakt lub rozstrzygnięcie.
Parafraza jest dozwolona — nie wymagaj dosłownego brzmienia, jeśli treść się zgadza.
Odpowiadasz w dwóch linijkach:
UZASADNIENIE: <jedno zdanie: co dokładnie kontekst mówi i o kim>
WERDYKT: <TAK albo NIE>"""


def to_claims(answer_text: str) -> list[str]:
    """Rozbija odpowiedź na tezy (zdania) — jednostką weryfikacji jest teza, nie cała odpowiedź.

    DLACZEGO per-teza: odpowiedź „Dania ma 1,75% PKB [2], a celu 2% nie osiągnie przed 2033 [4]"
    ma jedną tezę zmyśloną i jedną poprawną. Ocena całości jednym TAK/NIE zgubiłaby tę różnicę.

    KRYTERIUM TEZY: fragment ma choć jeden znak alfanumeryczny. Wcześniej był tu próg 15 znaków
    i to on był drugą połową dziury fail-open (pozycja 3 z `TODO.md`): odsiewał nagłe wielkości,
    więc „120 [1]" (przeciek arytmetyczny, `o05`) i „65% [1]" (POPRAWNA odpowiedź na `g13`) miały
    ZERO tez i przechodziły bez weryfikacji. Kształt mają identyczny — różni je wyłącznie to, czy
    kontekst tę wielkość twierdzi, a to widzi tylko sędzia. Kod, który odsiewa taki fragment,
    nie „upraszcza": odbiera decyzję jedynej warstwie, która ma na nią dowód.
    ZMIERZONY ZASIĘG ZMIANY (`src/eval/ab_claims.py --struktura`, 779 zapisanych odpowiedzi):
    zbiór tez zmienia się w 16 wystąpieniach / 4 unikalnych fragmentach — wszystkie to nagłe
    liczby. Pozostałe 763 mają zbiór tez IDENTYCZNY, z czego 19 to `"[1]"`, gdzie decyzję zmienia
    druga część poprawki (`grounded_fraction`, fail-closed przy pustce) — czyli w pełni
    tożsamościowych jest 744. Dzięki temu zmierzone liczby (faithfulness, A/B promptu, A/B
    chunkingu) zachowują ważność BEZ ponownej generacji: te same tezy = te same wywołania sędziego.
    Fragment bez znaku alfanumerycznego („—", „...") tezą nie jest: nie ma czego weryfikować,
    a wysłanie go do sędziego to spalone wywołanie LLM-a.
    """
    txt = re.sub(r"\[\d+\]", "", answer_text)          # usuń markery cytowań [n]
    # Uwaga: NIE normalizujemy dalej białych znaków (po markerze zostaje spacja przed kropką).
    # Tekst tezy jest wejściem promptu sędziego, więc każda kosmetyka zmienia WSZYSTKIE tezy
    # i unieważnia porównywalność z zapisanymi pomiarami. Osobna zmiana, osobny pomiar.
    return [s for s in split_sentences(txt) if re.search(r"\w", s)]


def judge_grounded(context: str, claim: str, system: str | None = None) -> tuple[bool, str]:
    """Pyta model-sędziego, czy teza jest poparta kontekstem. Zwraca (werdykt, uzasadnienie).

    Import lokalny — unika cyklu (answer.py importuje verify.py, a wywołanie LLM mieszka w answer.py).
    Brak linii WERDYKT traktujemy jako NIE: przy guardzie anty-halucynacyjnym niejasna odpowiedź
    sędziego ma blokować („fail-closed"), a nie przepuszczać.
    """
    from src.generate.answer import call_ollama

    p = f"KONTEKST:\n{context}\n\nTWIERDZENIE: {claim}\n\nWeryfikuj."
    out = call_ollama(p, system=system or JUDGE_GROUNDED, model=config.JUDGE_MODEL)
    upper = out.upper()
    verdict = "TAK" in upper.split("WERDYKT:")[-1][:12] if "WERDYKT:" in upper else False
    reason = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                   if l.upper().startswith("UZASADNIENIE")), out[:160])
    return verdict, reason


NO_CLAIM_REASON = ("odpowiedź nie zawiera żadnej treści do weryfikacji "
                   "(po usunięciu cytowań [n] zostaje sam marker albo pustka)")


def grounded_fraction(context: str, answer_text: str,
                      system: str | None = None) -> tuple[float, list[dict]]:
    """Zwraca (udział tez popartych kontekstem, lista {claim, grounded, reason}).

    BRAK TEZ = 0.0, czyli BLOKADA (fail-closed). Wcześniej było tu 1.0 („guard nie ma czego
    zakwestionować") i to była DZIURA, nie ustępstwo: wyjścia `"[1]"` (sam marker cytowania)
    przechodziły przez wszystkie trzy guardy do użytkownika. Zmierzone nie tylko w odrzuconym
    wariancie promptu (9 wyjść na pułapkach), ale i w PRODUKCYJNYM przebiegu — `g25` w
    `results/backend_answers_ollama.json` ma `answer="[1]"` i `grounded_fraction=1.0`.
    Po zmianie kryterium tezy w `to_claims()` „zero tez" znaczy już tylko jedno: po usunięciu
    cytowań w odpowiedzi NIE MA TREŚCI. Takiego wyjścia nie ma sensu publikować nigdy — nie jest
    ani odpowiedzią, ani odmową, a użytkownikowi wygląda jak uszkodzony cytat.
    Krótkie POPRAWNE odpowiedzi („65% [1]" na `g13`) ta reguła NIE tyka, bo mają teraz jedną tezę
    i idą do sędziego — to jest właśnie różnica między tą poprawką a jednolinijkowym
    „brak tez -> blokuj", które ubijało `g13` (pomiar: `src/eval/ab_claims.py`).

    Zwracamy przy tym SYNTETYCZNY werdykt z powodem, a nie pustą listę: blokada bez uzasadnienia
    zamieniłaby guard w czarną skrzynkę (UI pokazuje `verdicts` w panelu przebiegu).
    """
    claims = to_claims(answer_text)
    if not claims:
        return 0.0, [{"claim": answer_text.strip(), "grounded": False, "reason": NO_CLAIM_REASON}]
    verdicts = []
    for c in claims:
        ok, reason = judge_grounded(context, c, system=system)
        verdicts.append({"claim": c, "grounded": ok, "reason": reason})
    return sum(v["grounded"] for v in verdicts) / len(verdicts), verdicts
