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
    """
    txt = re.sub(r"\[\d+\]", "", answer_text)          # usuń markery cytowań [n]
    return [s for s in split_sentences(txt) if len(s) > 15]


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


def grounded_fraction(context: str, answer_text: str,
                      system: str | None = None) -> tuple[float, list[dict]]:
    """Zwraca (udział tez popartych kontekstem, lista {claim, grounded, reason}).

    Brak tez (np. bardzo krótka odpowiedź) traktujemy jako 1.0 — guard nie ma czego zakwestionować
    i nie powinien wtedy blokować; to świadoma decyzja „fail-open" w tym jednym przypadku.

    ZMIERZONA KONSEKWENCJA (etap 11, `src/eval/ab_prompt.py`): ta furtka jest DZIURĄ, nie tylko
    ustępstwem. `to_claims()` usuwa markery `[n]` i odsiewa fragmenty ≤15 znaków, więc wyjścia
    `"[1]"` (sam cytat) i `"444 [5]"` (arytmetyka z cytatem, zabroniona zasadą 1 promptu) mają
    ZERO tez i przechodzą przez wszystkie trzy guardy do użytkownika. Zmierzone na 9 wyjściach po
    stronie pułapek i 3 przeciekach spoza korpusu. NIE zamykam tego tutaj na wyczucie: „brak tez
    -> blokuj" tyka też krótkich POPRAWNYCH odpowiedzi („ok. 3 mld euro [2]"), więc zmiana wymaga
    własnego pomiaru obu stron błędu (pozycja 3 w `TODO.md`).
    """
    claims = to_claims(answer_text)
    if not claims:
        return 1.0, []
    verdicts = []
    for c in claims:
        ok, reason = judge_grounded(context, c, system=system)
        verdicts.append({"claim": c, "grounded": ok, "reason": reason})
    return sum(v["grounded"] for v in verdicts) / len(verdicts), verdicts
