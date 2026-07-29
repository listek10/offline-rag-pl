"""
tests/test_verify_claims.py
Guard #3, dziura fail-open: co jest TEZĄ i co się dzieje, gdy tez nie ma (pozycja 3 z TODO.md).

DLACZEGO TE TESTY, A NIE INNE — każdy z nich zamyka JEDEN zmierzony błąd z etapu 11/12:
1. `[1]` (sama etykieta cytowania) przechodziło przez wszystkie trzy guardy do użytkownika,
   bo `to_claims()` nie miało czego zwrócić, a `grounded_fraction()` czytało to jako 1.0.
   Zmierzone na `g25` w produkcyjnym przebiegu backendów — więc nie jest to defekt harnessu.
2. `120 [1]` (przeciek arytmetyczny na pytaniu spoza korpusu) i `65% [1]` (POPRAWNA odpowiedź
   na `g13`) mają IDENTYCZNY kształt. Kod nie ma prawa ich rozdzielić — obie muszą trafić do
   sędziego. Test pilnuje, że nagła wielkość jest tezą, a nie jest odsiewana przed weryfikacją.
3. Blokada bez uzasadnienia jest w tym projekcie zakazana (guard ma być audytowalny), więc
   „brak tez -> blokuj" musi zwrócić werdykt z powodem, a nie pustą listę.

Bez GPU i bez LLM-a: sędzia jest podmieniany, a tam, gdzie decyzja jest deterministyczna,
podstawiamy sędziego-wybuchowego — jeśli test przejdzie, guard NA PEWNO nie pytał modelu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generate import verify
from src.generate.verify import grounded_fraction, to_claims

# Prawdziwe odpowiedzi z `results/` (nie wymyślone) -> oczekiwane tezy.
# Ostatni wiersz to kontrola „operacji tożsamościowej": normalna odpowiedź musi dać DOKŁADNIE
# to samo co przed zmianą, bo inaczej unieważniamy zmierzone liczby (recall/faithfulness).
REALNE = [
    ("[1]", []),                                    # g25 — zero treści po usunięciu cytowania
    ("   ", []),                                    # pusta generacja
    ("[1] [2]", []),                                # same etykiety
    ("65% [1]", ["65%"]),                           # g13 — POPRAWNA krótka odpowiedź
    ("120 [1]", ["120"]),                           # o05 — przeciek arytmetyczny (15 x 8)
    ("444 [5]", ["444"]),                           # h27 — przeciek arytmetyczny (37 x 12)
    # Spacja przed kropką jest ARTEFAKTEM usuwania markera `[n]` i zostaje TAK, JAK BYŁA.
    # Normalizacja białych znaków zmieniłaby tekst KAŻDEJ tezy, czyli każdy prompt sędziego —
    # a wtedy dowód „poza 16 wypisanymi odpowiedziami zmiana jest operacją tożsamościową"
    # przestałby obowiązywać i unieważniłbym zmierzone faithfulness. Osobna zmiana, osobny pomiar.
    ("Arktyczny kwartet NATO składa się z USA, Kanady, Danii i Norwegii [1].",
     ["Arktyczny kwartet NATO składa się z USA, Kanady, Danii i Norwegii ."]),
]


@pytest.mark.parametrize("odpowiedz,tezy", REALNE)
def test_tezy_z_realnych_odpowiedzi(odpowiedz, tezy):
    assert to_claims(odpowiedz) == tezy


def test_nagla_wielkosc_jest_teza_a_nie_znika():
    """Sedno poprawki: `65%` i `120` różnią się WYŁĄCZNIE relacją do kontekstu, nie kształtem.

    Stary próg (>15 znaków) odsiewał oba, więc odpowiedź „nie miała tez" i przechodziła
    fail-openem. Nowa reguła oddaje decyzję sędziemu — jedynej warstwie, która widzi kontekst.
    """
    assert to_claims("65% [1]") == to_claims("65% [2]") == ["65%"]
    assert len(to_claims("120 [1]")) == 1


def test_sama_interpunkcja_nie_jest_teza():
    """Fragment bez ani jednego znaku alfanumerycznego nie jest twierdzeniem — nie ma czego
    weryfikować, a wysłanie go do sędziego to spalone wywołanie LLM-a na każdą taką odpowiedź."""
    assert to_claims("— [1]") == []
    assert to_claims("...") == []


def test_brak_tez_blokuje_i_nie_pyta_sedziego(monkeypatch):
    """Fail-CLOSED przy zerze tez + dowód, że decyzja jest deterministyczna (sędzia nietknięty)."""
    monkeypatch.setattr(verify, "judge_grounded",
                        lambda *a, **k: pytest.fail("guard nie ma o co pytać sędziego"))
    frac, verdicts = grounded_fraction("KONTEKST: cokolwiek", "[1]")
    assert frac == 0.0                       # < VERIFY_MIN_SUPPORTED przy każdym progu > 0
    assert len(verdicts) == 1
    assert verdicts[0]["grounded"] is False
    assert verdicts[0]["reason"]             # blokada MUSI mieć uzasadnienie (audytowalność)


def test_pelna_petla_blokuje_sam_marker_cytowania(monkeypatch):
    """WIRING: ten sam przypadek, ale przez `answer()` z PRAWDZIWYM guardem #3.

    Testy w `test_answer_events.py` podmieniają `grounded_fraction`, więc sprawdzają przepływ
    zdarzeń, a nie samą regułę. Tu podmieniony jest tylko retrieval i generator — gdyby ktoś
    przywrócił fail-open w guardzie, tamte testy zostałyby zielone, a TEN spada. Zmierzony
    przypadek produkcyjny: `g25` z `results/backend_answers_ollama.json` (`answer="[1]"`).
    """
    from unittest.mock import create_autospec

    from src.generate import answer as answer_mod
    from src.retrieve.hit import make_hit, set_rerank_score
    from src.retrieve.hybrid import hybrid_search

    hit = make_hit(score=0.01, source="raport", chunk_index=7, chunk_id="raport__7",
                   text="Deklarację z Ilulissat podpisano w 2008 r.")
    set_rerank_score(hit, 0.9)
    monkeypatch.setattr(answer_mod, "hybrid_search", create_autospec(hybrid_search,
                                                                    return_value=[hit]))
    monkeypatch.setattr(answer_mod, "call_llm", lambda *a, **k: "[1]")
    monkeypatch.setattr(verify, "judge_grounded",
                        lambda *a, **k: pytest.fail("blokada ma być deterministyczna"))

    result = answer_mod.answer("Kiedy podpisano deklarację z Ilulissat?", verify=True)

    assert result["blocked_by"] == "verify"
    assert result["rejected_answer"] == "[1]"          # odrzucone wyjście zostaje do audytu
    assert result["grounding"][0]["grounded"] is False


def test_odpowiedz_z_tezami_dziala_jak_dotad(monkeypatch):
    """Kontrola regresji: ścieżka z tezami nie zmienia się ani o krok (ten sam ułamek, te same
    wywołania sędziego) — inaczej zmiana unieważniłaby zmierzone faithfulness i A/B."""
    monkeypatch.setattr(verify, "judge_grounded",
                        lambda ctx, claim, system=None: (claim.startswith("Pierwsza"), "bo tak"))
    frac, verdicts = grounded_fraction("ctx", "Pierwsza teza o czymś [1]. Druga teza o czymś [2].")
    assert [v["claim"] for v in verdicts] == ["Pierwsza teza o czymś .", "Druga teza o czymś ."]
    assert frac == 0.5
