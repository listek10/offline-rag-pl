"""
src/retrieve/corpus.py
JEDNO miejsce, które wie, „NA CZYM szukamy" — korpus jako PARAMETR retrievalu (etap 2 UI).

DLACZEGO TO ISTNIEJE: do etapu 1 korpus był wpisany na sztywno w trzech niezależnych miejscach
— kolekcja Qdranta w `search.py`, plik chunków w `get_bm25()` i drugi raz w `_get_chunks()`.
Żeby wgrany ad hoc PDF przeszedł DOKŁADNIE TĄ SAMĄ ścieżką (dense + BM25 + RRF + rerank + trzy
guardy), te trzy miejsca muszą pytać o to samo: „który korpus?".

DWIE WŁASNOŚCI, KTÓRYCH TEN MODUŁ PILNUJE (obie zapisane w testach `tests/test_corpus.py`):

1. **Leniwość zostaje leniwością.** Cache'e BM25 i mapy chunków dostają KLUCZ (nazwę korpusu),
   a nie zamieniają się w eager. Na leniwości stoi możliwość uruchomienia całego zestawu testów
   w venvie bez torcha i bez danych (patrz `requirements-dev.txt`) — a to jest własność, którą
   łatwo stracić przy okazji „porządkowania cache'ów".

2. **Ulotność jest STRUKTURALNA, nie dyscyplinarna.** Kasowanie korpusu (`drop`) zamiata też jego
   wpisy we wszystkich cache'ach — inaczej „ulotny" dokument demo żyłby dalej w pamięci procesu
   i drugi upload widziałby chunki pierwszego. Dlatego cache'e nie są tu prywatnymi słownikami
   swoich modułów, a tworzy je `new_cache()`: rejestr wie, co ma zamieść.

Korpus bazowy (`DEFAULT`) NIE jest tworzony przy imporcie, tylko przy pierwszym `resolve()` —
dzięki temu nadpisanie `QDRANT_COLLECTION`/`CHUNKS_PATH` środowiskiem (A/B chunkingu, etap 8)
działa jak dotąd, a testy mogą je podmienić monkeypatchem.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

DEFAULT = "base"       # korpus zmierzony: kolekcja + chunki z configu
AD_HOC = "upload"      # korpus ulotny: dokument wgrany w UI (żyje tylko w tym procesie)


class UnknownCorpus(LookupError):
    """Korpus o tej nazwie nie istnieje.

    Osobny typ, bo warstwa HTTP MUSI odróżnić „ulotny korpus padł razem z procesem" (użytkownik
    ma wgrać PDF ponownie — 400) od awarii serwera (500). Bez tego po restarcie serwera
    przeglądarka z zapamiętanym przełącznikiem dostawałaby traceback.
    """


@dataclass
class Corpus:
    """Opis korpusu: gdzie są wektory (kolekcja Qdranta) i gdzie rekordy chunków.

    `records` podane wprost = korpus W PAMIĘCI (wgrany PDF — plik nie dotyka dysku).
    `records=None` = wczytaj z `chunks_path` (albo `config.CHUNKS_PATH`) przy PIERWSZYM użyciu.
    """

    name: str
    collection: str
    records: list[dict] | None = None
    chunks_path: Path | None = None
    label: str = ""
    measured: bool = False          # czy na tym korpusie stoją liczby z README
    meta: dict = field(default_factory=dict)   # statystyki uploadu (strony, znaki) dla UI

    def load_records(self) -> list[dict]:
        """Rekordy chunków (`id`, `source`, `chunk_index`, `text`). Leniwie i tylko raz."""
        if self.records is None:
            path = self.chunks_path or config.CHUNKS_PATH
            self.records = [json.loads(line) for line in path.open(encoding="utf-8")]
        return self.records


_registry: dict[str, Corpus] = {}
_caches: list[dict] = []


def new_cache() -> dict:
    """Zwraca słownik-cache kluczowany NAZWĄ korpusu i zapisuje go do zamiatania przy `drop()`.

    Modules retrievalu (`bm25.py`, `hybrid.py`) trzymają swoje cache'e właśnie tak, bo
    „skasuj korpus" musi znaczyć „skasuj też to, co o nim wiemy" — w JEDNYM miejscu, a nie
    przez pamiętanie o tym w każdym module osobno.
    """
    cache: dict = {}
    _caches.append(cache)
    return cache


def register(corpus: Corpus) -> Corpus:
    """Dokłada korpus do rejestru (nadpisując poprzedni o tej nazwie) i zwraca go."""
    drop(corpus.name)          # nowy dokument pod tą samą nazwą => stare cache'e do kosza
    _registry[corpus.name] = corpus
    return corpus


def drop(name: str) -> None:
    """Usuwa korpus i WSZYSTKO, co o nim zapamiętały cache'e. Bezpieczne, gdy nic nie było —
    zamiatacz przy starcie procesu woła to „na wszelki wypadek" (po `kill -9` mogła zostać sierota)."""
    _registry.pop(name, None)
    for cache in _caches:
        cache.pop(name, None)


def reset() -> None:
    """Czyści rejestr i cache'e — używane w testach, żeby stan globalny nie przeciekał."""
    _registry.clear()
    for cache in _caches:
        cache.clear()


def resolve(corpus: Corpus | str | None = None) -> Corpus:
    """Zamienia nazwę (albo brak nazwy) na korpus. `None` = korpus bazowy, czyli zachowanie
    identyczne jak przed parametryzacją — dzięki temu CLI, `run_eval.py` i `eval_answers.py`
    działają bez zmian."""
    if isinstance(corpus, Corpus):
        return corpus
    name = corpus or DEFAULT
    if name == DEFAULT and DEFAULT not in _registry:
        _registry[DEFAULT] = Corpus(
            name=DEFAULT,
            collection=config.QDRANT_COLLECTION,
            chunks_path=config.CHUNKS_PATH,
            # Etykieta pokazuje PRAWDZIWĄ nazwę kolekcji, a nie ładny opis wpisany z pamięci —
            # przy A/B chunkingu (env `QDRANT_COLLECTION`) UI ma mówić, na czym realnie pyta.
            label=f"korpus zmierzony ({config.QDRANT_COLLECTION})",
            measured=True,
        )
    try:
        return _registry[name]
    except KeyError:
        raise UnknownCorpus(name) from None


def ad_hoc() -> Corpus | None:
    """Aktualnie wgrany korpus ulotny albo `None`. Health czyta to, żeby UI po odświeżeniu
    strony wiedziało, czy przełącznik korpusu ma co pokazywać."""
    return _registry.get(AD_HOC)
