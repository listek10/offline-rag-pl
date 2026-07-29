"""
tests/test_ad_hoc_index.py
Etap 2 UI: indeksowanie wgranego dokumentu do OSOBNEJ, ULOTNEJ kolekcji.
Model embeddingowy i klient Qdranta są PODMIENIONE — test nie dotyka GPU ani bazy.

NAJWAŻNIEJSZA RZECZ W TYM PLIKU: `test_*_nigdy_nie_kasuje_kolekcji_zmierzonej`.
`build_index.py` kasuje kolekcję przed każdym buildem, a etap 2 dokłada drugą ścieżkę, która
tworzy i kasuje kolekcje w czasie działania serwera. Pomyłka w nazwie kolekcji nie daje błędu —
daje CICHO skasowany indeks, na którym stoją liczby recall@1 0.83 / MRR 0.89 z README.
To nie ryzyko niespójnego stanu, to utrata zbioru ewaluacyjnego.

PODZIAŁ PO ZALEŻNOŚCIACH: testy izolacji i orkiestracji lecą wszędzie (czysty stdlib), a testy
`index_records` wymagają `qdrant_client` (buduje `PointStruct`/`VectorParams`) i w venvie dev
się pomijają — jak testy PDF-a bez pymupdf. Własność krytyczna (nietykalność kolekcji zmierzonej)
jest broniona po stronie, która NIE wymaga bazy, więc nie znika z CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.index import ad_hoc as ad_hoc_mod
from src.index import build_index as build_mod
from src.retrieve import corpus as corpus_mod

wymaga_qdrant = pytest.mark.skipif(
    importlib.util.find_spec("qdrant_client") is None,
    reason="qdrant-client niedostępny — kształt punktów sprawdza env projektu")


class _Wektor(list):
    """Wektor „jak z numpy" — z metodą `.tolist()`, której używa produkcyjny kod.
    Atrapa MUSI mieć ten interfejs, inaczej test przechodziłby dla kodu, który w produkcji
    (gdzie `encode(convert_to_numpy=True)` zwraca ndarray) zachowuje się inaczej."""

    def tolist(self):
        return list(self)


class _Model:
    """Atrapa BGE-M3: wektory o wymiarze z configu, żeby asercja wymiaru miała sens."""

    def encode(self, texts, batch_size=None, normalize_embeddings=None,
               show_progress_bar=False, convert_to_numpy=True):
        return [_Wektor([0.0] * config.EMBED_DIM) for _ in texts]


class _Klient:
    """Atrapa klienta Qdranta: zapisuje, co skasowano, utworzono i wgrano."""

    def __init__(self, istniejace: set[str] | None = None):
        self.istniejace = set(istniejace or ())
        self.skasowane: list[str] = []
        self.utworzone: list[str] = []
        self.upserty: list[tuple[str, list]] = []
        self.wymiar: int | None = None

    def collection_exists(self, name):
        return name in self.istniejace

    def delete_collection(self, name):
        self.skasowane.append(name)
        self.istniejace.discard(name)

    def create_collection(self, collection_name, vectors_config):
        self.utworzone.append(collection_name)
        self.istniejace.add(collection_name)
        self.wymiar = vectors_config.size

    def upsert(self, collection_name, points):
        self.upserty.append((collection_name, points))

    def count(self, collection_name):
        return SimpleNamespace(count=sum(len(p) for c, p in self.upserty if c == collection_name))


def _rekordy(n: int, source: str = "obcy") -> list[dict]:
    return [{"id": f"{source}__{i}", "source": source, "chunk_index": i,
             "text": f"Zdanie numer {i}."} for i in range(n)]


def _statystyki(source: str = "obcy", chunks: int = 3) -> dict:
    return {"source": source, "pages": 4, "chars": 900, "chunks": chunks, "chars_per_page": 225}


@pytest.fixture(autouse=True)
def czysty_rejestr():
    corpus_mod.reset()
    yield
    corpus_mod.reset()


class _SpyIndex:
    """Atrapa `index_records`: zapamiętuje, DO KTÓREJ kolekcji poszły rekordy."""

    def __init__(self):
        self.wywolania: list[tuple[list[dict], str]] = []

    def __call__(self, records, collection, *, model, client):
        self.wywolania.append((records, collection))
        client.create_collection(collection_name=collection,
                                 vectors_config=SimpleNamespace(size=config.EMBED_DIM))
        return len(records)


def _podmien(monkeypatch, klient=None, records=None, stats=None) -> tuple[_Klient, _SpyIndex]:
    klient = klient or _Klient()
    spy = _SpyIndex()
    monkeypatch.setattr(ad_hoc_mod, "_client", lambda: klient)
    monkeypatch.setattr(ad_hoc_mod, "_model", lambda: _Model())
    monkeypatch.setattr(ad_hoc_mod, "index_records", spy)
    monkeypatch.setattr(ad_hoc_mod, "prepare_upload",
                        lambda data, filename: (records if records is not None else _rekordy(3),
                                                stats or _statystyki()))
    return klient, spy


# --- wspólna fabryka punktów: jeden opis payloadu dla obu ścieżek ------------------------

@wymaga_qdrant
def test_index_records_zapisuje_payload_do_cytowania():
    """Payload musi mieć dokładnie te pola, które czyta `search.py` (chunk_id/source/
    chunk_index/text). Po wyciągnięciu z `main()` ta funkcja jest JEDYNYM miejscem, które je
    wypisuje — używa jej build wsadowy i upload, więc nie ma dwóch opisów jednego kształtu."""
    klient, model = _Klient(), _Model()

    n = build_mod.index_records(_rekordy(2), "kolekcja-x", model=model, client=klient)

    assert n == 2
    kolekcja, punkty = klient.upserty[0]
    assert kolekcja == "kolekcja-x"
    assert punkty[0].payload == {"chunk_id": "obcy__0", "source": "obcy",
                                 "chunk_index": 0, "text": "Zdanie numer 0."}


@wymaga_qdrant
def test_index_records_tworzy_kolekcje_od_nowa():
    klient = _Klient(istniejace={"kolekcja-x"})

    build_mod.index_records(_rekordy(1), "kolekcja-x", model=_Model(), client=klient)

    assert klient.skasowane == ["kolekcja-x"]      # świeży indeks przy każdym buildzie
    assert klient.utworzone == ["kolekcja-x"]
    assert klient.wymiar == config.EMBED_DIM


@wymaga_qdrant
def test_index_records_wgrywa_partiami():
    """Jeden request z tysiącami punktów potrafi się wywrócić na limitach — dzielimy na partie.
    Test broni tej właściwości po wyciągnięciu funkcji z `main()`."""
    klient = _Klient()

    build_mod.index_records(_rekordy(600), "kolekcja-x", model=_Model(), client=klient)

    assert len(klient.upserty) > 1
    assert sum(len(p) for _, p in klient.upserty) == 600


# --- ulotna kolekcja: budowa --------------------------------------------------------------

def test_build_rejestruje_korpus_ulotny_i_zwraca_statystyki(monkeypatch):
    klient, spy = _podmien(monkeypatch)

    stat = ad_hoc_mod.build(b"%PDF-udawany", "obcy.pdf")

    assert stat["chunks"] == 3 and stat["points"] == 3
    assert stat["collection"] == config.UPLOAD_COLLECTION
    assert stat["corpus"] == corpus_mod.AD_HOC
    c = corpus_mod.resolve(corpus_mod.AD_HOC)
    assert c.collection == config.UPLOAD_COLLECTION
    assert c.measured is False              # guardy nie przenoszą gwarancji na obcy dokument
    assert c.records is not None            # rekordy w PAMIĘCI — plik nie dotknął dysku
    assert c.meta["pages"] == 4


def test_build_indeksuje_do_kolekcji_demo_a_nie_zmierzonej(monkeypatch):
    _, spy = _podmien(monkeypatch)

    ad_hoc_mod.build(b"%PDF", "obcy.pdf")

    (_, kolekcja), = spy.wywolania
    assert kolekcja == config.UPLOAD_COLLECTION
    assert kolekcja != config.QDRANT_COLLECTION


def test_build_odmawia_gdy_kolekcja_demo_rowna_sie_zmierzonej(monkeypatch):
    """Gdyby ktoś ustawił `QDRANT_COLLECTION=demo_upload` (env jest do tego dostępny — A/B
    chunkingu z etapu 8), upload przebudowałby kolekcję zmierzoną: `index_records` kasuje
    kolekcję przed zapisem. Kod ODMAWIA pracy zamiast ufać, że nikt tego nie zrobi."""
    monkeypatch.setattr(config, "UPLOAD_COLLECTION", config.QDRANT_COLLECTION)
    _, spy = _podmien(monkeypatch)

    with pytest.raises(RuntimeError):
        ad_hoc_mod.build(b"%PDF", "obcy.pdf")

    assert spy.wywolania == []              # nic nie poszło do Qdranta


def test_drugi_upload_zastepuje_pierwszy(monkeypatch):
    """„Ulotny" znaczy też: jeden dokument demo naraz. Drugi upload musi podmienić rekordy
    w rejestrze (i cache'e retrievalu), inaczej odpowiedzi mieszałyby dwa dokumenty."""
    _podmien(monkeypatch)
    ad_hoc_mod.build(b"%PDF", "pierwszy.pdf")

    monkeypatch.setattr(ad_hoc_mod, "prepare_upload",
                        lambda data, filename: (_rekordy(1, "drugi"), _statystyki("drugi", 1)))
    ad_hoc_mod.build(b"%PDF", "drugi.pdf")

    c = corpus_mod.resolve(corpus_mod.AD_HOC)
    assert [r["source"] for r in c.records] == ["drugi"]
    assert c.meta["source"] == "drugi"


# --- ulotna kolekcja: zamiatanie ---------------------------------------------------------

def test_zamiatacz_kasuje_kolekcje_demo(monkeypatch):
    klient, _ = _podmien(monkeypatch, _Klient(istniejace={config.UPLOAD_COLLECTION,
                                                          config.QDRANT_COLLECTION}))
    corpus_mod.register(corpus_mod.Corpus(name=corpus_mod.AD_HOC,
                                          collection=config.UPLOAD_COLLECTION, records=[]))

    ad_hoc_mod.drop()

    assert klient.skasowane == [config.UPLOAD_COLLECTION]
    assert config.QDRANT_COLLECTION in klient.istniejace
    assert corpus_mod.ad_hoc() is None


def test_zamiatacz_nigdy_nie_kasuje_kolekcji_zmierzonej(monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_COLLECTION", config.QDRANT_COLLECTION)
    klient, _ = _podmien(monkeypatch, _Klient(istniejace={config.QDRANT_COLLECTION}))

    with pytest.raises(RuntimeError):
        ad_hoc_mod.drop()

    assert klient.skasowane == []


def test_zamiatacz_dziala_gdy_nic_nie_bylo(monkeypatch):
    """Wołany przy KAŻDYM starcie procesu (po `kill -9` mogła zostać sierota) — nie może
    wymagać, żeby cokolwiek istniało."""
    klient, _ = _podmien(monkeypatch)

    ad_hoc_mod.drop()

    assert klient.skasowane == []


def test_zamiatacz_nie_wywala_sie_gdy_qdrant_lezy(monkeypatch):
    """Kasowanie ulotnej kolekcji nie może przewrócić ani startu, ani zamykania serwera —
    a przy starcie baza może jeszcze nie odpowiadać."""
    def boom():
        raise ConnectionError("Qdrant nie odpowiada")

    monkeypatch.setattr(ad_hoc_mod, "_client", boom)

    ad_hoc_mod.drop()      # nie wolno rzucić


def test_zamiatacz_czysci_korpus_takze_gdy_qdrant_lezy(monkeypatch):
    """Rejestr korpusów żyje w pamięci procesu i musi zostać wyczyszczony niezależnie od bazy —
    inaczej UI dalej oferowałby korpus, którego wektorów nikt nie potwierdził."""
    corpus_mod.register(corpus_mod.Corpus(name=corpus_mod.AD_HOC,
                                          collection=config.UPLOAD_COLLECTION, records=[]))
    monkeypatch.setattr(ad_hoc_mod, "_client", lambda: (_ for _ in ()).throw(ConnectionError("nie ma")))

    ad_hoc_mod.drop()

    assert corpus_mod.ad_hoc() is None
