"""
tests/test_corpus.py
Etap 2 UI: KORPUS JAKO PARAMETR retrievalu. Bez GPU, bez Qdranta, bez pliku chunków.

CO TU JEST BRONIONE (i dlaczego to nie są testy „na zapas"):
1. **Leniwość.** `resolve()` nie może czytać `chunks.jsonl` ani ładować modelu — na tym stoi
   uruchamialność całego zestawu testów w venvie bez torcha (patrz `requirements-dev.txt`).
   Kluczowanie cache'ów korpusem to dokładnie ta zmiana, która najłatwiej zamienia leniwy
   singleton w eager (bo „przecież trzeba wiedzieć, co jest w cache'u").
2. **Izolacja korpusów.** Dwa korpusy MUSZĄ mieć osobny indeks BM25 i osobną mapę chunków.
   Wspólny cache = odpowiedzi o wgranym PDF-ie cytujące korpus OSW (albo odwrotnie) — błąd
   cichy, bo trafienia wyglądają poprawnie.
3. **Czyszczenie.** Po `drop()` cache MUSI zniknąć, inaczej „ulotna" kolekcja żyje dalej
   w pamięci procesu i drugi upload widziałby chunki pierwszego.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config
from src.retrieve import bm25 as bm25_mod
from src.retrieve import corpus as corpus_mod
from src.retrieve import hybrid as hybrid_mod
from src.retrieve import search as search_mod
from src.retrieve.corpus import Corpus, UnknownCorpus


def _rec(cid: str, source: str, text: str, index: int = 0) -> dict:
    """Rekord chunka w kształcie, jaki produkuje `chunk.py` (id, source, chunk_index, text)."""
    return {"id": cid, "source": source, "chunk_index": index, "text": text}


@pytest.fixture(autouse=True)
def czysty_rejestr():
    """Każdy test startuje z pustym rejestrem i pustymi cache'ami — stan globalny procesu
    serwera nie może przeciekać między testami (ani między uploadami w produkcji)."""
    corpus_mod.reset()
    yield
    corpus_mod.reset()


# --- korpus bazowy: czytany z configu, ale LENIWIE ---------------------------------------

def test_domyslny_korpus_bierze_kolekcje_z_configu():
    c = corpus_mod.resolve(None)

    assert c.name == corpus_mod.DEFAULT
    assert c.collection == config.QDRANT_COLLECTION
    assert c.measured is True          # to jest korpus, na którym stoją liczby z README


def test_resolve_nie_czyta_pliku_chunkow(monkeypatch, tmp_path):
    """Gdyby `resolve()` wczytywało rekordy, sam import warstwy serwującej wymagałby
    `chunks.jsonl` — a testy kontraktowe działają bez danych i bez GPU."""
    monkeypatch.setattr(config, "CHUNKS_PATH", tmp_path / "nie-ma-mnie.jsonl")

    c = corpus_mod.resolve(None)       # nie wolno wybuchnąć

    with pytest.raises(FileNotFoundError):
        c.load_records()               # dopiero PRAWDZIWE użycie sięga po plik


def test_nieznany_korpus_daje_wlasny_wyjatek():
    """Warstwa HTTP musi umieć odróżnić „nie ma takiego korpusu" (ulotny padł razem z procesem)
    od awarii serwera — inaczej po restarcie użytkownik dostaje 500 zamiast podpowiedzi."""
    with pytest.raises(UnknownCorpus):
        corpus_mod.resolve("kolekcja-widmo")


def test_register_i_drop():
    corpus_mod.register(Corpus(name="upload", collection="demo_upload", records=[]))
    assert corpus_mod.resolve("upload").collection == "demo_upload"

    corpus_mod.drop("upload")
    with pytest.raises(UnknownCorpus):
        corpus_mod.resolve("upload")


def test_drop_nieistniejacego_korpusu_nie_wybucha():
    """Zamiatacz przy starcie/zamknięciu procesu woła `drop()` „na wszelki wypadek" — nie może
    wymagać wiedzy o tym, czy coś w ogóle było."""
    corpus_mod.drop("upload")


# --- BM25: cache kluczowany korpusem, nadal leniwy ---------------------------------------

def test_bm25_jest_budowany_z_rekordow_w_pamieci():
    """Wgrany PDF nie dotyka dysku, więc jego BM25 musi dać się zbudować z listy rekordów —
    bez `CHUNKS_PATH`."""
    up = corpus_mod.register(Corpus(
        name="upload", collection="demo_upload",
        records=[_rec("u__0", "raport-obcy", "Balticconnector to gazociąg fińsko-estoński.")]))

    trafienia = bm25_mod.get_bm25(up).search("Balticconnector", top_k=1)

    assert trafienia[0][0] == "u__0"


def test_dwa_korpusy_maja_osobne_indeksy_bm25():
    a = corpus_mod.register(Corpus(name="a", collection="ka", records=[
        _rec("a__0", "a", "Arktyczny kwartet NATO.")]))
    b = corpus_mod.register(Corpus(name="b", collection="kb", records=[
        _rec("b__0", "b", "Balticconnector i gazociąg.")]))

    assert bm25_mod.get_bm25(a).ids == ["a__0"]
    assert bm25_mod.get_bm25(b).ids == ["b__0"]


def test_bm25_tego_samego_korpusu_nie_jest_budowany_dwa_razy():
    """Sedno kluczowania cache'a: dokładamy KLUCZ, nie rezygnujemy z cache'a. Indeks BM25 nad
    1509 chunkami budowany przy każdym pytaniu skasowałby zysk z długo żyjącego procesu."""
    a = corpus_mod.register(Corpus(name="a", collection="ka", records=[_rec("a__0", "a", "tekst")]))

    assert bm25_mod.get_bm25(a) is bm25_mod.get_bm25(a)


def test_drop_usuwa_indeks_bm25_z_pamieci():
    """Ulotność musi być STRUKTURALNA: po skasowaniu korpusu jego chunki nie mogą zostać
    w cache'u procesu, bo drugi upload zobaczyłby treść pierwszego."""
    corpus_mod.register(Corpus(name="upload", collection="demo_upload",
                               records=[_rec("u__0", "pierwszy", "Arktyka.")]))
    bm25_mod.get_bm25(corpus_mod.resolve("upload"))

    corpus_mod.drop("upload")
    drugi = corpus_mod.register(Corpus(name="upload", collection="demo_upload",
                                       records=[_rec("u__0", "drugi", "Bałtyk.")]))

    assert bm25_mod.get_bm25(drugi).search("Bałtyk", top_k=1)[0][1] > 0
    assert bm25_mod.get_bm25(drugi).search("Arktyka", top_k=1)[0][1] == 0.0


# --- mapa chunków w hybrydzie ------------------------------------------------------------

def test_mapa_chunkow_jest_per_korpus():
    up = corpus_mod.register(Corpus(name="upload", collection="demo_upload",
                                    records=[_rec("u__3", "obcy", "treść", index=3)]))

    mapa = hybrid_mod._get_chunks(up)

    assert mapa["u__3"]["source"] == "obcy"
    assert mapa["u__3"]["chunk_index"] == 3
    assert hybrid_mod._get_chunks(up) is mapa      # nadal cache, tylko kluczowany


# --- przepływ korpusu przez retrieval ----------------------------------------------------

def test_hybryda_przekazuje_korpus_do_dense_i_cytuje_jego_chunki(monkeypatch):
    """Bez tego wgrany PDF trafiłby do BM25 (rekordy w pamięci), ale dense szukałby dalej
    w kolekcji OSW — i odpowiedź mieszałaby dwa korpusy."""
    up = corpus_mod.register(Corpus(name="upload", collection="demo_upload",
                                    records=[_rec("u__0", "obcy-raport", "Zdanie z obcego PDF-a.")]))
    dense = create_autospec(search_mod.search, return_value=[])
    monkeypatch.setattr(hybrid_mod, "dense_search", dense)
    monkeypatch.setattr(hybrid_mod, "get_bm25", lambda c: SimpleNamespace(
        search=lambda q, n: [("u__0", 3.2)]))

    hits = hybrid_mod.hybrid_search("pytanie", top_k=1, corpus=up)

    assert dense.call_args.kwargs["corpus"] is up
    assert hits[0]["source"] == "obcy-raport"
    assert hits[0]["chunk_id"] == "u__0"


def test_dense_pyta_kolekcje_wskazanego_korpusu(monkeypatch):
    """`search.py` czytał `config.QDRANT_COLLECTION` na sztywno — po parametryzacji musi pytać
    kolekcję KORPUSU, inaczej cała reszta parametryzacji jest dekoracją."""
    zapytane: dict = {}

    class _Wektor:
        def tolist(self):
            return [0.0, 1.0]

    class _Model:
        def encode(self, teksty, normalize_embeddings=None):
            return [_Wektor()]

    class _Klient:
        def query_points(self, collection_name, query, limit):
            zapytane["kolekcja"] = collection_name
            return SimpleNamespace(points=[SimpleNamespace(score=0.9, payload={
                "source": "obcy", "chunk_index": 0, "chunk_id": "u__0", "text": "tekst"})])

    monkeypatch.setattr(search_mod, "_get_model", lambda: _Model())
    monkeypatch.setattr(search_mod, "_get_client", lambda: _Klient())
    up = corpus_mod.register(Corpus(name="upload", collection="demo_upload", records=[]))

    hits = search_mod.search("pytanie", top_k=1, corpus=up)

    assert zapytane["kolekcja"] == "demo_upload"
    assert hits[0]["chunk_id"] == "u__0"


def test_dense_bez_korpusu_pyta_kolekcje_z_configu(monkeypatch):
    """Zgodność w tył: CLI, `run_eval.py` i A/B chunkingu wołają `search()` bez korpusu."""
    zapytane: dict = {}

    class _Wektor:
        def tolist(self):
            return [0.0]

    monkeypatch.setattr(search_mod, "_get_model",
                        lambda: SimpleNamespace(encode=lambda t, normalize_embeddings=None: [_Wektor()]))
    monkeypatch.setattr(search_mod, "_get_client", lambda: SimpleNamespace(
        query_points=lambda collection_name, query, limit: (
            zapytane.update(kolekcja=collection_name) or SimpleNamespace(points=[]))))

    search_mod.search("pytanie", top_k=1)

    assert zapytane["kolekcja"] == config.QDRANT_COLLECTION
