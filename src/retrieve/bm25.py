"""
src/retrieve/bm25.py
Etap 5: wyszukiwanie LEKSYKALNE BM25 (własna implementacja — bez czarnej skrzynki).

DLACZEGO własna, nie biblioteka: BM25 to klasyka i chcę móc obronić KAŻDY człon wzoru na
rozmowie. Zero nowych zależności (czysty stdlib).

WZÓR (BM25 Okapi) — score chunka d dla zapytania q:
    score(d,q) = Σ_{t∈q} idf(t) · [ f(t,d)·(k1+1) ] / [ f(t,d) + k1·(1 - b + b·|d|/avgdl) ]
gdzie:
    f(t,d)  – ile razy term t występuje w chunku d,
    |d|     – długość chunka (w tokenach), avgdl – średnia długość,
    idf(t)  = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )   (n(t) = w ilu chunkach jest t),
    k1      – nasycenie częstością (im większe f, tym mniejszy przyrost score) — BM25_K1,
    b       – normalizacja długości (dłuższy dokument nie wygrywa samą długością) — BM25_B.

MOCNE strony BM25: dosłowne terminy/nazwy własne (Balticconnector, NORDEFCO) — tam gdzie dense
jest rozmyty. SŁABE: brak synonimów/znaczenia; wrażliwe na polską fleksję (prosta tokenizacja
nie skleja "bezpieczeństwo"/"bezpieczeństwa") — świadoma, odnotowana wada.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config

_TOKEN = re.compile(r"\w+", re.UNICODE)  # \w łapie polskie znaki (ą, ć, ż...)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[dict], k1: float, b: float):
        self.k1, self.b = k1, b
        self.ids = [d["id"] for d in docs]
        corpus = [tokenize(d["text"]) for d in docs]
        self.N = len(corpus)
        self.doc_len = [len(t) for t in corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0
        self.tf = [Counter(t) for t in corpus]                 # częstości termów w chunku
        df: Counter = Counter()                                 # w ilu chunkach występuje term
        for toks in corpus:
            df.update(set(toks))
        # idf w wariancie z +1 pod logarytmem — zawsze nieujemne.
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = [0.0] * self.N
        for t in q:
            idf = self.idf.get(t)
            if idf is None:            # term spoza korpusu — nic nie wnosi
                continue
            for i in range(self.N):
                f = self.tf[i].get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = sorted(range(self.N), key=lambda i: s[i], reverse=True)[:top_k]
        return [(self.ids[i], s[i]) for i in order]


# Leniwy singleton — budujemy indeks raz z chunks.jsonl.
_bm25: BM25 | None = None


def get_bm25() -> BM25:
    global _bm25
    if _bm25 is None:
        docs = [json.loads(line) for line in config.CHUNKS_PATH.open(encoding="utf-8")]
        _bm25 = BM25(docs, k1=config.BM25_K1, b=config.BM25_B)
    return _bm25
