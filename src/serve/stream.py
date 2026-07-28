"""
src/serve/stream.py
Mostek: funkcja BLOKUJĄCA, która raportuje postęp CALLBACKIEM  ->  ITERATOR zdarzeń.

DLACZEGO to w ogóle potrzebne (sedno tego pliku):
`StreamingResponse` w FastAPI musi z czegoś CIĄGNĄĆ (iterator), a nasza pętla RAG WYPYCHA
zdarzenia przez callback. Ktoś musi blokować po drugiej stronie. Gdyby `answer()` leciała
w tym samym wątku co generator, nie dałoby się wyemitować NICZEGO, dopóki nie skończy —
czyli dokładnie to, czego chcemy uniknąć (użytkownik ma widzieć etapy w trakcie, nie po).
Stąd: wątek roboczy + `queue.Queue` + sentinel kończący iterację.

Ten moduł NIE WIE nic o FastAPI ani o RAG-u. To celowe: dzięki temu testuje się fałszywką,
bez modeli i bez sieci, a `api.py` zostaje czytelny.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Iterator

log = logging.getLogger("offline_rag.serve.stream")

_SENTINEL = object()  # znacznik „koniec strumienia" — nie da się go pomylić z żadnym zdarzeniem


def stream_events(
    work: Callable[[Callable[..., None]], dict],
    on_error: Callable[[BaseException], dict],
) -> Iterator[dict]:
    """Uruchamia `work` w wątku i oddaje jej zdarzenia jako iterator.

    `work(emit)` wykonuje robotę i zwraca payload zdarzenia końcowego.
    `on_error(exc)` mapuje wyjątek na payload zdarzenia błędu — klasyfikacja domenowa
    (Ollama? Qdrant?) należy do warstwy serwującej, nie tutaj.
    """
    q: queue.Queue = queue.Queue()

    def emit(**event) -> None:
        q.put(event)

    def runner() -> None:
        try:
            q.put({"event": "final", **work(emit)})
        except BaseException as exc:  # noqa: BLE001 - świadomie łapiemy wszystko
            # Pełny traceback TYLKO do logu serwera; do przeglądarki idzie komunikat
            # zmapowany przez on_error. Surowy traceback w UI to wyciek szczegółów wdrożenia.
            log.exception("blad w watku roboczym strumienia")
            q.put({"event": "error", **on_error(exc)})
        finally:
            # `finally` jest tu krytyczne: bez sentinela konsument zawisłby na q.get() na zawsze.
            q.put(_SENTINEL)

    # daemon=True: wątek nie może blokować zamknięcia procesu (Ctrl+C na uvicornie).
    threading.Thread(target=runner, name="rag-work", daemon=True).start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        yield item
