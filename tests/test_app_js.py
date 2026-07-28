"""
tests/test_app_js.py
Podpina weryfikację `app.js` (atrapa DOM w node) pod pytest, żeby leciała razem z resztą.

DLACZEGO NIE ZWYKŁY TEST PYTHONOWY: logika, o którą chodzi, jest w JavaScripcie — trzeba ją
wykonać, a nie czytać. DLACZEGO NIE PRZEGLĄDARKA (playwright/selenium): to byłaby ciężka
zależność i dodatkowy runtime do audytu, a sprawdzamy składanie węzłów, nie prezentację.
Atrapa DOM w node kosztuje zero nowych zależności — node albo jest, albo test się pomija.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHECK = Path(__file__).resolve().parent / "check_app_js.js"
APP_JS = ROOT / "src" / "serve" / "static" / "app.js"


@pytest.mark.skipif(shutil.which("node") is None,
                    reason="node niedostępny — logika app.js sprawdzana tylko ręcznie")
def test_logika_app_js_na_atrapie_dom():
    proc = subprocess.run(["node", str(CHECK), str(APP_JS)],
                          capture_output=True, text=True, timeout=60)
    # Wypisujemy wyjście, żeby przy czerwonym było widać KTÓRA kontrola padła,
    # a nie tylko „exit code 1".
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, "kontrole app.js nie przeszly (szczegoly w wyjsciu wyzej)"
