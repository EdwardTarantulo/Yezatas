"""Keep the tests independent of the machine's own AI settings."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_primary_ai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("YEZATAS_PRIMARY_AI", "MODAL_API_KEY", "MODAL_ENDPOINT_URL"):
        monkeypatch.delenv(name, raising=False)
