import socket
from pathlib import Path
from typing import Any

import pytest

from yezatas.__main__ import main


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ui_command_prints_tokenised_url_and_binds_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YEZATAS_HOME", str(tmp_path / "home"))
    calls: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(app=app, **kw))
    port = free_port()
    assert main(["ui", "--port", str(port), "--no-browser"]) == 0
    out = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}/?t=" in out
    assert calls["host"] == "127.0.0.1" and calls["port"] == port
    assert calls["app"].state.token in out


def test_ui_command_reports_a_port_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("YEZATAS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: pytest.fail("must not start"))
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert main(["ui", "--port", str(port), "--no-browser"]) == 1
    assert "already in use" in capsys.readouterr().err
