import json
import stat
import sys
from pathlib import Path

import pytest

from webhelpers import (FAKE_CLIENT_ID, FAKE_SECRET, client_json, make_config, make_ui,
                        step_class, write_client_file, write_token)
from yezatas.web.clientfile import (MAX_CLIENT_FILE_BYTES, ClientFileError, parse_client_file,
                                    read_saved_client)


def test_parse_valid_client_shows_only_safe_fields() -> None:
    info = parse_client_file(client_json("my-project"))
    assert info.project_id == "my-project"
    assert info.client_id_short == FAKE_CLIENT_ID[:12] + "…"
    assert FAKE_SECRET not in repr(info)


def test_parse_accepts_utf8_bom_and_missing_project_id() -> None:
    raw = json.loads(client_json())
    del raw["installed"]["project_id"]
    info = parse_client_file(b"\xef\xbb\xbf" + json.dumps(raw).encode())
    assert info.project_id == "(not given)"


def test_parse_rejects_web_client_with_desktop_hint() -> None:
    data = json.dumps({"web": {"client_id": "x", "client_secret": "y"}}).encode()
    with pytest.raises(ClientFileError, match="Desktop app"):
        parse_client_file(data)


@pytest.mark.parametrize("data", [
    b"not json at all",
    b"\xff\xfe\x00binary",
    b"[]",
    b'"text"',
    b"{}",
    b'{"installed": "nope"}',
    b'{"installed": {"client_id": "a", "client_secret": "b", "auth_uri": "c"}}',
    b'{"installed": {"client_id": "", "client_secret": "b", "auth_uri": "c", "token_uri": "d"}}',
    b'{"type": "authorized_user", "refresh_token": "x"}',
])
def test_parse_rejects_other_shapes(data: bytes) -> None:
    with pytest.raises(ClientFileError):
        parse_client_file(data)


def test_parse_names_missing_keys_without_values() -> None:
    data = json.dumps({"installed": {"client_id": "abc", "client_secret": FAKE_SECRET}}).encode()
    with pytest.raises(ClientFileError) as exc:
        parse_client_file(data)
    assert "auth_uri" in str(exc.value) and "token_uri" in str(exc.value)
    assert FAKE_SECRET not in str(exc.value)


def test_parse_rejects_service_account_and_oversize() -> None:
    with pytest.raises(ClientFileError, match="service account"):
        parse_client_file(b'{"type": "service_account"}')
    with pytest.raises(ClientFileError, match="64 KB"):
        parse_client_file(b" " * (MAX_CLIENT_FILE_BYTES + 1))


def test_read_saved_client_states(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert read_saved_client(config) == (None, "")
    write_client_file(config, b"junk")
    info, problem = read_saved_client(config)
    assert info is None and "not a JSON file" in problem
    write_client_file(config)
    info, problem = read_saved_client(config)
    assert info is not None and problem == ""


def test_upload_valid_file_is_saved_and_secret_never_shown(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.upload(client_json("demo-project"))
    assert resp.status_code == 200  # redirect followed to /setup
    assert ui.config.client_secrets_path.read_bytes() == client_json("demo-project")
    html = resp.text
    assert "demo-project" in html
    assert FAKE_CLIENT_ID[:12] + "…" in html
    assert FAKE_SECRET not in html and FAKE_CLIENT_ID not in html
    assert FAKE_SECRET not in ui.setup_html()
    assert "card done" in step_class(html, "client")
    # No leftover temp files next to the saved file.
    assert [p.name for p in ui.config.credentials_dir.iterdir()] == ["client_secrets.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_uploaded_file_is_private_on_posix(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    ui.upload(client_json())
    mode = stat.S_IMODE(ui.config.client_secrets_path.stat().st_mode)
    assert mode == 0o600


def test_upload_web_client_is_rejected_with_desktop_message(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    web = json.dumps({"web": {"client_id": "x", "client_secret": FAKE_SECRET}}).encode()
    resp = ui.upload(web)
    assert "Web application client" in resp.text and "Desktop app" in resp.text
    assert not ui.config.client_secrets_path.exists()
    assert FAKE_SECRET not in resp.text


def test_upload_far_too_big_is_refused_unread(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.upload(b"x" * (MAX_CLIENT_FILE_BYTES * 8))
    assert resp.status_code == 413 and "too big" in resp.text
    assert not ui.config.client_secrets_path.exists()


@pytest.mark.parametrize("content", [b"hello", b"{}", b"x" * (MAX_CLIENT_FILE_BYTES + 1)])
def test_upload_junk_and_oversize_are_rejected(tmp_path: Path, content: bytes) -> None:
    ui = make_ui(tmp_path)
    resp = ui.upload(content)
    assert resp.status_code == 200
    assert 'class="flash error"' in resp.text
    assert not ui.config.client_secrets_path.exists()


def test_upload_without_a_file_asks_for_one(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.post("/setup/client", data={"csrf": ui.csrf()})
    assert "Choose the JSON file" in resp.text


def test_replacing_needs_confirmation(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config, client_json("old-project"))
    resp = ui.upload(client_json("new-project"))
    assert "Replace the saved file" in resp.text
    assert b"old-project" in ui.config.client_secrets_path.read_bytes()

    resp = ui.upload(client_json("new-project"), replace_confirmed=True)
    assert "new-project" in resp.text
    assert b"new-project" in ui.config.client_secrets_path.read_bytes()


def test_replacing_signs_out(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    write_token(ui.config)
    ui.upload(client_json("new-project"), replace_confirmed=True)
    assert not ui.config.token_path.exists()


def test_broken_saved_file_can_be_replaced_without_confirmation(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config, b"junk")
    assert "problem" in step_class(ui.setup_html(), "client")
    ui.upload(client_json())
    assert "card done" in step_class(ui.setup_html(), "client")


def test_flash_is_shown_once(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    assert 'class="flash ok"' in ui.upload(client_json()).text
    assert 'class="flash' not in ui.setup_html()
