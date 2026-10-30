from pathlib import Path

from fastapi.testclient import TestClient

from webhelpers import BASE_URL, PORT, TOKEN, make_config, make_ui
from yezatas.web.app import create_app


def test_no_cookie_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    for path in ("/", "/setup", "/api/status", "/nope"):
        resp = ui.client.get(path, follow_redirects=False)
        assert resp.status_code == 403, path
        assert "link that Yezatas printed in the terminal" in resp.text


def test_static_files_need_no_cookie(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    assert ui.client.get("/static/app.css").status_code == 200
    assert ui.client.get("/static/app.js").status_code == 200


def test_correct_token_sets_cookie_and_redirects_without_token(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    resp = ui.client.get(f"/setup?t={TOKEN}&x=1", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/setup?x=1"
    cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert TOKEN not in resp.headers["set-cookie"]
    assert ui.client.get("/setup").status_code == 200


def test_start_url_goes_to_setup_when_nothing_pulled(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    assert ui.app.state.start_url == f"{BASE_URL}/?t={TOKEN}"
    resp = ui.client.get(f"/?t={TOKEN}", follow_redirects=False)
    assert resp.headers["location"] == "/"
    resp = ui.client.get("/", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"


def test_wrong_token_is_forbidden_and_sets_no_cookie(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    resp = ui.client.get("/?t=wrong", follow_redirects=False)
    assert resp.status_code == 403
    assert "set-cookie" not in resp.headers
    assert ui.client.get("/setup").status_code == 403


def test_forged_cookie_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    ui.client.cookies.set(f"yezatas_session_{PORT}", "made-up")
    assert ui.client.get("/setup").status_code == 403
    ui.client.cookies.set(f"yezatas_session_{PORT}", "made-up.0123abcd")
    assert ui.client.get("/setup").status_code == 403


def test_cookie_name_includes_the_port(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    assert f"yezatas_session_{PORT}" in ui.client.cookies


def test_open_tab_stays_signed_in_after_a_restart(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    csrf = ui.csrf()
    restarted = create_app(ui.config, port=PORT, token="new-run-token", open_browser=False)
    tab = TestClient(restarted, base_url=BASE_URL, cookies=dict(ui.client.cookies))
    assert tab.get("/setup").status_code == 200
    resp = tab.post("/setup/signout", data={"csrf": csrf}, follow_redirects=False)
    assert resp.status_code == 303


def test_a_cookie_from_another_home_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    other = create_app(make_config(tmp_path / "other"), port=PORT, token=TOKEN, open_browser=False)
    tab = TestClient(other, base_url=BASE_URL, cookies=dict(ui.client.cookies))
    assert tab.get("/setup").status_code == 403


def test_session_key_is_created_once_and_reused(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    key_file = ui.config.credentials_dir / "session.key"
    first = key_file.read_bytes()
    assert len(first) == 32
    make_ui(tmp_path, config=ui.config)
    assert key_file.read_bytes() == first


def test_bad_host_is_rejected(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    for base in ("http://evil.example:8765", f"http://127.0.0.1:{PORT + 1}", "http://testserver"):
        other = TestClient(ui.app, base_url=base)
        assert other.get(f"/?t={TOKEN}", follow_redirects=False).status_code == 400
        assert other.get("/static/app.css").status_code == 400


def test_localhost_host_is_allowed(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False)
    other = TestClient(ui.app, base_url=f"http://localhost:{PORT}")
    assert other.get(f"/?t={TOKEN}").status_code == 200


def test_post_without_csrf_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    for path in ("/setup/signin", "/setup/signout", "/setup/check", "/setup/pull",
                 "/setup/signin/cancel"):
        assert ui.client.post(path, data={}).status_code == 403, path
        assert ui.client.post(path, data={"csrf": "wrong"}).status_code == 403, path
    resp = ui.client.post("/setup/client", files={"client_file": ("c.json", b"{}")})
    assert resp.status_code == 403
    assert not ui.config.client_secrets_path.exists()


def test_post_with_csrf_header_is_accepted(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.post("/setup/signout", headers={"X-CSRF-Token": ui.csrf()},
                          follow_redirects=False)
    assert resp.status_code == 303


def test_csrf_token_is_per_session(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    other = TestClient(ui.app, base_url=BASE_URL)
    other.get(f"/?t={TOKEN}")
    assert other.post("/setup/signout", data={"csrf": ui.csrf()}).status_code == 403


def test_cross_origin_post_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.post("/setup/signout", data={"csrf": ui.csrf()},
                          headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403
    ok = ui.client.post("/setup/signout", data={"csrf": ui.csrf()},
                        headers={"Origin": BASE_URL}, follow_redirects=False)
    assert ok.status_code == 303


def test_state_changing_routes_are_post_only(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    for path in ("/setup/signin", "/setup/signout", "/setup/check", "/setup/pull", "/setup/client"):
        assert ui.client.get(path).status_code == 405, path


def test_security_headers(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.get("/setup")
    assert resp.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
    csp = resp.headers["content-security-policy"]
    # The only external source is YouTube's thumbnail host, for images.
    assert "img-src 'self' https://i.ytimg.com;" in csp
    assert csp.replace("https://i.ytimg.com", "").count("http") == 0
    assert "cdn" not in resp.text.lower()


def test_referrer_policy_lets_browsers_send_the_real_origin(tmp_path: Path) -> None:
    # "no-referrer" makes Chromium send `Origin: null` on form posts, which is refused.
    ui = make_ui(tmp_path)
    assert ui.client.get("/setup").headers["referrer-policy"] == "same-origin"


def test_null_origin_post_is_forbidden(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.post("/setup/signout", data={"csrf": ui.csrf()}, headers={"Origin": "null"})
    assert resp.status_code == 403
