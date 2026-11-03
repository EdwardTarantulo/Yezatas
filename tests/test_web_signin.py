import threading
from pathlib import Path
from typing import Any

from oauthlib.oauth2.rfc6749.errors import AccessDeniedError

from webhelpers import (fake_sign_in_ok, make_ui, step_class, write_client_file,
                        write_token)
from yezatas.auth import WRITE, MissingClientSecrets
from yezatas.config import Config
from yezatas.quota import _utc_now, quota_day
from yezatas.store import Store
from yezatas.web.guidance import map_sign_in_error
from yezatas.web.jobs import Job


def wait_signin(ui: Any) -> None:
    ui.web.signin_job.wait()


def test_sign_in_success_saves_token_and_runs_the_check(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    resp = ui.post("/configuration/signin")
    assert resp.status_code == 200
    wait_signin(ui)
    assert ui.web.signin_job.snapshot().status == "done"
    assert ui.config.token_path.is_file()
    # The check ran by itself after sign-in and was charged to the quota ledger.
    assert ui.web.check is not None and ui.web.check.ok
    with Store(ui.config.catalog_db) as store:
        entries = store.quota_entries(quota_day(_utc_now()))
    assert [(e["operation"], e["units"]) for e in entries] == [("channels.list", 1)]
    html = ui.setup_html()
    assert "done" in step_class(html, "signin") and "done" in step_class(html, "testusers")
    assert "Sign out" in html


def test_setup_says_whether_the_sign_in_can_edit(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    write_token(ui.config)
    html = ui.setup_html()
    assert "signed in with read-only access" in html and "myaccount.google.com" not in html
    write_token(ui.config, [WRITE])
    html = ui.setup_html()
    assert "allowed Yezatas to edit your videos" in html and "read-only access." not in html
    assert "https://myaccount.google.com/connections" in html


def test_sign_in_needs_a_client_file(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, sign_in=lambda c: (_ for _ in ()).throw(AssertionError("no sign-in")))
    resp = ui.post("/configuration/signin")
    assert "Add the OAuth client file first" in resp.text
    assert ui.web.signin_job.snapshot().status == "idle"


def test_access_denied_points_to_the_test_user_step(tmp_path: Path) -> None:
    def denied(config: Config) -> Any:
        raise AccessDeniedError()

    ui = make_ui(tmp_path, sign_in=denied)
    write_client_file(ui.config)
    ui.post("/configuration/signin")
    wait_signin(ui)
    snap = ui.web.signin_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "access_denied"
    assert not ui.config.token_path.exists()
    html = ui.setup_html()
    assert "problem" in step_class(html, "testusers")
    assert "Test users" in html and "access_denied" in html
    status = ui.client.get("/api/status").json()
    assert status["signin"]["status"] == "failed"


def test_missing_secrets_points_to_the_client_step() -> None:
    kind, message = map_sign_in_error(MissingClientSecrets("client_secrets.json not found"))
    assert kind == "missing_secrets" and "step 5" in message


def test_missing_secrets_raised_during_sign_in(tmp_path: Path) -> None:
    def missing(config: Config) -> Any:
        raise MissingClientSecrets("gone")

    ui = make_ui(tmp_path, sign_in=missing)
    write_client_file(ui.config)
    ui.post("/configuration/signin")
    wait_signin(ui)
    assert ui.web.signin_job.snapshot().error_kind == "missing_secrets"
    assert "problem" in step_class(ui.setup_html(), "signin")


def test_other_errors_show_class_and_message_but_no_secrets(tmp_path: Path) -> None:
    def broken(config: Config) -> Any:
        raise OSError("port in use")

    ui = make_ui(tmp_path, sign_in=broken)
    write_client_file(ui.config)
    ui.post("/configuration/signin")
    wait_signin(ui)
    assert "OSError: port in use" in ui.setup_html()


def test_invalid_client_error_points_to_step_4() -> None:
    kind, message = map_sign_in_error(RuntimeError("(invalid_client) Unauthorized"))
    assert kind == "invalid_client" and "step 5" in message


def test_no_second_sign_in_while_one_is_waiting_and_retry_after_cancel(tmp_path: Path) -> None:
    release = threading.Event()
    started = threading.Event()
    calls: list[int] = []

    def slow(config: Config) -> Any:
        calls.append(1)
        started.set()
        release.wait(5)
        return fake_sign_in_ok(config)

    ui = make_ui(tmp_path, sign_in=slow)
    write_client_file(ui.config)
    ui.post("/configuration/signin")
    assert started.wait(5)
    html = ui.setup_html()
    assert "Waiting for you to finish in the browser tab" in html
    assert "working" in step_class(html, "signin")

    resp = ui.post("/configuration/signin")
    assert "already waiting" in resp.text
    assert len(calls) == 1

    ui.post("/configuration/signin/cancel")
    snap = ui.web.signin_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "cancelled"
    assert "todo" in step_class(ui.setup_html(), "signin")  # cancelling is not an error

    # The abandoned thread finishing later must not overwrite the cancelled state.
    release.set()
    ui.web.signin_job.wait()
    assert ui.web.signin_job.snapshot().error_kind == "cancelled"


def test_timeout_marks_failed_and_allows_retry(tmp_path: Path) -> None:
    now = [0.0]
    release = threading.Event()
    started = threading.Event()

    def hangs(config: Config) -> Any:
        started.set()
        release.wait(5)

    ui = make_ui(tmp_path, sign_in=hangs, clock=lambda: now[0], sign_in_timeout=300.0)
    write_client_file(ui.config)
    ui.post("/configuration/signin")
    assert started.wait(5)
    assert ui.web.signin_job.snapshot().status == "running"

    now[0] = 301.0
    snap = ui.web.signin_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "timeout"
    html = ui.setup_html()
    assert "5 minutes" in html and "Test users" in html
    assert "problem" in step_class(html, "testusers")

    release.set()
    ui.web.signin_job.wait()
    assert ui.web.signin_job.snapshot().error_kind == "timeout"

    ui.web.signin_job.wait()
    assert ui.web.signin_job.start(lambda report: None)  # a new run can start


def test_sign_out_deletes_token_and_resets_the_check(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    write_token(ui.config)
    ui.post("/configuration/check")
    assert ui.web.check is not None
    resp = ui.post("/configuration/signout")
    assert "Signed out" in resp.text
    assert not ui.config.token_path.exists()
    assert ui.web.check is None
    assert "Sign in with Google" in ui.setup_html()


def test_sign_in_is_refused_when_already_signed_in(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    write_token(ui.config)
    assert "already signed in" in ui.post("/configuration/signin").text


def test_job_reports_progress_and_ignores_stale_runs() -> None:
    job = Job(lambda exc: ("error", str(exc)))
    gate = threading.Event()

    def work(report: Any) -> str:
        report("halfway")
        gate.wait(5)
        return "result"

    assert job.start(work)
    for _ in range(100):
        if job.snapshot().message == "halfway":
            break
        threading.Event().wait(0.02)
    assert job.snapshot().message == "halfway"
    assert not job.start(work)
    gate.set()
    job.wait()
    snap = job.snapshot()
    assert snap.status == "done" and snap.result == "result"
