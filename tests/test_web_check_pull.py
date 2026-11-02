from dataclasses import replace
from pathlib import Path

import httplib2
import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from fakes import CHANNEL_TITLE, FakeYouTubeReader
from webhelpers import (client_json, http_error, make_config, make_ui, step_class, write_client_file,
                        write_token)
from yezatas.quota import _utc_now, quota_day
from yezatas.store import Store
from yezatas.web.guidance import classify_api_error, map_api_error
from yezatas.youtube import api_error_info


def ready_ui(tmp_path: Path, **kwargs):
    ui = make_ui(tmp_path, **kwargs)
    write_client_file(ui.config)
    write_token(ui.config)
    return ui


def ledger(ui) -> list[tuple[str, int]]:
    with Store(ui.config.catalog_db) as store:
        rows = store.quota_entries(quota_day(_utc_now()))
    return [(r["operation"], r["units"]) for r in rows]


# --- connection check ---------------------------------------------------------------------------


def test_check_shows_channel_and_charges_one_unit(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    resp = ui.post("/setup/check")
    assert CHANNEL_TITLE in resp.text
    assert "120 public videos" in resp.text and "uploads playlist found" in resp.text
    assert ui.reader.calls == [("channels_mine", "snippet,statistics,contentDetails")]
    assert ledger(ui) == [("channels.list", 1)]
    assert "1 used, 9,999 remaining of 10,000" in resp.text
    assert "done" in step_class(resp.text, "check") and "done" in step_class(resp.text, "api")


def test_check_needs_a_sign_in(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.post("/setup/check")
    assert "Sign in first" in resp.text
    assert ui.reader.calls == [] and ledger(ui) == []


def test_check_when_token_is_unusable_says_sign_in_again(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    ui.config.token_path.write_text("{}", encoding="utf-8")  # present but not valid
    resp = ui.post("/setup/check")
    assert "Sign in again" in resp.text and "problem" in step_class(resp.text, "signin")
    assert ledger(ui) == []


def test_check_api_not_enabled_points_to_step_2(tmp_path: Path) -> None:
    err = http_error(403, "accessNotConfigured", "YouTube Data API v3 has not been used in project 1 "
                     "before or it is disabled.")
    ui = ready_ui(tmp_path, reader=FakeYouTubeReader(error=err))
    resp = ui.post("/setup/check")
    assert "problem" in step_class(resp.text, "api")
    assert "not enabled" in resp.text and "step 2" in resp.text
    assert ledger(ui) == [("channels.list", 1)]  # charged before the call, as everywhere


def test_check_service_disabled_detail_reason_is_recognised(tmp_path: Path) -> None:
    err = http_error(403, "forbidden", detail_reason="SERVICE_DISABLED")
    ui = ready_ui(tmp_path, reader=FakeYouTubeReader(error=err))
    assert "problem" in step_class(ui.post("/setup/check").text, "api")


def test_check_quota_exceeded_explains_the_reset(tmp_path: Path) -> None:
    err = http_error(403, "quotaExceeded", "quota")
    ui = ready_ui(tmp_path, reader=FakeYouTubeReader(error=err))
    resp = ui.post("/setup/check")
    assert "midnight Pacific" in resp.text and "problem" in step_class(resp.text, "check")


def test_check_without_a_channel(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path, reader=FakeYouTubeReader(has_channel=False))
    resp = ui.post("/setup/check")
    assert "has no YouTube channel" in resp.text and "brand account" in resp.text


def test_check_with_expired_google_sign_in(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path, reader=FakeYouTubeReader(error=RefreshError("invalid_grant")))
    resp = ui.post("/setup/check")
    assert "7 days" in resp.text and "problem" in step_class(resp.text, "signin")


def test_check_refused_locally_when_quota_is_gone(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path, config=replace(make_config(tmp_path), daily_quota_budget=1))
    ui.post("/setup/check")
    ui.post("/setup/check")  # second one does not fit in a budget of 1
    assert ledger(ui) == [("channels.list", 1)]
    assert len(ui.reader.calls) == 1
    assert ui.web.check is not None and ui.web.check.kind == "quota_local"


# --- catalog pull -------------------------------------------------------------------------------


def test_pull_stores_videos_and_charges_the_ledger(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    html = ui.setup_html()
    assert "<strong>7</strong> quota units" in html  # 1 + 3 + 3 for 120 videos
    assert "Nothing is changed on YouTube" in html

    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    snap = ui.web.pull_job.snapshot()
    assert snap.status == "done"
    assert snap.result.video_count == 120 and snap.result.units_used == 7
    with Store(ui.config.catalog_db) as store:
        assert len(store.list_videos()) == 120
    assert ledger(ui) == [("channels.list", 1), ("channels.list", 1), ("playlistItems.list", 1),
                          ("playlistItems.list", 1), ("playlistItems.list", 1),
                          ("videos.list", 1), ("videos.list", 1), ("videos.list", 1)]
    html = ui.setup_html()
    assert "<strong>120</strong> videos stored" in html and "7 quota units used" in html
    assert "done" in step_class(html, "pull")
    home = ui.client.get("/seo", follow_redirects=False)
    assert home.status_code == 200 and "120 videos were pulled" in home.text


def test_pull_progress_messages(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    seen: list[str] = []
    original = ui.web.pull_job._report

    def spy(generation: int, message: str) -> None:
        seen.append(message)
        original(generation, message)

    ui.web.pull_job._report = spy
    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    assert any(m.startswith("Listing your uploads (page 1") for m in seen)
    assert any(m.startswith("Fetching video details (batch 3") for m in seen)


def test_pull_checks_the_connection_itself_after_a_restart(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    assert "Check and pull" in ui.setup_html()
    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    snap = ui.web.pull_job.snapshot()
    assert snap.status == "done"
    assert ui.web.check.ok
    assert ledger(ui)[:2] == [("channels.list", 1), ("channels.list", 1)]


def test_pull_without_a_sign_in_asks_for_the_check(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config)
    resp = ui.post("/setup/pull")
    assert "Check the connection first" in resp.text
    assert ui.web.pull_job.snapshot().status == "idle"
    assert ledger(ui) == []
    assert ui.reader.calls == []


def test_pull_does_not_rerun_a_failed_check(tmp_path: Path) -> None:
    from yezatas.web.steps import CheckState

    ui = ready_ui(tmp_path)
    ui.web._check = CheckState(False, "auth", "Sign in again.")
    resp = ui.post("/setup/pull")
    assert "Check the connection first" in resp.text
    assert ledger(ui) == []
    assert ui.reader.calls == []


def test_pull_refused_when_estimate_exceeds_remaining_quota(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path, config=replace(make_config(tmp_path), daily_quota_budget=5))
    ui.post("/setup/check")
    resp = ui.post("/setup/pull")
    assert "needs about 7 quota units but only 4" in resp.text
    assert ui.web.pull_job.snapshot().status == "idle"
    assert "disabled" in resp.text


def test_pull_from_today_comes_back_to_today(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    resp = ui.post("/setup/pull", data={"back": "/"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    ui.web.pull_job.wait()
    html = ui.client.get("/").text
    assert '<h1 class="page-title">Today</h1>' in html and "120 videos" in html
    assert 'id="pull-msg"' not in html and 'id="today-pull-failed"' not in html


def test_pull_from_today_refused_still_comes_back(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path, config=replace(make_config(tmp_path), daily_quota_budget=5))
    ui.post("/setup/check")
    resp = ui.post("/setup/pull", data={"back": "/"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/"
    assert ui.web.pull_job.snapshot().status == "idle"
    other = ui.post("/setup/pull", data={"back": "https://example.com/"}, follow_redirects=False)
    assert other.headers["location"] == "/setup#step-pull"


def test_pull_failure_is_reported_on_the_step(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")

    def boom(*args, **kwargs):
        raise http_error(500, "backendError", "Backend broke.")

    ui.reader.playlist_items = boom  # type: ignore[method-assign]
    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    snap = ui.web.pull_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "other"
    html = ui.setup_html()
    assert "problem" in step_class(html, "pull") and "HTTP 500" in html
    assert ui.post("/setup/pull").status_code == 200  # can retry


def test_pull_when_signed_out_in_the_meantime(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    ui.config.token_path.unlink()
    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    assert ui.web.pull_job.snapshot().error_kind == "auth"


def test_home_redirects_to_setup_until_a_pull_exists(tmp_path: Path) -> None:
    ui = ready_ui(tmp_path)
    assert ui.client.get("/", follow_redirects=False).headers["location"] == "/setup"


def test_each_request_uses_its_own_store(tmp_path: Path) -> None:
    # Repeated page loads and background jobs would raise sqlite3.ProgrammingError if a
    # connection crossed threads.
    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    ui.post("/setup/pull")
    for _ in range(5):
        assert ui.client.get("/setup").status_code == 200
    ui.web.pull_job.wait()
    assert ui.web.pull_job.snapshot().status == "done"


# --- error helper ---------------------------------------------------------------------------------


def make_error(status: int, body: str) -> HttpError:
    return HttpError(httplib2.Response({"status": str(status)}), body.encode())


ACCESS_NOT_CONFIGURED = (
    '{"error": {"code": 403, "message": "YouTube Data API v3 has not been used in project 123 '
    'before or it is disabled.", "errors": [{"message": "x", "domain": "usageLimits", '
    '"reason": "accessNotConfigured", "extendedHelp": "https://example.invalid"}], '
    '"status": "PERMISSION_DENIED"}}'
)
SERVICE_DISABLED = (
    '{"error": {"code": 403, "message": "API disabled.", "status": "PERMISSION_DENIED", '
    '"details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": '
    '"SERVICE_DISABLED", "domain": "googleapis.com"}]}}'
)
QUOTA_EXCEEDED = (
    '{"error": {"code": 403, "message": "The request cannot be completed because you have '
    'exceeded your quota.", "errors": [{"message": "x", "domain": "youtube.quota", '
    '"reason": "quotaExceeded"}]}}'
)


def test_api_error_info_parses_reasons_and_status() -> None:
    info = api_error_info(make_error(403, ACCESS_NOT_CONFIGURED))
    assert info.status == 403 and info.reason == "accessNotConfigured"
    assert "has not been used in project" in info.message
    assert classify_api_error(info) == "api_disabled"

    info = api_error_info(make_error(403, SERVICE_DISABLED))
    assert info.reasons == ("SERVICE_DISABLED",)
    assert classify_api_error(info) == "api_disabled"

    info = api_error_info(make_error(403, QUOTA_EXCEEDED))
    assert info.reason == "quotaExceeded" and classify_api_error(info) == "quota"


@pytest.mark.parametrize("status, body, kind", [
    (403, '{"error": {"errors": [{"reason": "rateLimitExceeded"}]}}', "rate"),
    (401, '{"error": {"errors": [{"reason": "authError"}]}}', "auth"),
    (401, "", "auth"),
    (403, '{"error": {"errors": [{"reason": "insufficientPermissions"}]}}', "auth"),
    (404, '{"error": {"errors": [{"reason": "channelNotFound"}]}}', "no_channel"),
    (500, '{"error": {"errors": [{"reason": "backendError"}]}}', "other"),
    (400, "<html>not json</html>", "other"),
    (403, '{"error": "just a string"}', "other"),
    (403, '[1, 2]', "other"),
])
def test_classify_other_reasons_and_garbage_bodies(status: int, body: str, kind: str) -> None:
    assert classify_api_error(api_error_info(make_error(status, body))) == kind


def test_map_api_error_messages() -> None:
    kind, message = map_api_error(make_error(403, ACCESS_NOT_CONFIGURED))
    assert kind == "api_disabled" and "step 2" in message
    kind, message = map_api_error(make_error(403, QUOTA_EXCEEDED))
    assert kind == "quota" and "midnight Pacific" in message
    kind, message = map_api_error(ValueError("odd"))
    assert (kind, message) == ("other", "ValueError: odd")


def test_page_while_pull_is_running_shows_progress_and_no_second_start(tmp_path: Path) -> None:
    import threading

    ui = ready_ui(tmp_path)
    ui.post("/setup/check")
    gate, entered = threading.Event(), threading.Event()
    real = ui.reader.playlist_items

    def slow(playlist_id, page_token):
        entered.set()
        gate.wait(5)
        return real(playlist_id, page_token)

    ui.reader.playlist_items = slow  # type: ignore[method-assign]
    ui.post("/setup/pull")
    assert entered.wait(5)
    html = ui.setup_html()
    assert "working" in step_class(html, "pull")
    assert "Listing your uploads (page 1" in html
    assert 'data-poll="1"' in html
    assert "already running" in ui.post("/setup/pull").text
    status = ui.client.get("/api/status").json()
    assert status["pull"]["status"] == "running"
    gate.set()
    ui.web.pull_job.wait()
    assert ui.web.pull_job.snapshot().status == "done"


def test_api_page_link_opens_the_client_project(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    write_client_file(ui.config, client_json("my-proj-123"))
    html = ui.setup_html()
    assert "youtube.googleapis.com?project=my-proj-123" in html
