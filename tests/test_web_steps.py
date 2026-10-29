from pathlib import Path

import pytest

from fakes import FakeYouTubeReader
from webhelpers import make_ui, step_class, write_client_file, write_token
from yezatas.web.clientfile import ClientInfo
from yezatas.web.jobs import JobSnapshot
from yezatas.web.steps import CheckState, compute_steps, estimate_pull_units

CLIENT = ClientInfo("demo-project", "123456789012…")
IDLE = JobSnapshot("idle")
GOOD_CHECK = CheckState(True, channel_title="Sample", video_count=10, has_uploads=True)


def statuses(**kw) -> dict[str, str]:
    args = dict(client=None, client_problem="", signed_in=False, signin=IDLE, check=None,
                pull=IDLE, pulled=False)
    args.update(kw)
    return {s.key: s.status for s in compute_steps(**args)}


def test_nothing_done() -> None:
    assert set(statuses().values()) == {"todo"}
    steps = compute_steps(None, "", False, IDLE, None, IDLE, False)
    assert [s.enabled for s in steps] == [True, True, True, True, False, False, False]


def test_client_file_unlocks_sign_in() -> None:
    s = statuses(client=CLIENT)
    assert s["project"] == "done" and s["client"] == "done" and s["signin"] == "todo"
    assert compute_steps(CLIENT, "", False, IDLE, None, IDLE, False)[4].enabled


def test_invalid_client_file_is_a_problem_on_step_4() -> None:
    steps = compute_steps(None, "This is not JSON.", False, IDLE, None, IDLE, False)
    assert steps[3].status == "problem" and steps[3].note == "This is not JSON."


def test_signed_in_marks_consent_done_and_unlocks_check() -> None:
    steps = compute_steps(CLIENT, "", True, IDLE, None, IDLE, False)
    by = {s.key: s for s in steps}
    assert by["consent"].status == "done" and by["signin"].status == "done"
    assert by["check"].enabled and by["check"].status == "todo"
    assert not by["pull"].enabled


def test_sign_in_running_and_failed() -> None:
    running = JobSnapshot("running", "Waiting…")
    assert statuses(client=CLIENT, signin=running)["signin"] == "working"
    denied = JobSnapshot("failed", "add yourself", "access_denied")
    s = statuses(client=CLIENT, signin=denied)
    assert s["consent"] == "problem" and s["signin"] == "problem"
    cancelled = JobSnapshot("failed", "Sign-in cancelled.", "cancelled")
    s = statuses(client=CLIENT, signin=cancelled)
    assert s["signin"] == "todo" and s["consent"] == "todo"


def test_checked_connection_unlocks_pull_and_confirms_api() -> None:
    s = statuses(client=CLIENT, signed_in=True, check=GOOD_CHECK)
    assert s["check"] == "done" and s["api"] == "done" and s["pull"] == "todo"
    steps = compute_steps(CLIENT, "", True, IDLE, GOOD_CHECK, IDLE, False)
    assert steps[6].enabled


def test_api_disabled_is_a_problem_on_step_2_and_6() -> None:
    check = CheckState(False, "api_disabled", "Enable the API.")
    steps = {s.key: s for s in compute_steps(CLIENT, "", True, IDLE, check, IDLE, False)}
    assert steps["api"].status == "problem" and steps["api"].note == "Enable the API."
    assert steps["check"].status == "problem" and "step 2" in steps["check"].note


def test_other_check_failures_stay_on_step_6() -> None:
    check = CheckState(False, "no_channel", "No channel here.")
    steps = {s.key: s for s in compute_steps(CLIENT, "", True, IDLE, check, IDLE, False)}
    assert steps["check"].note == "No channel here." and steps["api"].status == "todo"
    auth_check = CheckState(False, "auth", "Sign in again.")
    steps = {s.key: s for s in compute_steps(CLIENT, "", True, IDLE, auth_check, IDLE, False)}
    assert steps["signin"].status == "problem"


def test_pull_states() -> None:
    base = dict(client=CLIENT, signed_in=True, check=GOOD_CHECK)
    assert statuses(**base, pull=JobSnapshot("running", "Listing…"))["pull"] == "working"
    failed = statuses(**base, pull=JobSnapshot("failed", "boom", "other"))
    assert failed["pull"] == "problem"
    assert statuses(**base, pulled=True)["pull"] == "done"
    disabled = statuses(**base, pull=JobSnapshot("failed", "off", "api_disabled"))
    assert disabled["api"] == "problem" and disabled["pull"] == "todo"


@pytest.mark.parametrize("videos, units", [(0, 2), (1, 3), (50, 3), (51, 5), (120, 7), (1000, 41)])
def test_estimate_pull_units(videos: int, units: int) -> None:
    assert estimate_pull_units(videos) == units


def test_page_reflects_each_state(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    html = ui.setup_html()
    assert "todo" in step_class(html, "client") and "current" in step_class(html, "project")
    assert "locked" in step_class(html, "signin")

    write_client_file(ui.config)
    html = ui.setup_html()
    assert "done" in step_class(html, "client") and "current" in step_class(html, "signin")

    write_token(ui.config)
    html = ui.setup_html()
    assert "done" in step_class(html, "signin") and "current" in step_class(html, "check")

    assert ui.post("/setup/check").status_code == 200
    html = ui.setup_html()
    assert "done" in step_class(html, "check") and "done" in step_class(html, "api")
    assert "current" in step_class(html, "pull")

    ui.post("/setup/pull")
    ui.web.pull_job.wait()
    html = ui.setup_html()
    assert "done" in step_class(html, "pull")
    assert "120" in html


def test_footer_shows_quota_and_data_location(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    html = ui.setup_html()
    assert "0 used, 10000 remaining of 10000" in html
    assert str(ui.config.home) in html
    assert "Nothing is changed on YouTube" in html


def test_empty_reader_fixture_is_valid() -> None:
    assert FakeYouTubeReader(0).channels_mine("snippet,statistics")["items"][0]["statistics"][
        "videoCount"] == "0"
