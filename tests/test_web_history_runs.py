"""History page: pushes grouped by run, and one-click undo of a run. Fakes only, no network."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from webhelpers import make_push_ui
from yezatas import edits
from yezatas.store import Store

V1, V2, V3 = "vid000001", "vid000002", "vid000003"
UNDO = "Undo this push"


def seed_push(ui: Any, run: int | None, video_id: str, title: str, status: str = "pushed") -> int:
    """A push row whose backup holds `title`, as if that text had been replaced."""
    with Store(ui.config.catalog_db) as store:
        push_id = store.begin_push(
            video_id, "manual", {"title": title, "description": "Old text", "tags": ["old"]}, None, run)
        store.finish_push(push_id, status, {}, "Pushed to YouTube." if status == "pushed" else "No.")
        return push_id


def seed_run(ui: Any, rows: list[tuple[str, str, str]]) -> int:
    """A closed run of (video id, backup title, status) rows."""
    with Store(ui.config.catalog_db) as store:
        run = store.begin_push_run()
    for video_id, title, status in rows:
        seed_push(ui, run, video_id, title, status)
    with Store(ui.config.catalog_db) as store:
        store.finish_push_run(run, len(rows))
    return run


def draft_of(ui: Any, video_id: str) -> Any:
    with Store(ui.config.catalog_db) as store:
        return store.get_draft(video_id)


def test_history_groups_rows_by_run_newest_first_with_an_undo_button(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    seed_push(ui, None, V3, "Old three")  # an older push, from before runs existed
    first = seed_run(ui, [(V1, "Old one", "pushed"), (V2, "Old two", "conflict")])
    second = seed_run(ui, [(V2, "Older two", "pushed")])
    html = ui.client.get("/history").text
    assert "Undo makes drafts with the text from before the push." in html
    assert "Nothing changes on YouTube until you approve and push them." in html
    assert html.index(f'id="run-{second}"') < html.index(f'id="run-{first}"')  # newest run first
    assert html.index(f'id="run-{first}"') < html.index('id="p-1"')  # the old push comes last
    assert re.search(r"Push of [^<]+ · 2 videos · 1 changed", html)
    assert re.search(r"Push of [^<]+ · 1 video · 1 changed", html)
    assert html.count(UNDO) == 2  # one per run; the old push is restored one by one
    assert f'action="/history/runs/{first}/undo"' in html
    assert 'action="/history/runs/' not in html.split('id="p-1"')[1]
    assert html.count("Restore this text") == 4  # every row keeps its own Restore
    assert writer.calls == []


def test_the_undo_button_is_hidden_when_nothing_in_the_run_was_pushed(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "conflict"), (V2, "Old two", "failed")])
    html = ui.client.get("/history").text
    assert f'id="run-{run}"' in html and "0 changed" in html
    assert UNDO not in html and "/undo" not in html


def test_a_run_cut_by_the_page_limit_is_still_shown_whole(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "pushed"), (V2, "Old two", "pushed")])
    seed_push(ui, None, V3, "Newer three")
    monkeypatch.setattr("yezatas.web.app.HISTORY_LIMIT", 2)  # newest two records: #3 and #2
    html = ui.client.get("/history").text
    assert f'id="run-{run}"' in html and "2 videos · 2 changed" in html
    assert 'id="p-1"' in html and 'id="p-2"' in html and 'id="p-3"' in html


def test_undo_makes_drafts_flashes_a_summary_and_goes_to_changes(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "pushed"), (V2, "Old two", "pushed"),
                        (V3, "Old three", "failed")])
    resp = ui.post(f"/history/runs/{run}/undo", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    assert draft_of(ui, V1).title == "Old one" and draft_of(ui, V2).title == "Old two"
    assert draft_of(ui, V1).note == f"Restore of push run #{run}"
    assert draft_of(ui, V1).source == "restore" and draft_of(ui, V1).approved_at is None
    assert draft_of(ui, V3) is None
    html = ui.client.get("/changes").text
    assert "Restore drafts made for 2 videos." in html
    assert "1 skipped: Repotting cactus #3: This video was not changed by that push." in html
    assert "Nothing changes on YouTube until you approve and push them." in html
    assert writer.calls == [] and writer.updates == []  # undo only makes drafts


def test_undo_with_nothing_to_restore_goes_back_to_history(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "pushed")])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "Work in progress", "d", [])
    resp = ui.post(f"/history/runs/{run}/undo", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/history"
    html = ui.client.get("/history").text
    assert "No restore drafts were made." in html
    assert "1 skipped: Repotting cactus #1: This video already has a draft." in html
    assert draft_of(ui, V1).title == "Work in progress"
    assert writer.calls == []


def test_undo_never_overwrites_a_later_change(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "pushed"), (V2, "Old two", "pushed")])
    later = seed_push(ui, None, V1, "Newer one")  # V1 was pushed again afterwards
    html = ui.post(f"/history/runs/{run}/undo").text  # follows the redirect: the flash is on it
    assert draft_of(ui, V1) is None and draft_of(ui, V2).title == "Old two"
    assert f"Changed again by push #{later}; restore that one from History" in html


def test_the_skip_list_is_capped_at_five(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path, count=8)
    ids = [f"vid00000{n}" for n in range(1, 9)]
    run = seed_run(ui, [(v, f"Old {v}", "failed") for v in ids])
    html = ui.post(f"/history/runs/{run}/undo").text
    assert "8 skipped:" in html and "and 3 more" in html
    assert html.count("This video was not changed by that push.") == 5


def test_undoing_an_unknown_run_says_so(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    resp = ui.post("/history/runs/99/undo", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/history"
    assert "This push run does not exist." in ui.client.get("/history").text
    assert writer.calls == []


def test_undo_requires_the_csrf_token(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    run = seed_run(ui, [(V1, "Old one", "pushed")])
    path = f"/history/runs/{run}/undo"
    assert ui.client.post(path, data={}).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong"}).status_code == 403
    assert draft_of(ui, V1) is None and writer.calls == []


def test_a_real_push_of_two_videos_can_be_undone_from_history(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    originals = {}
    with Store(ui.config.catalog_db) as store:
        for video_id in (V1, V2):
            originals[video_id] = store.get_video(video_id).title
            edits.save_draft(store, video_id, originals[video_id] + " (new)", "New text", ["x"])
            edits.approve(store, video_id)
    html = ui.client.get("/changes/push").text
    data = {f"fp_{k}": v for k, v in re.findall(r'name="fp_(\w+)" value="(\w+)"', html)}
    ui.post("/changes/push", data={**data, "confirm": "yes"})
    ui.web.push_job.wait()
    history = ui.client.get("/history").text
    assert history.count(UNDO) == 1 and "2 videos" in history and "2 changed" in history
    run = int(re.search(r"/history/runs/(\d+)/undo", history).group(1))
    writer.calls.clear()
    updates = len(writer.updates)
    resp = ui.post(f"/history/runs/{run}/undo", follow_redirects=False)
    assert resp.headers["location"] == "/changes"
    assert {v: draft_of(ui, v).title for v in (V1, V2)} == originals
    assert writer.calls == [] and len(writer.updates) == updates  # still the pushed text on YouTube


@pytest.mark.parametrize("title", ["<b>Bold</b> & co", "Tom's \"quoted\" title"])
def test_titles_in_the_summary_are_escaped(tmp_path: Path, title: str) -> None:
    ui, _ = make_push_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        store.update_video_text(V1, title, "d", [])
    run = seed_run(ui, [(V1, "Old one", "failed")])
    html = ui.post(f"/history/runs/{run}/undo").text
    assert "<b>Bold</b>" not in html
