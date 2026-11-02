"""Edit form, changes page, push confirmation and result, history. Fakes only, no network."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest
from oauthlib.oauth2.rfc6749.errors import AccessDeniedError

from webhelpers import fake_creds, make_config, make_push_ui, write_client_file
from yezatas import edits
from yezatas.store import Store

V1, V2, V3 = "vid000001", "vid000002", "vid000003"


def save(ui: Any, video_id: str, *, title: str | None = None, description: str | None = None,
         tags: list[str] | None = None, approve: bool = False) -> None:
    """Make a draft straight in the store (defaults: the catalog text with a new title)."""
    with Store(ui.config.catalog_db) as store:
        video = store.get_video(video_id)
        edits.save_draft(
            store, video_id,
            title if title is not None else video.title + " (new)",
            description if description is not None else video.description,
            tags if tags is not None else video.tags,
        )
        if approve:
            edits.approve(store, video_id)


def get_draft(ui: Any, video_id: str) -> Any:
    with Store(ui.config.catalog_db) as store:
        return store.get_draft(video_id)


def fingerprints(html: str) -> dict[str, str]:
    return dict(re.findall(r'name="fp_(\w+)" value="(\w+)"', html))


def push_form(fps: dict[str, str], confirm: bool = True) -> dict[str, str]:
    data = {f"fp_{k}": v for k, v in fps.items()}
    if confirm:
        data["confirm"] = "yes"
    return data


def pushed(ui: Any) -> str:
    """Approve V1 and V2, push with the confirmation box ticked, and return the result page."""
    save(ui, V1, approve=True)
    save(ui, V2, approve=True)
    fps = fingerprints(ui.client.get("/changes/push").text)
    ui.post("/changes/push", data=push_form(fps))
    ui.web.push_job.wait()
    return ui.client.get("/changes/push").text


# --- the edit form -------------------------------------------------------------------------------
def test_video_page_has_an_edit_form_prefilled_with_the_catalog_text(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    html = ui.client.get(f"/videos/{V1}").text
    assert 'id="edit"' in html and f'action="/videos/{V1}/draft"' in html
    assert 'name="title" value="Repotting cactus #1"' in html
    assert re.search(r'<textarea[^>]*rows="14"[^>]*>\s*How to repot cactus number 1\.</textarea>', html)
    assert 'spellcheck="false">cactus, garden</textarea>' in html
    assert "Review changes" not in html
    assert html.index('id="edit"') < html.index('id="group-')
    assert "<script>" not in html and "style=" not in html


def test_edit_form_is_prefilled_with_the_draft_and_shows_its_issues(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1, title="Bad <title>", description="Draft text", tags=["a", "b"])
    html = ui.client.get(f"/videos/{V1}").text
    assert 'value="Bad &lt;title&gt;"' in html
    assert "Draft text</textarea>" in html and 'spellcheck="false">a, b</textarea>' in html
    assert "Error:" in html and f'href="/changes#d-{V1}"' in html and "Review changes" in html


def test_saving_makes_a_draft_and_goes_to_the_changes_page(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    resp = ui.post(f"/videos/{V1}/draft", follow_redirects=False,
                   data={"title": "  A better title ", "description": "Line1\r\nLine2",
                         "tags": "cactus, , Garden, garden, new"})
    assert resp.status_code == 303 and resp.headers["location"] == f"/changes#d-{V1}"
    row = get_draft(ui, V1)
    assert row.title == "A better title" and row.description == "Line1\nLine2"
    assert row.tags == ("cactus", "Garden", "new") and row.approved_at is None
    html = ui.client.get(f"/changes").text
    assert "Draft saved." in html and f'id="d-{V1}"' in html
    assert writer.calls == []


def test_saving_unchanged_text_keeps_no_draft_and_stays_on_the_video(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    resp = ui.post(f"/videos/{V1}/draft", follow_redirects=False,
                   data={"title": "Repotting cactus #1",
                         "description": "How to repot cactus number 1.", "tags": "cactus,garden"})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#edit"
    assert get_draft(ui, V1) is None
    assert "no draft was kept" in ui.client.get(f"/videos/{V1}").text


def test_saving_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    resp = ui.post("/videos/nope/draft", data={"title": "x", "description": "", "tags": ""})
    assert resp.status_code == 404 and "Video not found" in resp.text


def test_absurdly_long_text_is_refused_with_a_message(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    resp = ui.post(f"/videos/{V1}/draft", data={"title": "x" * 1001, "description": "", "tags": ""})
    assert "far longer than YouTube allows" in resp.text
    assert get_draft(ui, V1) is None


# --- the changes page ------------------------------------------------------------------------------
def test_changes_page_empty_state_points_to_content(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    html = ui.client.get("/changes").text
    assert "No changes waiting" in html and 'href="/videos"' in html
    assert 'id="nav-drafts"' not in html
    assert 'href="/changes"' in html and 'href="/history"' in html  # sidebar items


def test_changes_page_shows_the_diff_and_the_issues(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1, title="Repotting cactus #1 <updated>", description="Brand new text",
         tags=["cactus", "succulent"])
    html = ui.client.get("/changes").text
    card = html.split(f'id="d-{V1}"')[1].split("</article>")[0]
    assert "0 approved" in html and "1 draft, 0 approved" in html
    assert "Changes: title; description (1 line added, 1 removed); tags (1 added, 1 removed)" in card
    # title as word segments
    assert '<ins class="seg added">' in card and "&lt;updated&gt;" in card
    # description lines, with line numbers and markers that do not rely on colour
    assert 'class="dl removed"' in card and "How to repot cactus number 1." in card
    assert 'class="dl added"' in card and "Brand new text" in card
    assert "Removed line: " in card and "Added line: " in card
    # tags as chips
    assert 'class="tagchip added"' in card and "succulent" in card
    assert 'class="tagchip removed"' in card and "garden" in card
    assert 'class="tagchip kept"' in card
    # validation issues, errors visibly distinct from warnings
    assert 'class="issue error"' in card and "<strong>Error:</strong>" in card
    # an error blocks approval
    assert re.search(r'/approve"><input[^>]*><button type="submit" class="button primary" disabled', card)
    assert "Fix the errors first." in card
    assert 'id="nav-drafts"' in html


def test_diff_shows_gaps_for_unchanged_stretches(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    old = "\n".join(f"line {i}" for i in range(1, 21))
    with Store(ui.config.catalog_db) as store:
        video = store.get_video(V1)
        store.update_video_text(V1, video.title, old, video.tags)
    save(ui, V1, description=old.replace("line 10", "line ten"))
    html = ui.client.get("/changes").text
    assert 'class="dl gap"' in html and "unchanged lines" in html
    assert "line ten" in html


def test_most_viewed_draft_comes_first(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    for vid in (V1, V3, V2):
        save(ui, vid)
    html = ui.client.get("/changes").text
    order = re.findall(r'<article class="card change[^"]*" id="d-(\w+)"', html)
    assert order == [V3, V2, V1]


def test_approve_unapprove_and_the_push_button(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1)
    html = ui.post(f"/changes/{V1}/approve").text
    assert "Approved." in html and "1 draft, 1 approved" in html
    assert "Review and push 1 approved" in html and 'href="/changes/push"' in html
    assert '<span class="pill done">Approved</span>' in html
    resp = ui.post(f"/changes/{V1}/unapprove", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/changes#d-{V1}"
    html = ui.client.get("/changes").text
    assert "1 draft, 0 approved" in html
    assert '<span class="button primary disabled" aria-disabled="true">Review and push 0 approved' in html


def test_approve_is_refused_with_a_message_when_there_are_errors(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1, title="<b>")
    html = ui.post(f"/changes/{V1}/approve").text
    assert "Fix the problems marked as errors" in html
    assert get_draft(ui, V1).approved_at is None
    html = ui.post("/changes/vid999999/approve").text
    assert "There is no draft for this video." in html


def test_discard_removes_the_draft(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1)
    html = ui.post(f"/changes/{V1}/discard").text
    assert "Draft discarded." in html and "No changes waiting" in html
    assert get_draft(ui, V1) is None


def test_stale_draft_gets_a_notice_and_cannot_be_approved(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1, title="My new title", approve=True)
    with Store(ui.config.catalog_db) as store:  # the video changed after the draft was made
        video = store.get_video(V1)
        store.update_video_text(V1, "Edited in Studio", video.description, video.tags)
    html = ui.client.get("/changes").text
    card = html.split(f'id="d-{V1}"')[1].split("</article>")[0]
    assert "YouTube's text changed since you started this draft" in card
    assert "Use the new text as the base" in card and f'/changes/{V1}/rebase"' in card
    assert card.count(f'/changes/{V1}/discard"') == 1
    assert "Update the draft to YouTube&#39;s new text first." in card
    assert re.search(r'/approve"><input[^>]*><button type="submit" class="button primary" disabled', card)
    assert "0 approved" in html  # the earlier approval no longer counts
    assert "text changed after this draft was made" in ui.post(f"/changes/{V1}/approve").text

    html = ui.post(f"/changes/{V1}/rebase").text
    assert "builds on YouTube" in html and "changed since you started this draft" not in html
    assert get_draft(ui, V1).base_title == "Edited in Studio"
    assert "Edited in Studio" in html


def test_rebase_onto_identical_text_removes_the_draft(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1, title="Same as Studio")
    with Store(ui.config.catalog_db) as store:
        video = store.get_video(V1)
        store.update_video_text(V1, "Same as Studio", video.description, video.tags)
    html = ui.post(f"/changes/{V1}/rebase").text
    assert "so it was removed" in html and get_draft(ui, V1) is None


# --- the push confirmation ---------------------------------------------------------------------------
def test_push_page_with_nothing_approved(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    save(ui, V1)  # a draft, but not approved
    html = ui.client.get("/changes/push").text
    assert "Nothing approved yet" in html and 'action="/changes/push"' not in html


def test_push_page_without_write_access_asks_for_permission(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path, write_access=False)
    save(ui, V1, approve=True)
    html = ui.client.get("/changes/push").text
    assert "Allow Yezatas to edit your videos" in html and 'action="/changes/allow-edits"' in html
    assert "verified this app" in html and "reading still works" in html.lower()
    # The explainer: Google's two pages drawn as slides, what the permission is used for, how to remove it.
    assert "aria-label=\"Google's pages for the edit permission\"" in html
    assert "See, edit, and permanently delete your YouTube videos" in html
    assert "<strong>videos.update</strong> for a video's title," in html
    assert "<strong>thumbnails.set</strong> when you confirm a new thumbnail" in html
    assert "never deletes" in html and "https://myaccount.google.com/connections" in html
    assert 'id="about-edit-access"' not in html
    assert 'action="/changes/push"' not in html and "Push to YouTube</button>" not in html
    assert "Quota cost" in html  # the cost is shown before asking for anything
    assert writer.calls == []


def test_push_page_with_write_access_shows_the_cost_and_the_form(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, title="Fresh title", approve=True)
    save(ui, V2, approve=True)
    save(ui, V3)  # not approved: stays out
    html = ui.client.get("/changes/push").text
    assert "2 approved changes will be sent to YouTube" in html
    assert "Fresh title" in html and "Changes: title" in html
    assert "1 other draft is not approved" in html
    # videos.list x1 + videos.update x2 = 1 + 100
    assert "(videos.list)" in html and "(videos.update)" in html and "<strong>101</strong>" in html
    assert "This push fits." in html
    assert 'action="/changes/push"' in html and "Push to YouTube</button>" in html
    assert re.search(r'<input type="checkbox" name="confirm" value="yes" required>', html)
    assert "I approve publishing these 2 changes to YouTube" in html
    assert set(fingerprints(html)) == {V1, V2}
    assert 'id="about-edit-access"' in html and "https://myaccount.google.com/connections" in html
    assert "Google's pages for the edit permission" not in html  # no slides once access is given
    assert 'action="/changes/allow-edits"' not in html
    assert writer.calls == []


def test_push_page_says_when_the_quota_does_not_fit(tmp_path: Path) -> None:
    config = make_config(tmp_path, daily_quota_budget=40)
    ui, _ = make_push_ui(tmp_path, config=config)
    save(ui, V1, approve=True)
    html = ui.client.get("/changes/push").text
    assert "does not fit" in html and "midnight Pacific time" in html
    assert 'action="/changes/push"' not in html


def test_allow_edits_runs_a_write_sign_in_job(tmp_path: Path) -> None:
    signed = {"in": False}

    def write_sign_in(config: Any) -> Any:
        signed["in"] = True
        return fake_creds()

    ui, _ = make_push_ui(tmp_path, write_access=False, write_sign_in=write_sign_in,
                         write_credentials_loader=lambda c: fake_creds() if signed["in"] else None)
    write_client_file(ui.config)
    save(ui, V1, approve=True)
    assert 'action="/changes/allow-edits"' in ui.client.get("/changes/push").text
    resp = ui.post("/changes/allow-edits")
    assert resp.status_code == 200
    ui.web.write_job.wait()
    assert ui.web.write_job.snapshot().status == "done" and signed["in"]
    html = ui.client.get("/changes/push").text
    assert 'action="/changes/push"' in html and 'action="/changes/allow-edits"' not in html


def test_allow_edits_needs_the_client_file_and_reports_failures(tmp_path: Path) -> None:
    def denied(config: Any) -> Any:
        raise AccessDeniedError()

    ui, _ = make_push_ui(tmp_path, write_access=False, write_sign_in=denied)
    save(ui, V1, approve=True)
    assert "Add the OAuth client file first" in ui.post("/changes/allow-edits").text
    write_client_file(ui.config)
    ui.post("/changes/allow-edits")
    ui.web.write_job.wait()
    html = ui.client.get("/changes/push").text
    assert "access_denied" in html and 'action="/changes/allow-edits"' in html


def test_the_write_sign_in_can_be_cancelled(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(config: Any) -> Any:
        entered.set()
        gate.wait(5)

    ui, _ = make_push_ui(tmp_path, write_access=False, write_sign_in=slow)
    write_client_file(ui.config)
    save(ui, V1, approve=True)
    ui.post("/changes/allow-edits")
    assert entered.wait(5)
    html = ui.client.get("/changes/push").text
    assert 'id="allow-waiting"' in html and 'data-poll="1"' in html
    assert ui.client.get("/api/status").json()["write"]["status"] == "running"
    ui.post("/changes/allow-edits/cancel")
    assert ui.web.write_job.snapshot().status == "failed"
    gate.set()


# --- pushing ------------------------------------------------------------------------------------------
def test_push_without_the_box_ticked_is_refused(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, approve=True)
    fps = fingerprints(ui.client.get("/changes/push").text)
    html = ui.post("/changes/push", data=push_form(fps, confirm=False)).text
    assert "Tick the box" in html
    assert ui.web.push_job.snapshot().status == "idle" and writer.calls == []


def test_push_without_write_access_is_refused(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path, write_access=False)
    save(ui, V1, approve=True)
    with Store(ui.config.catalog_db) as store:
        fps = {V1: edits.fingerprint(store.get_draft(V1))}
    html = ui.post("/changes/push", data=push_form(fps)).text
    assert "permission to edit your videos" in html
    assert ui.web.push_job.snapshot().status == "idle" and writer.calls == []


def test_push_with_nothing_posted_is_refused(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    assert "nothing to push" in ui.post("/changes/push", data={"confirm": "yes"}).text
    assert writer.calls == []


def test_push_runs_and_the_result_lists_pushed_and_changed_in_studio(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, title="Pushed title", description="Pushed text", tags=["x"], approve=True)
    save(ui, V2, approve=True)
    writer.studio_edit(V2, title="Edited in Studio meanwhile")
    fps = fingerprints(ui.client.get("/changes/push").text)
    resp = ui.post("/changes/push", data=push_form(fps), follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes/push"
    ui.web.push_job.wait()
    assert ui.web.push_job.snapshot().status == "done"

    html = ui.client.get("/changes/push").text
    assert "Push finished" in html and "1 pushed, 1 changed in Studio" in html
    assert "Quota used: 51 units" in html
    pushed_row = html.split(f'id="r-{V1}"')[1].split("</li>")[0]
    assert ">Pushed<" in pushed_row and f'href="/videos/{V1}"' in pushed_row
    assert "Pushed title" in pushed_row
    conflict_row = html.split(f'id="r-{V2}"')[1].split("</li>")[0]
    assert "Changed in Studio, not pushed" in conflict_row and "Nothing was pushed" in conflict_row
    assert f'href="/changes#d-{V2}"' in conflict_row
    # the fake channel received exactly one update, with the draft's text
    (body, part), = writer.updates
    assert body["id"] == V1 and body["snippet"]["title"] == "Pushed title"
    assert body["snippet"]["tags"] == ["x"] and part == "snippet"
    # the pushed draft is gone, the conflicted one is now stale
    assert get_draft(ui, V1) is None and get_draft(ui, V2) is not None
    assert "changed since you started this draft" in ui.client.get("/changes").text


def test_a_draft_edited_after_the_confirmation_was_rendered_is_skipped(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, title="First version", approve=True)
    save(ui, V2, approve=True)
    fps = fingerprints(ui.client.get("/changes/push").text)
    save(ui, V1, title="Second version", approve=True)  # edited and approved again meanwhile
    ui.post("/changes/push", data=push_form(fps))
    ui.web.push_job.wait()
    html = ui.client.get("/changes/push").text
    skipped = html.split(f'id="r-{V1}"')[1].split("</li>")[0]
    assert "Skipped" in skipped and "changed after you reviewed it" in skipped
    assert [b["id"] for b, _ in writer.updates] == [V2]
    assert get_draft(ui, V1).title == "Second version"


def test_a_failed_update_is_listed_with_its_message(tmp_path: Path) -> None:
    from fakes import make_http_error

    ui, writer = make_push_ui(tmp_path)
    writer.update_errors[V1] = make_http_error(400, "invalidVideoMetadata", "Bad metadata.")
    html = pushed(ui)
    row = html.split(f'id="r-{V1}"')[1].split("</li>")[0]
    assert ">Failed<" in row and "invalidVideoMetadata" in row
    assert ">Pushed<" in html.split(f'id="r-{V2}"')[1].split("</li>")[0]


def test_push_progress_is_shown_while_it_runs_and_a_second_push_is_refused(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, approve=True)
    fps = fingerprints(ui.client.get("/changes/push").text)
    entered, gate = threading.Event(), threading.Event()
    writer.on_update = lambda body: (entered.set(), gate.wait(5))
    ui.post("/changes/push", data=push_form(fps))
    assert entered.wait(5)
    html = ui.client.get("/changes/push").text
    assert 'id="push-progress"' in html and 'aria-live="polite"' in html
    assert 'data-poll="1"' in html and 'id="push-msg"' in html
    assert "A push to YouTube is running." in ui.client.get("/changes").text
    assert "already running" in ui.post("/changes/push", data=push_form(fps)).text
    assert ui.client.get("/api/status").json()["push"]["status"] == "running"
    gate.set()
    ui.web.push_job.wait()
    assert ui.web.push_job.snapshot().status == "done"


def test_push_that_does_not_fit_fails_before_any_call(tmp_path: Path) -> None:
    config = make_config(tmp_path, daily_quota_budget=40)
    ui, writer = make_push_ui(tmp_path, config=config)
    save(ui, V1, approve=True)
    with Store(ui.config.catalog_db) as store:
        fps = {V1: edits.fingerprint(store.get_draft(V1))}
    ui.post("/changes/push", data=push_form(fps))
    ui.web.push_job.wait()
    html = ui.client.get("/changes/push").text
    assert "The push stopped" in html and "Not enough quota left" in html
    assert "midnight Pacific time" in html
    assert writer.calls == []
    assert get_draft(ui, V1) is not None


def test_dismissing_a_finished_push_returns_to_the_changes(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    assert "Push finished" in pushed(ui)
    resp = ui.post("/changes/push/done", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    assert "Push finished" not in ui.client.get("/changes/push").text
    assert "push-banner" not in ui.client.get("/changes").text


# --- history and restore ------------------------------------------------------------------------------
def test_history_lists_pushes_with_backups_and_restore_makes_a_draft(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    assert "Nothing pushed yet" in ui.client.get("/history").text
    pushed(ui)
    html = ui.client.get("/history").text
    assert "Before every push, Yezatas saves YouTube" in html
    assert html.count("Restore this text") == 2 and html.count("<details") == 2
    assert "Pushed" in html and "Push #2" in html and "Push #1" in html
    assert html.index("Push #2") < html.index("Push #1")  # newest first
    assert "Repotting cactus #1" in html and "How to repot cactus number 1." in html  # the backup
    assert "cactus, garden" in html

    writer.calls.clear()
    # the most viewed draft (V2) is pushed first, so push #2 belongs to V1
    resp = ui.post("/history/2/restore", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/changes#d-{V1}"
    row = get_draft(ui, V1)
    assert row.source == "restore" and row.title == "Repotting cactus #1"
    assert row.note == "Restore of the text from before push #2" and row.approved_at is None
    html = ui.client.get("/changes").text
    assert "Restore of the text from before push #2" in html and "Restore draft made" in html
    assert writer.calls == []  # restoring touches nothing on YouTube


def test_restoring_text_that_is_already_live_says_so(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, approve=True)
    writer.studio_edit(V1, title="Changed by hand")
    fps = fingerprints(ui.client.get("/changes/push").text)
    ui.post("/changes/push", data=push_form(fps))
    ui.web.push_job.wait()
    html = ui.client.get("/history").text
    assert "Changed in Studio, not pushed" in html
    assert "This video already has a draft: push or discard it before restoring." in html
    resp = ui.post("/history/1/restore")
    assert "already has a draft" in resp.text
    ui.post(f"/changes/{V1}/discard")
    # the catalog already shows the live text, so restoring that backup changes nothing
    resp = ui.post("/history/1/restore")
    assert "That text is already on YouTube; nothing to restore." in resp.text
    assert get_draft(ui, V1) is None


def test_restoring_an_unknown_record_shows_a_message(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    assert "does not exist" in ui.post("/history/99/restore").text


# --- links from other pages ------------------------------------------------------------------------
def test_content_table_and_dashboard_point_at_drafts(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path)
    assert "waiting for review" not in ui.client.get("/").text
    save(ui, V1)
    html = ui.client.get("/videos").text
    assert html.count("draft-pill") == 1
    row = html.split(f'href="/videos/{V1}">')[1].split("</tr>")[0]
    assert ">Draft<" in row
    dash = ui.client.get("/").text
    assert "1 change waiting for review" in dash and 'href="/changes"' in dash
    save(ui, V2)
    assert "2 changes waiting for review" in ui.client.get("/").text
    assert 'id="nav-drafts">2<' in ui.client.get("/videos").text


# --- safety rules -----------------------------------------------------------------------------------
def test_no_get_ever_calls_youtube(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, approve=True)
    save(ui, V2)
    for path in ("/", "/videos", f"/videos/{V1}", "/changes", "/changes/push", "/history",
                 "/api/status", "/setup"):
        assert ui.client.get(path).status_code == 200, path
    assert writer.calls == [] and writer.updates == []
    pushed(ui)
    writer.calls.clear()
    for path in ("/changes", "/changes/push", "/history"):
        assert ui.client.get(path).status_code == 200, path
    assert writer.calls == []


@pytest.mark.parametrize("path", [
    f"/videos/{V1}/draft", f"/changes/{V1}/approve", f"/changes/{V1}/unapprove",
    f"/changes/{V1}/discard", f"/changes/{V1}/rebase", "/changes/push", "/changes/push/done",
    "/changes/allow-edits", "/changes/allow-edits/cancel", "/history/1/restore",
])
def test_every_new_post_requires_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, writer = make_push_ui(tmp_path)
    save(ui, V1, approve=True)
    resp = ui.client.post(path, data={"title": "x", "description": "", "tags": "", "confirm": "yes",
                                      f"fp_{V1}": "abc"})
    assert resp.status_code == 403
    resp = ui.client.post(path, data={"csrf": "wrong", "confirm": "yes"})
    assert resp.status_code == 403
    assert writer.calls == [] and ui.web.push_job.snapshot().status == "idle"
    assert get_draft(ui, V1).title.endswith("(new)") and get_draft(ui, V1).approved_at


def test_pages_need_a_signed_in_browser(tmp_path: Path) -> None:
    ui, _ = make_push_ui(tmp_path, signed_in_browser=False)
    for path in ("/changes", "/changes/push", "/history"):
        assert ui.client.get(path).status_code == 403, path
