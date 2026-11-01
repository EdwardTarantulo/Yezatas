"""The Bulk edit Ask card: asking with a fake model, the proposal, previewing, making drafts,
tampered plans, no-key, CSRF and escaping. No model or network call, and nothing reaches YouTube."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import bulkask, edits
from yezatas.agent import ModelError, ModelReply
from yezatas.store import Store
from yezatas.web.app import NO_PACKAGING_KEY

A, B, C, D, E = (f"vid{n:08d}" for n in range(1, 6))
OLD = "https://shop.example.com/old"
NEW = "https://shop.example.com/new"
REQUEST = f"Swap {OLD} for {NEW} on videos from 2024"
EXPLAIN_FIND = f'Replace "{OLD}"'
EVIL = "<script>alert(1)</script>"


def reply(data: Any) -> ModelReply:
    return ModelReply(data if isinstance(data, str) else json.dumps(data), (), {}, 3000, 200, 150)


def good_answer(**step_override: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "kind": "replace", "find": OLD, "replace_with": NEW, "fields": ["description"],
        "footer": "", "marker": "", "text": "", "contains": ""}
    step.update(step_override)
    return {"steps": [step],
            "scope": {"visibility": "public", "published_from": "2024-01-01",
                      "published_to": "", "title_contains": ""},
            "cannot": ""}


def good_plan_dict() -> dict[str, Any]:
    return good_answer()


def asker(script: list[Any]) -> tuple[Any, FakeModelClient]:
    client = FakeModelClient(script)
    return (lambda store, req: bulkask.propose(client, store, req)), client


def catalog(ui: Any) -> None:
    seed_catalog(ui.config, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description=f"Intro line\nShop: {OLD}\nBye", view_count=900,
                   published_at="2024-02-01T10:00:00Z"),
        make_video(2, video_id=B, privacy_status="public", title="Cactus two",
                   description=f"Shop: {OLD} and again {OLD}", view_count=500,
                   published_at="2024-03-01T10:00:00Z"),
        make_video(3, video_id=C, privacy_status="public", title="Cactus three",
                   description="Nothing to see here", view_count=300,
                   published_at="2024-04-01T10:00:00Z"),
        make_video(4, video_id=D, privacy_status="private", title="Cactus four",
                   description=f"Private: {OLD}", view_count=200,
                   published_at="2024-05-01T10:00:00Z"),
        make_video(5, video_id=E, privacy_status="public", title="Cactus five",
                   description=f"Draft already: {OLD}", view_count=100,
                   published_at="2024-06-01T10:00:00Z"),
    ])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, E, "Cactus five (edited)", f"Draft already: {OLD}", ["cactus"])


def make_bulk_ui(tmp_path: Path, script: list[Any] | None = None, key: bool = True,
                 bulk_asker: Any = None) -> Any:
    client: FakeModelClient | None = None
    if bulk_asker is None:
        bulk_asker, client = asker(script if script is not None else [reply(good_answer())])
    ui = make_ui(tmp_path, bulk_asker=bulk_asker, packaging_key=lambda cfg: key)
    catalog(ui)
    ui.ask_client = client
    return ui


def drafts(ui: Any) -> dict[str, Any]:
    with Store(ui.config.catalog_db) as store:
        return {d.video_id: d for d in store.list_drafts()}


def proposal_fields(html_text: str) -> tuple[str, str]:
    """The (request, plan) hidden fields, HTML-unescaped."""
    request = re.search(r'name="request" value="([^"]*)"', html_text)
    plan = re.search(r'name="plan" value="([^"]*)"', html_text)
    assert request and plan, "the proposal hidden fields are missing"
    return unescape(request.group(1)), unescape(plan.group(1))


def ask_good(ui: Any, request_text: str = REQUEST) -> str:
    resp = ui.post("/bulk/ask", data={"request": request_text}, follow_redirects=False)
    assert resp.status_code == 200
    return resp.text


def preview_good(ui: Any) -> tuple[str, str, str]:
    req, plan_json = proposal_fields(ask_good(ui))
    resp = ui.post("/bulk/ask/preview", data={"request": req, "plan": plan_json},
                   follow_redirects=False)
    assert resp.status_code == 200
    return resp.text, req, plan_json


# --- the page --------------------------------------------------------------------------------------
def test_get_bulk_shows_the_ask_card_and_the_plain_form(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.client.get("/bulk")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="bulk-ask"' in html and "Describe the change" in html
    assert 'id="bulk-ask-form"' in html and 'action="/bulk/ask"' in html
    assert 'id="ba-request"' in html and 'name="request"' in html
    assert 'maxlength="1000"' in html and bulkask.REQUEST_MAX == 1000
    assert 'data-count-limit="1000"' in html and 'data-count-unit="characters"' in html
    assert 'data-count-for="ba-count"' in html
    assert 'id="ba-count"' in html and "Up to 1000 characters." in html
    assert "Muse Spark only picks steps from a fixed list." in html
    assert "Any new text must be in your words; code makes the changes, and only as drafts." in html
    assert ("Your words, and the links and repeated lines from your descriptions, go to "
            "Meta's Model API (muse-spark-1.3).") in html
    assert "Nothing is sent to YouTube." in html
    form = html.split('id="bulk-ask-form"')[1].split("</form>")[0]
    assert "Plan it with Muse Spark" in form and "disabled" not in form
    assert 'id="bulk-ask-no-key"' not in html
    assert 'id="bulk-ask-proposal"' not in html and 'id="bulk-preview"' not in html
    assert 'id="bulk-form"' in html and 'name="find"' in html and 'name="footer"' in html
    assert ui.ask_client.calls == []


def test_no_key_disables_asking_and_sends_nothing(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path, key=False)
    html = ui.client.get("/bulk").text
    assert 'id="bulk-ask-no-key"' in html
    assert "There is no Meta API key. Add one in Setup. Nothing was sent." in html
    form = html.split('id="bulk-ask-form"')[1].split("</form>")[0]
    assert 'class="button primary" disabled' in form or "button primary\" disabled" in form
    resp = ui.post("/bulk/ask", data={"request": REQUEST}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert NO_PACKAGING_KEY in ui.client.get("/bulk").text
    assert ui.ask_client.calls == []
    assert set(drafts(ui)) == {E}


# --- asking ----------------------------------------------------------------------------------------
def test_ask_success_shows_the_proposal_without_storing_it(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    html = ask_good(ui)
    assert 'id="bulk-ask-proposal"' in html and "Proposed change" in html
    plain = unescape(html)
    assert EXPLAIN_FIND in plain and NEW in plain and "in descriptions." in plain
    assert "Only public videos published from 2024-01-01." in plain
    proposal = html.split('id="bulk-ask-proposal"')[1].split("</section>")[0]
    assert "muse-spark-1.3" in proposal and "3,150 tokens" in proposal and "cost $" in proposal
    assert 'action="/bulk/ask/preview"' in proposal and "Preview these changes" in proposal
    assert 'action="/bulk/ask/discard"' in proposal and "Discard" in proposal
    req, plan_json = proposal_fields(proposal)
    assert req == REQUEST
    data = json.loads(plan_json)
    assert data["steps"][0]["kind"] == "replace"
    assert data["steps"][0]["find"] == OLD and data["steps"][0]["replace_with"] == NEW
    assert data["scope"]["visibility"] == "public"
    assert set(drafts(ui)) == {E}
    assert len(ui.ask_client.calls) == 1


def test_ask_cannot_is_shown_escaped_with_no_preview(tmp_path: Path) -> None:
    answer = {"steps": [], "scope": {"visibility": "public", "published_from": "",
                                     "published_to": "", "title_contains": ""},
              "cannot": EVIL}
    ui = make_bulk_ui(tmp_path, [reply(answer)])
    html = ask_good(ui)
    assert 'id="ba-cannot"' in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert 'action="/bulk/ask/preview"' not in html
    assert 'name="plan"' not in html.split('id="bulk-ask-proposal"')[1].split("</section>")[0]
    assert 'action="/bulk/ask/discard"' in html
    assert set(drafts(ui)) == {E}


def test_ask_cannot_shows_cost_line(tmp_path: Path) -> None:
    answer = {"steps": [], "scope": {"visibility": "public", "published_from": "",
                                     "published_to": "", "title_contains": ""},
              "cannot": "That needs rewriting whole descriptions, which these steps cannot do."}
    ui = make_bulk_ui(tmp_path, [reply(answer)])
    html = ask_good(ui)
    assert 'id="ba-cannot"' in html
    proposal = html.split('id="bulk-ask-proposal"')[1].split("</section>")[0]
    assert 'id="ba-cost"' in proposal
    assert "muse-spark-1.3" in proposal and "3,150 tokens" in proposal and "cost $" in proposal


def test_proposal_keeps_your_words_in_the_text_box(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    get_box = re.search(r'<textarea id="ba-request"[^>]*>(.*?)</textarea>',
                        ui.client.get("/bulk").text, re.S)
    assert get_box and unescape(get_box.group(1)) == ""
    html = ask_good(ui)
    box = re.search(r'<textarea id="ba-request"[^>]*>(.*?)</textarea>', html, re.S)
    assert box and unescape(box.group(1)) == REQUEST


def test_proposal_text_box_escapes_your_words(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    words = f"Swap {OLD} for {NEW} on videos from 2024 {EVIL}"
    html = ask_good(ui, request_text=words)
    assert "<script>alert(1)</script>" not in html
    box = re.search(r'<textarea id="ba-request"[^>]*>(.*?)</textarea>', html, re.S)
    assert box and unescape(box.group(1)) == words


def test_ask_with_a_model_error_redirects_with_a_flash(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path, [ModelError("rate_limit", "Slow down, then try again.")])
    resp = ui.post("/bulk/ask", data={"request": REQUEST}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert "Slow down, then try again." in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


def test_ask_with_an_unavailable_kind_redirects_with_a_flash(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path, [reply(good_answer(kind="fly"))])
    resp = ui.post("/bulk/ask", data={"request": REQUEST}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert "not available" in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


def test_ask_empty_request_never_calls_the_model(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.post("/bulk/ask", data={"request": "  "}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert ui.ask_client.calls == []
    assert "Describe the change in a few words first." in ui.client.get("/bulk").text


def test_discard_goes_back_to_bulk(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    ask_good(ui)
    resp = ui.post("/bulk/ask/discard", data={}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert 'id="bulk-ask-proposal"' not in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


# --- preview ---------------------------------------------------------------------------------------
def test_preview_shows_rows_and_writes_nothing(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    html, req, plan_json = preview_good(ui)
    assert "2 videos would change, 1 blocked" in html
    assert f'value="{A}"' in html and f'value="{B}"' in html and f'value="{E}"' in html
    assert f'value="{C}"' not in html and f'value="{D}"' not in html
    assert re.search(rf'name="video_id" value="{A}"[^>]* checked', html)
    assert re.search(rf'name="video_id" value="{B}"[^>]* checked', html)
    row = re.search(rf'<tr id="b-{E}"[^>]*>.*?</tr>', html, re.S)
    assert row and "disabled" in row.group(0) and "already has a draft" in row.group(0)
    assert f'href="/videos/{A}"' in html
    assert re.search(r'<del class="seg removed">[^<]*' + re.escape(OLD) + r'</del>', html)
    assert re.search(r'<ins class="seg added">[^<]*' + re.escape(NEW) + r'</ins>', html)
    form = html.split('id="bulk-apply"')[1].split("</form>")[0]
    assert 'action="/bulk/ask/apply"' in html
    preq, pplan = proposal_fields(form)
    assert preq == REQUEST and json.loads(pplan)["steps"][0]["find"] == OLD
    assert set(drafts(ui)) == {E}
    with Store(ui.config.catalog_db) as store:
        assert OLD in store.get_video(A).description


def test_preview_keeps_your_words_in_the_text_box(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    html, _req, _plan_json = preview_good(ui)
    box = re.search(r'<textarea id="ba-request"[^>]*>(.*?)</textarea>', html, re.S)
    assert box and unescape(box.group(1)) == REQUEST


# --- tampered plans ----------------------------------------------------------------------------------
def tampered(kind: str) -> str:
    data = good_plan_dict()
    if kind == "invented":
        data["steps"][0]["replace_with"] = "https://shop.example.com/invented"
    elif kind == "kind":
        data["steps"][0]["kind"] = "evil"
    elif kind == "date":
        data["scope"]["published_from"] = "2025-01-01"
    return json.dumps(data)


@pytest.mark.parametrize("kind, needle", [
    ("invented", "is not in your words"),
    ("kind", "not available"),
    ("date", "is not in your words"),
])
def test_tampered_plan_on_preview_makes_no_drafts(tmp_path: Path, kind: str, needle: str) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.post("/bulk/ask/preview", data={"request": REQUEST, "plan": tampered(kind)},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert needle in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


@pytest.mark.parametrize("kind, needle", [
    ("invented", "is not in your words"),
    ("kind", "not available"),
    ("date", "is not in your words"),
])
def test_tampered_plan_on_apply_makes_no_drafts(tmp_path: Path, kind: str, needle: str) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.post("/bulk/ask/apply",
                   data={"request": REQUEST, "plan": tampered(kind), "video_id": [A]},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert needle in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


def test_bad_plan_json_on_preview_makes_no_drafts(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.post("/bulk/ask/preview", data={"request": REQUEST, "plan": "not json"},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert "That plan cannot be used. Ask for it again." in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


# --- apply -------------------------------------------------------------------------------------------
def test_apply_makes_bulk_ask_drafts_for_ticked_only(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    _html, req, plan_json = preview_good(ui)
    resp = ui.post("/bulk/ask/apply",
                   data={"request": req, "plan": plan_json, "video_id": [A]},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    made = drafts(ui)
    assert set(made) == {A, E}
    assert made[A].source == "bulk-ask"
    assert made[A].description == f"Intro line\nShop: {NEW}\nBye"
    assert made[A].note.startswith("Bulk by asking")
    assert "Drafts made for 1 video." in ui.client.get("/changes").text


def test_apply_with_only_a_blocked_video_goes_back_to_bulk(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    _html, req, plan_json = preview_good(ui)
    resp = ui.post("/bulk/ask/apply",
                   data={"request": req, "plan": plan_json, "video_id": [E]},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert "No drafts were made." in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


def test_apply_with_nothing_ticked_goes_back_to_bulk(tmp_path: Path) -> None:
    ui = make_bulk_ui(tmp_path)
    _html, req, plan_json = preview_good(ui)
    resp = ui.post("/bulk/ask/apply",
                   data={"request": req, "plan": plan_json, "video_id": []},
                   follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/bulk"
    assert "Tick at least one video to make drafts." in ui.client.get("/bulk").text
    assert set(drafts(ui)) == {E}


# --- safety ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path, data", [
    ("/bulk/ask", {"request": REQUEST}),
    ("/bulk/ask/preview", {"request": REQUEST, "plan": "{}"}),
    ("/bulk/ask/apply", {"request": REQUEST, "plan": "{}", "video_id": [A]}),
    ("/bulk/ask/discard", {}),
])
def test_posts_without_csrf_are_refused(tmp_path: Path, path: str, data: dict[str, Any]) -> None:
    ui = make_bulk_ui(tmp_path)
    resp = ui.client.post(path, data=data, follow_redirects=False)
    assert resp.status_code == 403
    bad = ui.client.post(path, data={**data, "csrf": "wrong"}, follow_redirects=False)
    assert bad.status_code == 403
    assert ui.ask_client.calls == []
    assert set(drafts(ui)) == {E}
