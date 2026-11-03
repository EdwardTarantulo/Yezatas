"""The Custom checks page: asking for a check with a fake model, the proposal, saving, discarding,
the saved checks with their code-run counts, turning them off and deleting them, no-key,
unknown ids, CSRF and escaping. No model or network call."""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pytest
from markupsafe import escape

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, customchecks
from yezatas.agent import ModelError, ModelReply
from yezatas.store import Store
from yezatas.web.app import NO_PACKAGING_KEY

V1, V2, V3 = "vid00000001", "vid00000002", "vid00000003"
REQUEST = "Every public video links example.com"
EXPLAIN = "Every public video's description links example.com."
EVIL = "<script>alert(1)</script>"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


# --- fakes ---------------------------------------------------------------------------------------
def reply(data: Any) -> ModelReply:
    return ModelReply(data if isinstance(data, str) else json.dumps(data), (), {}, 3000, 200, 150)


def good_answer(**override: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "kind": "must_link", "domain": "example.com", "text": "", "field": "",
        "chars": 0, "count": 0, "visibility": "public"}
    answer.update(override)
    return answer


def maker(script: list[Any]) -> tuple[Any, FakeModelClient]:
    client = FakeModelClient(script)
    return (lambda req: customchecks.rule_from_request(client, req)), client


def make_checks_ui(tmp_path: Path, rule_maker: Any = "default", key: bool = True,
                   seeded: bool = True) -> Any:
    client: FakeModelClient | None = None
    if rule_maker == "default":
        rule_maker, client = maker([reply(good_answer())])
    elif isinstance(rule_maker, tuple):
        rule_maker, client = rule_maker
    ui = make_ui(tmp_path, rule_maker=rule_maker, packaging_key=lambda cfg: key)
    if seeded:
        seed_catalog(ui.config, [
            make_video(1, video_id=V1, title="Cactus guide",
                       description="See https://example.com/guide",
                       tags=["cactus", "garden"], privacy_status="public"),
            make_video(2, video_id=V2, title="Another much longer title here",
                       description="No link here", tags=[], privacy_status="public"),
            make_video(3, video_id=V3, title="Private video",
                       description="No link here either", tags=[], privacy_status="private")])
    ui.maker_client = client
    return ui


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any) -> str:
    resp = ui.client.get("/checks")
    assert resp.status_code == 200
    return resp.text


def stored(ui: Any) -> list[Any]:
    with Store(ui.config.catalog_db) as store:
        return store.list_custom_rules()


def add_rule(ui: Any, name: str = "A check", request: str = REQUEST,
             rule: dict[str, Any] | None = None) -> int:
    with Store(ui.config.catalog_db) as store:
        return store.add_custom_rule(name, request, rule or good_answer_rule())


def good_answer_rule() -> dict[str, Any]:
    return {"kind": "must_link", "domain": "example.com", "visibility": "public"}


def proposal_fields(html_text: str) -> tuple[str, str]:
    """The (request, rule_json) hidden fields of the proposal, HTML-unescaped."""
    request = re.search(r'name="request" value="([^"]*)"', html_text)
    rule = re.search(r'name="rule" value="([^"]*)"', html_text)
    assert request and rule, "the proposal hidden fields are missing"
    return unescape(request.group(1)), unescape(rule.group(1))


def ask_good(ui: Any) -> str:
    resp = nf(ui, "/checks/ask", data={"request": REQUEST})
    assert resp.status_code == 200
    return resp.text


def save_good(ui: Any, name: str = "") -> Any:
    request, rule_json = proposal_fields(ask_good(ui))
    return nf(ui, "/checks/save", data={"name": name, "request": request, "rule": rule_json})


def check_row(html_text: str, rule_id: int) -> str:
    marker = f'id="check-{rule_id}"'
    assert marker in html_text
    return html_text.split(marker, 1)[1].split('id="check-', 1)[0]


# --- the page ------------------------------------------------------------------------------------
def test_page_renders_the_ask_box(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    html = page(ui)
    assert "<title>Checks - Yezatas</title>" in html
    assert "<h1 class=\"page-title\">Checks</h1>" in html
    assert 'id="checks-about"' in html
    assert "The model only picks a rule from a fixed list; the check itself is code." in html
    assert 'id="checks-ask"' in html and "Ask for a check" in html
    assert 'id="checks-form"' in html and 'action="/checks/ask"' in html
    assert 'name="csrf"' in html
    assert 'id="ck-request"' in html and 'name="request"' in html
    assert 'maxlength="300"' in html and customchecks.REQUEST_MAX == 300
    assert 'data-count-limit="300"' in html and 'data-count-for="ck-count"' in html
    assert 'id="ck-count"' in html and "Up to 300 characters." in html
    assert "muse-spark-1.3" in html  # the privacy hint names the model
    assert 'class="button primary">Make the check<' in html
    assert "disabled" not in html.split('id="checks-form"')[1].split("</form>")[0]
    assert 'id="checks-no-key"' not in html
    assert 'id="checks-proposal"' not in html
    assert 'id="checks-list"' in html and "Your checks" in html
    assert 'id="checks-empty"' in html and "No checks yet. Ask for one above." in html
    assert ui.maker_client.calls == []


def test_no_key_disables_asking_with_a_plain_reason(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, key=False)
    html = page(ui)
    assert 'id="checks-no-key"' in html
    assert "There is no Meta API key yet, so Yezatas cannot turn your words into a check." in html
    assert '<a href="/configuration#step-ai">Add one in Configuration</a>' in html
    form = html.split('id="checks-form"')[1].split("</form>")[0]
    assert "button primary\" disabled>" in form or 'class="button primary" disabled' in form


# --- asking --------------------------------------------------------------------------------------
def test_ask_success_shows_the_proposal_without_storing_it(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    html = ask_good(ui)
    assert 'id="checks-proposal"' in html and "Proposed check" in html
    assert 'id="ck-explain">' in html and EXPLAIN in unescape(html)
    proposal = html.split('id="checks-proposal"')[1].split("</section>")[0]
    assert "muse-spark-1.3" in proposal
    assert "3,150 tokens" in proposal
    assert "cost $" in proposal
    assert 'action="/checks/save"' in proposal and 'action="/checks/discard"' in proposal
    request, rule_json = proposal_fields(proposal)
    assert request == REQUEST
    assert "must_link" in rule_json
    assert json.loads(rule_json) == good_answer_rule()
    name = re.search(r'id="ck-name" name="name" value="([^"]*)"', proposal)
    assert name and unescape(name.group(1)) == EXPLAIN
    assert "Save this check" in proposal and "Discard" in proposal
    assert stored(ui) == []
    assert len(ui.maker_client.calls) == 1


def test_ask_with_an_unavailable_kind_redirects_with_a_flash(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, maker([reply(good_answer(kind="nope"))]))
    resp = nf(ui, "/checks/ask", data={"request": REQUEST})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert "not available" in page(ui)
    assert stored(ui) == []


def test_ask_with_a_model_error_redirects_with_a_flash(tmp_path: Path) -> None:
    ui = make_checks_ui(
        tmp_path, maker([ModelError("rate_limit", "Slow down, then try again.")]))
    resp = nf(ui, "/checks/ask", data={"request": REQUEST})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert "Slow down, then try again." in page(ui)
    assert stored(ui) == []


def test_ask_empty_request_never_calls_the_model(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, maker([reply(good_answer())]))
    resp = nf(ui, "/checks/ask", data={"request": "  "})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert ui.maker_client.calls == []
    assert "Describe the check in a few words first." in page(ui)
    assert stored(ui) == []


def test_ask_too_long_request_never_calls_the_model(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, maker([reply(good_answer())]))
    resp = nf(ui, "/checks/ask", data={"request": "x" * 301})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert ui.maker_client.calls == []
    assert "at most 300 characters" in page(ui)
    assert stored(ui) == []


# --- saving and discarding -----------------------------------------------------------------------
def test_save_stores_the_rule_and_names_it_after_the_explanation(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    resp = save_good(ui, name="")
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    rules = stored(ui)
    assert len(rules) == 1
    assert rules[0].name == EXPLAIN
    assert rules[0].request == REQUEST
    assert rules[0].rule == good_answer_rule()
    assert rules[0].enabled is True
    html = page(ui)
    assert "Check saved." in html
    assert 'id="checks-proposal"' not in html


def test_save_keeps_a_given_name(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    resp = save_good(ui, name="My link check")
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    rules = stored(ui)
    assert len(rules) == 1 and rules[0].name == "My link check"
    assert "My link check" in page(ui)


def test_save_with_a_tampered_kind_stores_nothing(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    resp = nf(ui, "/checks/save", data={
        "name": "Evil", "request": REQUEST, "rule": '{"kind":"evil","visibility":"public"}'})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui) == []
    html = page(ui)
    assert "Check saved." not in html
    assert "not available" in html


def test_save_with_a_tampered_domain_stores_nothing(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    resp = nf(ui, "/checks/save", data={
        "name": "Evil", "request": REQUEST,
        "rule": '{"kind":"must_link","domain":"https://example.com","visibility":"public"}'})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui) == []
    html = page(ui)
    assert "Check saved." not in html
    assert "bare domain" in html


def test_discard_stores_nothing(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    proposal_fields(ask_good(ui))
    resp = nf(ui, "/checks/discard")
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui) == []
    assert 'id="checks-proposal"' not in page(ui)


# --- your checks ---------------------------------------------------------------------------------
def test_saved_check_shows_counts_and_failing_links(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    save_good(ui)
    rules = stored(ui)
    html = page(ui)
    row = check_row(html, rules[0].id)
    assert EXPLAIN in unescape(row)
    assert "1 passing, 1 not applicable" in row
    assert f'<a href="/videos/{V2}">Another much longer title here</a>' in row
    assert f'href="/videos/{V1}"' not in row


def test_only_the_first_twenty_failing_videos_are_listed(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, seeded=False)
    seed_catalog(ui.config, [
        make_video(i, video_id=f"vid{i:08d}", title=f"Failing video {i}",
                   description="Nothing wanted here", tags=[], privacy_status="public",
                   view_count=(26 - i) * 100)
        for i in range(1, 26)])
    add_rule(ui, name="Absent check", request="titles contain absent phrase",
             rule={"kind": "must_contain", "text": "absent phrase", "field": "title",
                   "visibility": "public"})
    html = page(ui)
    listing = html.split('id="checks-list"')[1]
    links = re.findall(r'<a href="/videos/([^"]+)">', listing)
    assert len(links) == 20
    assert links[0] == "vid00000001" and links[-1] == "vid00000020"
    assert "and 5 more" in listing
    assert "0 passing, 0 not applicable" in listing


def test_a_check_that_cannot_be_run_shows_the_results_error(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    rule_id = add_rule(ui, name="Broken check", rule={"kind": "evil", "visibility": "public"})
    html = page(ui)
    assert 'id="checks-error"' in html
    row = check_row(html, rule_id)
    assert "Broken check" in row
    assert 'class="ck-counts"' not in row and "Turned off." not in row


def test_toggle_turns_a_check_off_and_back_on(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    save_good(ui)
    rule_id = stored(ui)[0].id
    resp = nf(ui, f"/checks/{rule_id}/toggle", data={"enabled": "0"})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui)[0].enabled is False
    row = check_row(page(ui), rule_id)
    assert '<span class="pill">Off</span>' in row
    assert "Turned off." in row
    assert 'class="ck-counts"' not in row and "passing," not in row
    assert ">Turn on<" in row
    resp = nf(ui, f"/checks/{rule_id}/toggle", data={"enabled": "1"})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui)[0].enabled is True
    row = check_row(page(ui), rule_id)
    assert "1 passing, 1 not applicable" in row and ">Turn off<" in row


def test_delete_removes_a_check(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    rule_id = add_rule(ui)
    resp = nf(ui, f"/checks/{rule_id}/delete")
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert stored(ui) == []
    html = page(ui)
    assert f'id="check-{rule_id}"' not in html
    assert 'id="checks-empty"' in html


def test_unknown_toggle_and_delete_flash_a_plain_sentence(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    resp = nf(ui, "/checks/99999/toggle", data={"enabled": "0"})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert "This check does not exist." in page(ui)
    resp = nf(ui, "/checks/99999/delete")
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert "This check does not exist." in page(ui)
    assert stored(ui) == []


# --- safety --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path, data", [
    ("/checks/ask", {"request": REQUEST}),
    ("/checks/save", {"name": "X", "request": REQUEST,
                      "rule": '{"kind":"must_link","domain":"example.com","visibility":"public"}'}),
    ("/checks/discard", {}),
])
def test_posts_without_csrf_are_refused(tmp_path: Path, path: str, data: dict[str, str]) -> None:
    ui = make_checks_ui(tmp_path, maker([reply(good_answer())]))
    resp = ui.client.post(path, data=data, follow_redirects=False)
    assert resp.status_code == 403
    bad = ui.client.post(path, data={**data, "csrf": "wrong"}, follow_redirects=False)
    assert bad.status_code == 403
    assert ui.maker_client.calls == []
    assert stored(ui) == []


def test_toggle_and_delete_without_csrf_are_refused(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    rule_id = add_rule(ui)
    for path, data in [(f"/checks/{rule_id}/toggle", {"enabled": "0"}),
                       (f"/checks/{rule_id}/delete", {})]:
        resp = ui.client.post(path, data=data, follow_redirects=False)
        assert resp.status_code == 403
        bad = ui.client.post(path, data={**data, "csrf": "wrong"}, follow_redirects=False)
        assert bad.status_code == 403
    rules = stored(ui)
    assert len(rules) == 1 and rules[0].enabled is True


def test_a_failing_video_title_is_escaped(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path, seeded=False)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title=EVIL, description="No link here",
                   tags=[], privacy_status="public")])
    add_rule(ui)
    html = page(ui)
    assert "<script>alert(1)" not in html
    assert str(escape(EVIL)) in html


def test_a_stored_check_name_is_escaped(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    add_rule(ui, name=EVIL)
    html = page(ui)
    assert "<script>alert(1)" not in html
    assert str(escape(EVIL)) in html


# --- the shell and the default maker -------------------------------------------------------------
def test_checks_is_in_the_nav_right_after_strategy(tmp_path: Path) -> None:
    ui = make_checks_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/checks") == items.index("/seo") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/checks"]


def test_the_default_maker_without_a_key_flashes_the_no_key_sentence(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, packaging_key=lambda cfg: True)  # no rule_maker: the real default
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    resp = nf(ui, "/checks/ask", data={"request": REQUEST})
    assert resp.status_code == 303 and resp.headers["location"] == "/checks"
    assert NO_PACKAGING_KEY in page(ui)
    assert stored(ui) == []


