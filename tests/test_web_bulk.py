"""The Bulk edit page: preview is a pure read, Make drafts makes drafts only. Fakes, no network."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from webhelpers import make_ui, make_video, seed_catalog
from yezatas import edits
from yezatas.store import Store

OLD = "https://example.invalid/old"
NEW = "https://example.invalid/new"
A, B, C, D, E, F = (f"vid{n:06d}" for n in range(1, 7))


def catalog(ui: Any) -> None:
    seed_catalog(ui.config, [
        make_video(1, privacy_status="public", title="Cactus one", description=f"Intro line\nShop: {OLD}\nBye", view_count=900),
        make_video(2, privacy_status="public", title="Cactus two", description=f"Shop: {OLD} and again {OLD}", view_count=500),
        make_video(3, privacy_status="public", title="Cactus three", description="Nothing to see here", view_count=300),
        make_video(4, title="Cactus four", description=f"Private: {OLD}", privacy_status="private",
                   view_count=200),
        make_video(5, privacy_status="public", title="Cactus five", description=f"Draft already: {OLD}", view_count=100),
        make_video(6, privacy_status="public", title="Cactus six", description="a" * 4996 + " x", view_count=50),
    ])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, E, "Cactus five (edited)", f"Draft already: {OLD}", ["cactus"])


def drafts(ui: Any) -> dict[str, Any]:
    with Store(ui.config.catalog_db) as store:
        return {d.video_id: d for d in store.list_drafts()}


@pytest.fixture
def ui(tmp_path: Path) -> Any:
    ui = make_ui(tmp_path)
    catalog(ui)
    return ui


def replace_url(find: str = OLD, repl: str = NEW, **extra: str) -> str:
    from urllib.parse import urlencode
    params = {"kind": "replace", "find": find, "replace_with": repl, "field": "description",
              "visibility": "public", "preview": "1", **extra}
    return "/bulk?" + urlencode(params)


def test_empty_form_renders_with_the_plain_promises(ui: Any) -> None:
    resp = ui.client.get("/bulk")
    assert resp.status_code == 200
    html = resp.text
    assert "Bulk edit" in html and 'name="find"' in html and 'name="footer"' in html
    assert 'name="kind" value="replace" checked' in html
    assert 'name="visibility" value="public" checked' in html
    assert 'name="field" value="description" checked' in html
    assert "No model is used" in html and "only drafts are made" in html
    assert "Nothing reaches YouTube until you approve and push on the Changes page" in html
    assert "After you push these drafts, History can undo that whole push in one step." in html
    assert "bulk-preview" not in html and "bulk-apply" not in html
    assert 'name="preview"' in html


def test_replace_preview_rows_counts_and_excerpt(ui: Any) -> None:
    html = ui.client.get(replace_url()).text
    assert "2 videos would change" in html
    # public only: B and A (most viewed first); C no match, D private
    assert f'value="{A}" aria-label' in html and f'value="{B}" aria-label' in html
    assert f'value="{D}"' not in html and f'value="{C}"' not in html
    assert html.index(f'id="b-{A}"') < html.index(f'id="b-{B}"')
    assert re.search(rf'id="b-{B}".*?<td class="col-num">2</td>', html, re.S)
    assert re.search(rf'id="b-{A}".*?<td class="col-num">1</td>', html, re.S)
    assert f'href="/videos/{A}"' in html
    # the first changed line, before -> after
    assert f'<del class="seg removed">Shop: {OLD}</del>' in html
    assert f'<ins class="seg added">Shop: {NEW}</ins>' in html
    assert "Intro line" not in html.split('id="bulk-preview"')[1]
    # both ticked
    assert re.search(rf'name="video_id" value="{A}"[^>]* checked', html)
    assert re.search(rf'name="video_id" value="{B}"[^>]* checked', html)


def test_blocked_rows_are_disabled_with_the_reason(ui: Any) -> None:
    html = ui.client.get(replace_url(find="Draft already", repl="Changed")).text
    assert "0 videos would change, 1 blocked" in html
    row = re.search(rf'<tr id="b-{E}"[^>]*>.*?</tr>', html, re.S)
    assert row
    assert "disabled" in row.group(0) and "checked" not in row.group(0)
    assert "already has a draft" in row.group(0)
    assert re.search(r'<button type="submit" class="button primary" disabled>Make drafts', html)


def test_warning_and_error_blocking_show_in_the_row(ui: Any) -> None:
    html = ui.client.get(replace_url(find=OLD, repl="", **{"visibility": "all"})).text
    assert "was in the original description but is missing now" in html
    assert OLD in html
    html = ui.client.get(replace_url(find=" x", repl=" xyz" * 3)).text
    row = re.search(rf'<tr id="b-{F}"[^>]*>.*?</tr>', html, re.S)
    assert row and "Blocked:" in row.group(0) and "5,000 bytes" in row.group(0)
    assert "disabled" in row.group(0)
    assert "0 videos would change, 1 blocked" in html


def test_footer_preview(ui: Any) -> None:
    from urllib.parse import urlencode
    url = "/bulk?" + urlencode({"kind": "footer", "footer": "Links:\nhttps://example.invalid/shop",
                                "marker": "", "visibility": "public", "preview": "1"})
    html = ui.client.get(url).text
    assert "3 videos would change, 2 blocked" in html  # E has a draft, F would pass 5,000 bytes
    assert 'name="kind" value="footer" checked' in html
    assert "Links:" in html and "https://example.invalid/shop" in html
    assert '<ins class="seg added">Links:</ins>' in html


def test_title_excerpt(ui: Any) -> None:
    html = ui.client.get(replace_url(find="Cactus", repl="Succulent", field="title")).text
    assert "Title:" in html
    assert '<del class="seg removed">Cactus one</del>' in html
    assert '<ins class="seg added">Succulent one</ins>' in html


def test_value_error_is_shown_and_the_values_are_kept(ui: Any) -> None:
    html = ui.client.get(replace_url(find="same", repl="same")).text
    assert 'id="bulk-error"' in html and "are the same" in html
    assert 'value="same"' in html  # kept
    assert "bulk-preview" not in html
    html = ui.client.get("/bulk?kind=replace&find=&preview=1&field=description").text
    assert "Enter the text to find." in html
    html = ui.client.get("/bulk?kind=replace&find=x&replace_with=y&preview=1").text  # no field ticked
    assert "Choose the title, the description or both" in html
    html = ui.client.get(replace_url(**{"from": "2026-05-01", "to": "2026-01-01"})).text
    assert "is after" in html
    html = ui.client.get("/bulk?kind=footer&footer=&marker=Hi&preview=1").text
    assert "Enter the footer text." in html and 'value="Hi"' in html


def test_dates_and_visibility_filters(ui: Any) -> None:
    html = ui.client.get(replace_url(visibility="all")).text
    assert f'value="{D}" aria-label' in html
    assert 'name="visibility" value="all" checked' in html
    html = ui.client.get(replace_url(**{"from": "2099-01-01"})).text
    assert "No video matches" in html and "bulk-apply" not in html


def test_too_many_rows_says_so(ui: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from yezatas import bulk
    from yezatas.web import views
    monkeypatch.setattr(views, "MAX_BULK", 1)
    html = ui.client.get(replace_url()).text
    assert 'id="bulk-too-many"' in html and "at most 1 can be changed" in html
    assert bulk.MAX_BULK == 500


def test_preview_writes_nothing(ui: Any) -> None:
    before = drafts(ui)
    for url in (replace_url(), replace_url(find="same", repl="same"), "/bulk"):
        assert ui.client.get(url).status_code == 200
    assert drafts(ui).keys() == before.keys() == {E}
    with Store(ui.config.catalog_db) as store:
        assert OLD in store.get_video(A).description


def apply_data(**extra: Any) -> dict[str, Any]:
    data = {"kind": "replace", "find": OLD, "replace_with": NEW, "field": "description",
            "footer": "", "marker": "", "visibility": "public", "from": "", "to": "",
            "video_id": [A, B]}
    data.update(extra)
    return data


def test_apply_makes_bulk_drafts_and_redirects_to_changes(ui: Any) -> None:
    resp = ui.post("/bulk/apply", data=apply_data(), follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/changes"
    made = drafts(ui)
    assert set(made) == {A, B, E}
    assert made[A].source == "bulk" and made[B].source == "bulk"
    assert made[E].source != "bulk"
    assert made[A].description == f"Intro line\nShop: {NEW}\nBye"
    assert made[B].description == f"Shop: {NEW} and again {NEW}"
    assert made[A].note.startswith("Bulk: replace")
    page = ui.client.get("/changes").text
    assert "Drafts made for 2 videos." in page
    assert "Review them on the Changes page; nothing changes on YouTube until you approve and push." in page


def test_apply_only_drafts_ticked_ids(ui: Any) -> None:
    resp = ui.post("/bulk/apply", data=apply_data(video_id=[B]), follow_redirects=False)
    assert resp.headers["location"] == "/changes"
    assert set(drafts(ui)) == {B, E}
    assert "Drafts made for 1 video." in ui.client.get("/changes").text


def test_apply_reports_blocked_and_no_longer_matching(ui: Any) -> None:
    # E is blocked (has a draft); C does not contain the text any more
    ui.post("/bulk/apply", data=apply_data(video_id=[A, C, E]), follow_redirects=False)
    text = ui.client.get("/changes").text
    assert "Drafts made for 1 video. 1 video blocked. 1 video no longer matched." in text
    assert set(drafts(ui)) == {A, E}


def test_nothing_ticked_goes_back_to_the_preview_with_an_error(ui: Any) -> None:
    resp = ui.post("/bulk/apply", data=apply_data(video_id=[]), follow_redirects=False)
    assert resp.status_code == 303
    target = urlparse(resp.headers["location"])
    assert target.path == "/bulk"
    query = parse_qs(target.query)
    assert query["find"] == [OLD] and query["preview"] == ["1"] and query["field"] == ["description"]
    page = ui.client.get(resp.headers["location"]).text
    assert "Tick at least one video" in page and "2 videos would change" in page
    assert set(drafts(ui)) == {E}


def test_apply_value_error_goes_back_with_the_message(ui: Any) -> None:
    resp = ui.post("/bulk/apply", data=apply_data(find="same", replace_with="same"),
                   follow_redirects=False)
    assert urlparse(resp.headers["location"]).path == "/bulk"
    assert "are the same" in ui.client.get(resp.headers["location"]).text
    assert set(drafts(ui)) == {E}


def test_apply_over_the_limit_is_refused_and_nothing_is_drafted(ui: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from yezatas import bulk
    monkeypatch.setattr(bulk, "MAX_BULK", 1)
    resp = ui.post("/bulk/apply", data=apply_data(), follow_redirects=False)
    assert urlparse(resp.headers["location"]).path == "/bulk"
    assert "at most 1 can be changed" in ui.client.get(resp.headers["location"]).text
    assert set(drafts(ui)) == {E}


def test_apply_requires_the_csrf_token(ui: Any) -> None:
    resp = ui.client.post("/bulk/apply", data={**apply_data(), "csrf": "wrong"})
    assert resp.status_code == 403
    resp = ui.client.post("/bulk/apply", data=apply_data())
    assert resp.status_code == 403
    assert set(drafts(ui)) == {E}


def test_preview_form_carries_a_csrf_and_the_values_as_hidden_fields(ui: Any) -> None:
    html = ui.client.get(replace_url()).text
    form = html.split('id="bulk-apply"')[1]
    assert 'name="csrf" value="' in form
    assert f'<input type="hidden" name="find" value="{OLD}">' in form
    assert '<input type="hidden" name="field" value="description">' in form


def test_creator_text_is_escaped(ui: Any) -> None:
    attack = '"><script>alert(1)</script>'
    html = ui.client.get(replace_url(find=attack, repl="<b>x</b>")).text
    assert "<script>alert(1)</script>" not in html
    assert "<b>x</b>" not in html
    assert "&lt;script&gt;" in html or "&#34;&gt;&lt;script&gt;" in html
    html = ui.client.get("/bulk?kind=footer&preview=1&footer=%3Cscript%3Ealert(2)%3C/script%3E").text
    assert "<script>alert(2)" not in html
    # a title with markup is escaped in the table and in the checkbox label
    seed_catalog(ui.config, [make_video(7, privacy_status="public", title='<img src=x onerror=1>', description="Hello old text",
                                        view_count=10)])
    html = ui.client.get(replace_url(find="old text", repl="new text")).text
    assert "<img src=x" not in html and "&lt;img src=x" in html


def test_links_on_changes_and_content_pages(ui: Any) -> None:
    changes = ui.client.get("/changes").text
    assert 'href="/bulk"' in changes and "Bulk edit" in changes
    content = ui.client.get("/videos").text
    assert 'href="/bulk"' in content and "Bulk edit" in content
    nav = re.search(r"<nav.*?</nav>", ui.client.get("/bulk").text, re.S)
    assert nav and "/bulk" not in nav.group(0)


def test_post_to_bulk_apply_without_a_catalog_still_answers(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.post("/bulk/apply", data=apply_data(), follow_redirects=False)
    assert resp.status_code == 303
    assert ui.client.get("/bulk?kind=replace&find=a&replace_with=b&field=title&preview=1").status_code == 200
