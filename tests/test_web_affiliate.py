"""The Affiliate links page: your own rules, a preview that writes nothing, drafts only.

Fakes only, no network. Rules live in ``affiliate.json`` in the config home; the page
never invents a program, and Make drafts goes through the normal Changes review.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from webhelpers import make_ui, make_video, seed_catalog
from yezatas import edits
from yezatas.affiliate import AffiliateRule, load_rules, save_rules
from yezatas.store import Store

RULE = AffiliateRule("Shop", ("example.com",), "tag", "partner123")
A, B, C, D, E, F, G = (f"vid{n:08d}" for n in range(1, 8))


def affiliate_path(ui: Any) -> Path:
    return ui.config.home / "affiliate.json"


def rules_on_disk(ui: Any) -> list[AffiliateRule]:
    return load_rules(affiliate_path(ui))


def drafts(ui: Any) -> dict[str, Any]:
    with Store(ui.config.catalog_db) as store:
        return {d.video_id: d for d in store.list_drafts()}


def catalog(ui: Any) -> None:
    seed_catalog(ui.config, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description="Shop https://shop.example.com/item", view_count=900),
        make_video(2, video_id=B, privacy_status="public", title="Cactus two",
                   description="Links https://shop.example.com/a and https://store.example.com/b",
                   view_count=500),
        make_video(3, video_id=C, privacy_status="public", title="Cactus three",
                   description="Elsewhere https://other.org/x", view_count=300),
        make_video(4, video_id=D, privacy_status="private", title="Cactus four",
                   description="Private https://shop.example.com/secret", view_count=200),
        make_video(5, video_id=E, privacy_status="public", title="Cactus five",
                   description="Draft https://shop.example.com/draft", view_count=100),
    ])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, E, "Cactus five (edited)", "Draft https://shop.example.com/draft",
                         ["cactus"])


@pytest.fixture
def ui(tmp_path: Path) -> Any:
    ui = make_ui(tmp_path)
    catalog(ui)
    save_rules(affiliate_path(ui), [RULE])
    return ui


def bare_ui(tmp_path: Path, videos: list[Any], rules: list[AffiliateRule] | None = None) -> Any:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, videos)
    save_rules(affiliate_path(ui), [RULE] if rules is None else rules)
    return ui


def preview_url(visibility: str = "public") -> str:
    return f"/affiliate?visibility={visibility}&preview=1"


def row_for(html: str, video_id: str) -> str:
    match = re.search(rf'<tr id="aff-{video_id}"[^>]*>.*?</tr>', html, re.S)
    assert match, f"no preview row for {video_id}"
    return match.group(0)


# --- the page: about, rules table, add form ------------------------------------------------------

def test_page_shows_rules_table_add_form_and_the_plain_promises(ui: Any) -> None:
    resp = ui.client.get("/affiliate")
    assert resp.status_code == 200
    html = resp.text
    assert "<title>Affiliate - Yezatas</title>" in html
    assert '<h1 class="page-title">Affiliate</h1>' in html
    assert 'id="affiliate-about"' in html
    assert "No program is built in: these rules are your own." in html
    assert ("Only drafts are made. Nothing reaches YouTube until you approve and push "
            "on the Changes page.") in html

    assert 'id="affiliate-rules"' in html and "Your rules" in html
    for header in ("Rule", "Domains", "Parameter", "Value"):
        assert f'<th scope="col">{header}</th>' in html
    assert 'id="rule-0"' in html
    row = re.search(r'<tr id="rule-0">.*?</tr>', html, re.S)
    assert row
    for cell in ("Shop", "example.com", "tag", "partner123"):
        assert cell in row.group(0)
    assert 'action="/affiliate/rules/delete"' in row.group(0)
    assert 'name="index" value="0"' in row.group(0)
    assert ">Delete</button>" in row.group(0)
    assert 'id="affiliate-no-rules"' not in html

    assert 'id="affiliate-add"' in html and "Add a rule" in html
    assert 'action="/affiliate/rules/add"' in html
    assert 'id="aff-name" name="name"' in html
    assert 'id="aff-domains" name="domains"' in html
    assert "Separate domains with commas, like example.com, shop.example.com." in html
    assert 'id="aff-param" name="param"' in html and "For example: tag" in html
    assert 'id="aff-value" name="value"' in html
    assert ">Add rule</button>" in html

    assert 'id="affiliate-preview-card"' in html
    assert 'name="visibility" value="public" checked' in html
    assert 'name="preview" value="1"' in html and ">Preview</button>" in html
    assert "Preview writes nothing." in html
    assert 'id="affiliate-preview"' not in html and 'id="affiliate-apply"' not in html


def test_no_rules_yet_says_so(tmp_path: Path) -> None:
    ui = bare_ui(tmp_path, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description="Shop https://shop.example.com/item", view_count=900),
    ], rules=[])
    html = ui.client.get("/affiliate").text
    assert 'id="affiliate-no-rules"' in html
    assert "No rules yet. Add one below." in html
    assert 'id="rule-0"' not in html
    assert 'id="affiliate-add"' in html  # the way out of the empty state


# --- adding and deleting rules -------------------------------------------------------------------

def test_add_rule_posts_saves_and_shows_the_new_rule(ui: Any) -> None:
    resp = ui.post("/affiliate/rules/add", data={
        "name": "Other shop", "domains": "example.org, shop.example.org",
        "param": "ref", "value": "partner123",
    }, follow_redirects=False)
    assert resp.status_code == 303
    assert urlparse(resp.headers["location"]).path == "/affiliate"
    assert rules_on_disk(ui) == [
        RULE, AffiliateRule("Other shop", ("example.org", "shop.example.org"), "ref", "partner123")]
    html = ui.client.get("/affiliate").text
    assert 'id="rule-1"' in html and "Other shop" in html
    assert "example.org, shop.example.org" in html


def test_add_rule_with_empty_domains_flashes_and_the_file_is_unchanged(ui: Any) -> None:
    before_text = affiliate_path(ui).read_text(encoding="utf-8")
    resp = ui.post("/affiliate/rules/add", data={
        "name": "Nowhere", "domains": "", "param": "tag", "value": "partner123",
    }, follow_redirects=False)
    assert resp.status_code == 303
    page = ui.client.get("/affiliate").text
    assert "at least one domain" in page
    assert affiliate_path(ui).read_text(encoding="utf-8") == before_text
    assert rules_on_disk(ui) == [RULE]


def test_delete_rule_by_index_removes_only_that_rule(ui: Any) -> None:
    save_rules(affiliate_path(ui), [
        AffiliateRule("Alpha", ("example.com",), "tag", "partner123"),
        AffiliateRule("Beta", ("example.org",), "ref", "partner123"),
    ])
    resp = ui.post("/affiliate/rules/delete", data={"index": "0"}, follow_redirects=False)
    assert resp.status_code == 303
    assert urlparse(resp.headers["location"]).path == "/affiliate"
    assert rules_on_disk(ui) == [AffiliateRule("Beta", ("example.org",), "ref", "partner123")]
    html = ui.client.get("/affiliate").text
    assert "Beta" in html and "Alpha" not in html
    assert 'id="rule-0"' in html and 'id="rule-1"' not in html


def test_delete_a_missing_rule_flashes_and_keeps_the_rules(ui: Any) -> None:
    resp = ui.post("/affiliate/rules/delete", data={"index": "9"}, follow_redirects=False)
    assert resp.status_code == 303
    page = ui.client.get("/affiliate").text
    assert "This rule does not exist." in page
    assert rules_on_disk(ui) == [RULE]


def test_broken_rules_file_shows_the_error_and_no_add_form(ui: Any) -> None:
    affiliate_path(ui).write_text("not json", encoding="utf-8")
    resp = ui.client.get("/affiliate")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="affiliate-error"' in html and "Affiliate rules" in html
    assert "affiliate-add" not in html  # a broken file is never overwritten from this page
    assert 'id="affiliate-no-rules"' not in html
    assert 'id="affiliate-preview-card"' in html
    assert affiliate_path(ui).read_text(encoding="utf-8") == "not json"


# --- preview --------------------------------------------------------------------------------------

def test_preview_lists_affected_videos_tagged_counts_and_writes_nothing(ui: Any) -> None:
    before = drafts(ui)
    resp = ui.client.get(preview_url())
    assert resp.status_code == 200
    html = resp.text
    assert 'id="affiliate-preview"' in html and 'id="affiliate-summary"' in html
    assert "would change" in html
    assert f'href="/videos/{A}"' in html and "Cactus one" in html
    assert f'href="/videos/{B}"' in html and "Cactus two" in html
    assert '<td class="col-num">1</td>' in row_for(html, A)
    assert '<td class="col-num">2</td>' in row_for(html, B)
    assert "Links tagged" in html
    # ticked by default; the private and the non-matching video are not in a public preview
    assert re.search(rf'name="video_id" value="{A}"[^>]* checked', html)
    assert re.search(rf'name="video_id" value="{B}"[^>]* checked', html)
    assert f'value="{C}"' not in html and f'value="{D}"' not in html
    # the apply form carries the csrf token and the previewed visibility
    form = html.split('id="affiliate-apply"')[1]
    assert 'name="csrf" value="' in form
    assert '<input type="hidden" name="visibility" value="public">' in form
    assert '<a href="/changes">Review drafts on Changes</a>' in html
    assert "missing now" not in row_for(html, A)
    assert "missing now" not in row_for(html, B)
    # a preview is a pure read
    assert drafts(ui).keys() == before.keys() == {E}
    with Store(ui.config.catalog_db) as store:
        assert store.get_video(A).description == "Shop https://shop.example.com/item"
        assert "tag=partner123" not in store.get_video(B).description


def test_preview_summary_for_a_single_video(tmp_path: Path) -> None:
    ui = bare_ui(tmp_path, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description="Shop https://shop.example.com/item", view_count=900),
    ])
    html = ui.client.get(preview_url()).text
    assert "1 video would change" in html


def test_preview_visibility_all_includes_private_videos(ui: Any) -> None:
    html = ui.client.get(preview_url("all")).text
    assert f'value="{D}"' in html and "Cactus four" in html
    assert 'name="visibility" value="all" checked' in html
    assert '<input type="hidden" name="visibility" value="all">' in html
    public = ui.client.get(preview_url()).text
    assert f'value="{D}"' not in public
    assert 'name="visibility" value="public" checked' in public


def test_blocked_row_is_disabled_with_the_reason(ui: Any) -> None:
    html = ui.client.get(preview_url()).text
    row = row_for(html, E)
    assert "Blocked:" in row and "already has a draft" in row
    tick = re.search(rf'<input[^>]*name="video_id" value="{E}"[^>]*>', row)
    assert tick and "disabled" in tick.group(0) and "checked" not in tick.group(0)
    # other rows can still be drafted, so Make drafts stays enabled
    assert not re.search(r'<button type="submit" class="button primary" disabled>Make drafts', html)


def test_make_drafts_is_disabled_when_every_row_is_blocked(tmp_path: Path) -> None:
    ui = bare_ui(tmp_path, [
        make_video(1, video_id=A, privacy_status="public", title="Cactus one",
                   description="Shop https://shop.example.com/item", view_count=900),
    ])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, A, "Cactus one (edited)", "Shop https://shop.example.com/item",
                         ["cactus"])
    html = ui.client.get(preview_url()).text
    assert "Blocked:" in html
    assert re.search(r'<button type="submit" class="button primary" disabled>Make drafts', html)


def test_preview_shows_each_issue_of_a_row(ui: Any) -> None:
    seed_catalog(ui.config, [
        make_video(7, video_id=G, privacy_status="public", title="Cactus seven",
                   description="Bad < link https://shop.example.com/a", view_count=10),
    ])
    html = ui.client.get(preview_url()).text
    row = row_for(html, G)
    assert "Blocked:" in row and 'class="aff-issue"' in row


# --- make drafts ----------------------------------------------------------------------------------

def test_apply_makes_affiliate_drafts_only_for_ticked_videos(ui: Any) -> None:
    resp = ui.post("/affiliate/apply", data={"visibility": "public", "video_id": [A]},
                   follow_redirects=False)
    assert resp.status_code == 303
    assert urlparse(resp.headers["location"]).path == "/changes"
    made = drafts(ui)
    assert set(made) == {A, E}
    assert made[A].source == "affiliate"
    assert made[A].description == "Shop https://shop.example.com/item?tag=partner123"
    assert made[E].source != "affiliate"
    page = ui.client.get("/changes").text
    assert "Drafts made for 1 video." in page


def test_apply_with_visibility_all_drafts_a_private_video(ui: Any) -> None:
    resp = ui.post("/affiliate/apply", data={"visibility": "all", "video_id": [D]},
                   follow_redirects=False)
    assert resp.status_code == 303
    assert urlparse(resp.headers["location"]).path == "/changes"
    made = drafts(ui)
    assert made[D].source == "affiliate"
    assert made[D].description == "Private https://shop.example.com/secret?tag=partner123"
    assert "Drafts made for 1 video." in ui.client.get("/changes").text


def test_apply_reports_blocked_and_no_longer_matching(ui: Any) -> None:
    # A drafts, E is blocked by its existing draft, C has no matching link any more
    resp = ui.post("/affiliate/apply", data={"visibility": "public", "video_id": [A, E, C]},
                   follow_redirects=False)
    assert resp.status_code == 303
    assert urlparse(resp.headers["location"]).path == "/changes"
    page = ui.client.get("/changes").text
    assert "Drafts made for 1 video." in page
    assert "1 video blocked." in page
    assert "1 video no longer matched." in page
    assert set(drafts(ui)) == {A, E}


# --- safety: csrf and escaping ---------------------------------------------------------------------

def test_add_delete_and_apply_require_the_csrf_token(ui: Any) -> None:
    assert ui.client.post("/affiliate/rules/add", data={
        "name": "Sneaky", "domains": "example.org", "param": "tag", "value": "x",
    }).status_code == 403
    assert ui.client.post("/affiliate/rules/add", data={
        "name": "Sneaky", "domains": "example.org", "param": "tag", "value": "x",
        "csrf": "wrong",
    }).status_code == 403
    assert ui.client.post("/affiliate/rules/delete", data={"index": "0"}).status_code == 403
    assert ui.client.post("/affiliate/apply", data={
        "visibility": "public", "video_id": [A],
    }).status_code == 403
    assert rules_on_disk(ui) == [RULE]
    assert set(drafts(ui)) == {E}


def test_rule_names_and_video_titles_are_escaped(ui: Any) -> None:
    save_rules(affiliate_path(ui), [
        AffiliateRule("<script>alert(1)</script>", ("example.com",), "tag", "partner123")])
    seed_catalog(ui.config, [
        make_video(6, video_id=F, privacy_status="public", title="<img src=x onerror=1>",
                   description="Evil https://shop.example.com/evil", view_count=10),
    ])
    html = ui.client.get("/affiliate").text
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    preview = ui.client.get(preview_url()).text
    assert "<img src=x" not in preview and "&lt;img src=x" in preview
    assert "<script>alert(1)</script>" not in preview


# --- the shell --------------------------------------------------------------------------------------

def test_nav_has_an_affiliate_item_after_links_and_active_here(ui: Any) -> None:
    html = ui.client.get("/affiliate").text
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/affiliate") == items.index("/links") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/affiliate"]
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and "style=" not in html
