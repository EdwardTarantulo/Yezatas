"""Affiliate links: rules load plainly, tagging is exact, plans write nothing, apply drafts only."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from yezatas.affiliate import (
    AffiliateError,
    AffiliateRule,
    apply_affiliate,
    load_rules,
    plan_affiliate,
    save_rules,
    tag_url,
)
from yezatas.bulk import BLOCKED_HAS_DRAFT, MAX_BULK
from yezatas.edits import save_draft
from yezatas.store import Store, Video

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)

RULE = AffiliateRule("Shop", ("example.com",), "tag", "partner123")
REF_RULE = AffiliateRule("Shop ref", ("example.com",), "ref", "partner123")


def video(n: int, description: str = "", title: str | None = None, privacy: str = "public",
          published: str = "2026-01-10T10:00:00Z", views: int | None = None) -> Video:
    return Video(
        video_id=f"vid{n:06d}", channel_id="UC_fake", title=title or f"Cactus tip {n}",
        description=description, tags=["cactus", "garden"], privacy_status=privacy,
        published_at=published, view_count=views if views is not None else n,
    )


def vid(n: int) -> str:
    return f"vid{n:06d}"


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        yield s


def fill(store: Store, *videos: Video) -> None:
    pull = store.begin_pull("UC_fake", NOW)
    for v in videos:
        store.upsert_video(v, pull, NOW)
    store.finish_pull(pull, len(videos), NOW)


def snapshot(store: Store) -> tuple:
    return tuple(store.list_drafts()), tuple(v.to_dict().__repr__() for v in store.list_videos())


def test_tag_url_matches_subdomain_and_www() -> None:
    assert tag_url("https://shop.example.com/a", [RULE]) == "https://shop.example.com/a?tag=partner123"
    assert tag_url("https://www.example.com/a", [RULE]) == "https://www.example.com/a?tag=partner123"
    assert tag_url("www.example.com/a", [RULE]) == "www.example.com/a?tag=partner123"
    assert tag_url("https://example.com/a", [RULE]) == "https://example.com/a?tag=partner123"
    assert tag_url("https://deep.shop.example.com/a", [RULE]) == (
        "https://deep.shop.example.com/a?tag=partner123")


def test_tag_url_rejects_lookalikes_and_other_domains() -> None:
    assert tag_url("https://notexample.com/a", [RULE]) == "https://notexample.com/a"
    assert tag_url("https://example.com.evil.com/a", [RULE]) == "https://example.com.evil.com/a"
    assert tag_url("https://other.org/a", [RULE]) == "https://other.org/a"
    assert tag_url("https://example.org/a?tag=old", [RULE]) == "https://example.org/a?tag=old"


def test_tag_url_replaces_in_place_and_drops_later_duplicates() -> None:
    assert tag_url("https://example.com/a?x=1&tag=old&y=2", [RULE]) == (
        "https://example.com/a?x=1&tag=partner123&y=2")
    assert tag_url("https://example.com/a?tag=old&x=1&tag=older", [RULE]) == (
        "https://example.com/a?tag=partner123&x=1")


def test_tag_url_appends_and_keeps_order_blanks_and_fragment() -> None:
    assert tag_url("https://example.com/a?x=1", [RULE]) == "https://example.com/a?x=1&tag=partner123"
    assert tag_url("https://example.com/a?b=2&empty=&a=1#frag", [RULE]) == (
        "https://example.com/a?b=2&empty=&a=1&tag=partner123#frag")
    assert tag_url("www.example.com/a?x=1#frag", [RULE]) == "www.example.com/a?x=1&tag=partner123#frag"


def test_tag_url_already_tagged_is_byte_identical() -> None:
    url = "https://example.com/a?tag=partner123&x=1#frag"
    assert tag_url(url, [RULE]) == url
    bare = "www.example.com/a?x=1&tag=partner123"
    assert tag_url(bare, [RULE]) == bare


def test_tag_url_leaves_non_http_links_alone() -> None:
    assert tag_url("ftp://example.com/a", [RULE]) == "ftp://example.com/a"
    assert tag_url("mailto:someone@example.com", [RULE]) == "mailto:someone@example.com"
    assert tag_url("example.com/a", [RULE]) == "example.com/a"
    assert tag_url("https://example.com/a", []) == "https://example.com/a"


def test_tag_url_applies_every_matching_rule_in_order() -> None:
    assert tag_url("https://example.com/a", [RULE, REF_RULE]) == (
        "https://example.com/a?tag=partner123&ref=partner123")
    assert tag_url("https://example.com/a?ref=old", [RULE, REF_RULE]) == (
        "https://example.com/a?ref=partner123&tag=partner123")


def test_load_rules_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_rules(tmp_path / "affiliate.json") == []


def test_load_and_save_rules_round_trip(tmp_path: Path) -> None:
    rules = [
        AffiliateRule("Shop", ("example.com", "shop.example.com"), "tag", "partner123"),
        AffiliateRule("Other shop", ("example.org",), "ref", "partner123"),
    ]
    path = tmp_path / "deep" / "affiliate.json"
    save_rules(path, rules)
    assert load_rules(path) == rules
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == [
        {"name": "Shop", "domains": ["example.com", "shop.example.com"],
         "param": "tag", "value": "partner123"},
        {"name": "Other shop", "domains": ["example.org"], "param": "ref", "value": "partner123"},
    ]


def test_load_rules_normalizes_domains_and_ignores_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "affiliate.json"
    path.write_text(json.dumps([
        {"name": " Shop ", "domains": [" WWW.Example.COM. "], "param": " tag ",
         "value": "partner123", "extra": "ignored"},
    ]), encoding="utf-8")
    assert load_rules(path) == [AffiliateRule("Shop", ("example.com",), "tag", "partner123")]


def test_load_rules_bad_json_names_the_problem(tmp_path: Path) -> None:
    path = tmp_path / "affiliate.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AffiliateError, match="not valid JSON"):
        load_rules(path)
    path.write_text('{"name": "Shop"}', encoding="utf-8")
    with pytest.raises(AffiliateError, match="must be a list"):
        load_rules(path)


@pytest.mark.parametrize("data, text", [
    ([42], "Affiliate rule 1"),
    ([{"domains": ["example.com"], "param": "tag", "value": "partner123"}], "Affiliate rule 1"),
    ([{"name": "  ", "domains": ["example.com"], "param": "tag", "value": "partner123"}],
     "Affiliate rule 1"),
    ([{"name": "Shop", "param": "tag", "value": "partner123"}], "domains"),
    ([{"name": "Shop", "domains": "example.com", "param": "tag", "value": "partner123"}], "domains"),
    ([{"name": "Shop", "domains": [], "param": "tag", "value": "partner123"}], "domains"),
    ([{"name": "Shop", "domains": ["example.com", " "], "param": "tag", "value": "partner123"}],
     "domains"),
    ([{"name": "Shop", "domains": ["example.com"], "param": " ", "value": "partner123"}], "param"),
    ([{"name": "Shop", "domains": ["example.com"], "param": "tag", "value": 3}], "value"),
])
def test_load_rules_bad_rule_names_the_rule_and_problem(tmp_path: Path, data: object, text: str) -> None:
    path = tmp_path / "affiliate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AffiliateError) as caught:
        load_rules(path)
    assert text in str(caught.value)
    if isinstance(data, list) and isinstance(data[0], dict) and data[0].get("name") == "Shop":
        assert "Shop" in str(caught.value)


def test_plan_tags_links_and_keeps_sentence_punctuation_outside(store: Store) -> None:
    fill(store,
         video(1, "See https://example.com/a."),
         video(2, "Shop (https://shop.example.com/a?x=1), then www.example.com/b!"),
         video(3, "Already https://example.com/a?tag=partner123 done"))
    before = snapshot(store)
    plan = {c.video_id: c for c in plan_affiliate(store, [RULE])}
    assert set(plan) == {vid(1), vid(2)}
    assert plan[vid(1)].description == "See https://example.com/a?tag=partner123."
    assert plan[vid(1)].tagged == 1
    assert plan[vid(2)].description == (
        "Shop (https://shop.example.com/a?x=1&tag=partner123), "
        "then www.example.com/b?tag=partner123!")
    assert plan[vid(2)].tagged == 2
    assert snapshot(store) == before  # a plan writes nothing


def test_plan_skips_videos_without_a_changed_link(store: Store) -> None:
    fill(store,
         video(1, "No links here"),
         video(2, "Other https://other.org/a and ftp://example.com/a"),
         video(3, "Shop https://example.com/a"))
    assert [c.video_id for c in plan_affiliate(store, [RULE])] == [vid(3)]
    assert plan_affiliate(store, []) == []


def test_plan_visibility_and_bad_visibility(store: Store) -> None:
    fill(store,
         video(1, "Shop https://example.com/a", privacy="public"),
         video(2, "Shop https://example.com/a", privacy="unlisted"),
         video(3, "Shop https://example.com/a", privacy="private"))
    assert [c.video_id for c in plan_affiliate(store, [RULE])] == [vid(1)]
    assert {c.video_id for c in plan_affiliate(store, [RULE], visibility="all")} == {
        vid(1), vid(2), vid(3)}
    with pytest.raises(ValueError, match="public only"):
        plan_affiliate(store, [RULE], visibility="private")


def test_apply_visibility_all_drafts_private_video_that_default_reports_not_in_plan(
    store: Store,
) -> None:
    fill(store, video(1, "Shop https://shop.example.com/a", privacy="private"))
    assert [c.video_id for c in plan_affiliate(store, [RULE], visibility="all")] == [vid(1)]
    assert plan_affiliate(store, [RULE]) == []
    result = apply_affiliate(store, [RULE], [vid(1)], NOW, visibility="all")
    assert (result.drafted, result.blocked, result.not_in_plan) == (1, 0, 0)
    assert [(i.video_id, i.status, i.message) for i in result.items] == [
        (vid(1), "drafted", "Draft made.")]
    row = store.get_draft(vid(1))
    assert row.description == "Shop https://shop.example.com/a?tag=partner123"
    store.delete_draft(vid(1))
    defaulted = apply_affiliate(store, [RULE], [vid(1)], NOW)
    assert (defaulted.drafted, defaulted.blocked, defaulted.not_in_plan) == (0, 0, 1)
    assert [(i.video_id, i.status) for i in defaulted.items] == [(vid(1), "not_in_plan")]
    assert store.get_draft(vid(1)) is None


def test_plan_blocks_a_video_that_already_has_a_draft(store: Store) -> None:
    fill(store, video(1, "Shop https://shop.example.com/a"), video(2, "Shop https://shop.example.com/b"))
    save_draft(store, vid(1), "Cactus tip 1", "something else", ["cactus"], now=NOW)
    plan = {c.video_id: c for c in plan_affiliate(store, [RULE])}
    assert plan[vid(1)].blocked == BLOCKED_HAS_DRAFT
    assert plan[vid(1)].blocked == "This video already has a draft: push or discard it first."
    assert plan[vid(1)].description == "Shop https://shop.example.com/a?tag=partner123"
    assert plan[vid(2)].blocked == ""
    result = apply_affiliate(store, [RULE], [vid(1), vid(2)], NOW)
    assert (result.drafted, result.blocked, result.not_in_plan) == (1, 1, 0)
    assert {i.video_id: (i.status, i.message) for i in result.items} == {
        vid(1): ("blocked", BLOCKED_HAS_DRAFT), vid(2): ("drafted", "Draft made.")}
    assert store.get_draft(vid(1)).description == "something else"  # untouched


def test_plan_blocks_when_the_new_text_has_an_error(store: Store) -> None:
    fill(store, video(1, "Bad < link https://example.com/a"))
    change = plan_affiliate(store, [RULE])[0]
    assert "< or >" in change.blocked
    assert any(i.severity == "error" for i in change.issues)


def test_apply_makes_drafts_with_source_note_and_catalog_base(store: Store) -> None:
    fill(store,
         video(1, "Shop https://shop.example.com/a"),
         video(2, "Shop https://shop.example.com/a"),
         video(3, "One https://shop.example.com/a two https://shop.example.com/b"))
    result = apply_affiliate(store, [RULE], [vid(1), vid(3)], NOW)
    assert (result.drafted, result.blocked, result.not_in_plan) == (2, 0, 0)
    assert [(i.video_id, i.status, i.message) for i in result.items] == [
        (vid(1), "drafted", "Draft made."), (vid(3), "drafted", "Draft made.")]
    assert store.get_draft(vid(2)) is None  # not approved
    row = store.get_draft(vid(1))
    assert row.source == "affiliate"
    assert row.note == "Affiliate links: 1 link tagged"
    assert row.description == "Shop https://shop.example.com/a?tag=partner123"
    assert row.title == "Cactus tip 1" and row.tags == ("cactus", "garden")
    assert row.base_description == "Shop https://shop.example.com/a"  # the base is the catalog text
    assert row.base_title == "Cactus tip 1" and row.approved_at is None
    assert row.created_at == NOW.isoformat()
    assert store.get_draft(vid(3)).note == "Affiliate links: 2 links tagged"


def test_apply_counts_not_in_plan_and_blocked_and_replans(store: Store) -> None:
    fill(store,
         video(1, "Shop https://example.com/a"),
         video(2, "Shop https://example.com/b"),
         video(3, "No matching link"))
    approved = [c.video_id for c in plan_affiliate(store, [RULE])] + [vid(3), vid(99)]
    fill(store, video(1, "edited elsewhere"))
    save_draft(store, vid(2), "Cactus tip 2", "mine", ["cactus"], now=NOW)
    result = apply_affiliate(store, [RULE], approved, NOW)
    assert {i.video_id: (i.status, i.message) for i in result.items} == {
        vid(1): ("not_in_plan", "This edit no longer applies to this video."),
        vid(2): ("blocked", BLOCKED_HAS_DRAFT),
        vid(3): ("not_in_plan", "This edit no longer applies to this video."),
        vid(99): ("not_in_plan", "This edit no longer applies to this video."),
    }
    assert (result.drafted, result.blocked, result.not_in_plan) == (0, 1, 3)
    assert store.get_draft(vid(1)) is None and store.get_draft(vid(3)) is None
    assert store.get_draft(vid(2)).description == "mine"


def test_apply_dedupes_ids_and_handles_empty(store: Store) -> None:
    fill(store, video(1, "Shop https://shop.example.com/a"))
    empty = apply_affiliate(store, [RULE], [], NOW)
    assert (empty.drafted, empty.items) == (0, ())
    twice = apply_affiliate(store, [RULE], [vid(1), vid(1)], NOW)
    assert twice.drafted == 1 and len(twice.items) == 1
    assert store.get_draft(vid(1)).note == "Affiliate links: 1 link tagged"


def test_apply_refuses_more_than_the_limit(store: Store) -> None:
    fill(store, video(1, "Shop https://example.com/a"))
    with pytest.raises(ValueError, match="at most 500"):
        apply_affiliate(store, [RULE], [f"id{n:09d}" for n in range(MAX_BULK + 1)], NOW)
    assert store.list_drafts() == []


def test_tagging_keeps_other_parameters_as_written() -> None:
    rule = AffiliateRule("Shop", ("example.com",), "tag", "partner123")
    url = "https://example.com/p?q=a%20b&x=1+2"
    assert tag_url(url, [rule]) == "https://example.com/p?q=a%20b&x=1+2&tag=partner123"
