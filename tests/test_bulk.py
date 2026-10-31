"""Bulk edits (find & replace, footer block): plans write nothing, apply makes drafts only."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yezatas.bulk import MAX_BULK, BulkSpec, apply_bulk, plan_bulk
from yezatas.edits import save_draft
from yezatas.store import Store, Video

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


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


OLD = "https://example.invalid/old"
NEW = "https://example.invalid/new"


def test_replace_description_counts_and_leaves_out_videos_without_match(store: Store) -> None:
    fill(store,
         video(1, f"Shop: {OLD}\nAgain {OLD}"),
         video(2, "Nothing to see"),
         video(3, f"One: {OLD}"))
    before = snapshot(store)
    plan = plan_bulk(store, BulkSpec("replace", find=OLD, replace_with=NEW))
    assert {c.video_id: c.count for c in plan} == {vid(1): 2, vid(3): 1}
    first = next(c for c in plan if c.video_id == vid(1))
    assert first.description == f"Shop: {NEW}\nAgain {NEW}"
    assert first.title == "Cactus tip 1" and first.blocked == ""
    assert snapshot(store) == before  # a plan writes nothing


def test_replace_is_case_sensitive_plain_text(store: Store) -> None:
    fill(store, video(1, "Cactus. cactus. c.ctus"))
    assert [c.count for c in plan_bulk(store, BulkSpec("replace", find="cactus", replace_with="x"))] == [1]
    assert plan_bulk(store, BulkSpec("replace", find="c.ctus.*", replace_with="x")) == []  # no regex
    only = plan_bulk(store, BulkSpec("replace", find="c.ctus", replace_with="x"))[0]
    assert only.description == "Cactus. cactus. x"


def test_replace_in_title_and_description(store: Store) -> None:
    fill(store, video(1, "About Cactus tip", title="Cactus tip 1"), video(2, "Other", title="Plain"))
    spec = BulkSpec("replace", find="Cactus", replace_with="Succulent", fields=("title", "description"))
    plan = plan_bulk(store, spec)
    assert len(plan) == 1
    assert (plan[0].title, plan[0].description, plan[0].count) == ("Succulent tip 1", "About Succulent tip", 2)
    only_title = plan_bulk(store, BulkSpec("replace", find="Cactus", replace_with="S", fields=("title",)))
    assert only_title[0].title == "S tip 1" and only_title[0].description == "About Cactus tip"
    default = plan_bulk(store, BulkSpec("replace", find="Cactus", replace_with="S"))
    assert default[0].title == "Cactus tip 1"  # description only by default


def test_replace_can_delete(store: Store) -> None:
    fill(store, video(1, "Keep this. Drop this."))
    plan = plan_bulk(store, BulkSpec("replace", find=" Drop this.", replace_with=""))
    assert plan[0].description == "Keep this."


@pytest.mark.parametrize("spec, text", [
    (BulkSpec("replace", find="", replace_with="x"), "text to find"),
    (BulkSpec("replace", find="same", replace_with="same"), "same"),
    (BulkSpec("replace", find="x" * 501, replace_with="y"), "500"),
    (BulkSpec("replace", find="a", replace_with="b", fields=()), "title"),
    (BulkSpec("replace", find="a", replace_with="b", fields=("tags",)), "title"),
    (BulkSpec("replace", find="a", replace_with="b\nc", fields=("title",)), "line break"),
    (BulkSpec("footer", footer="  \n "), "footer"),
    (BulkSpec("footer", footer="x" * 1501), "1500"),
    (BulkSpec("footer", footer="x", footer_marker="two\nlines"), "one line"),
    (BulkSpec("other"), "Choose"),
    (BulkSpec("replace", find="a", replace_with="b", visibility="private"), "public only"),
    (BulkSpec("replace", find="a", replace_with="b", published_from="2026-13-40"), "published from"),
    (BulkSpec("replace", find="a", replace_with="b", published_from="2026-02-01",
              published_to="2026-01-01"), "after"),
])
def test_bad_specs_are_refused_with_a_sentence(store: Store, spec: BulkSpec, text: str) -> None:
    fill(store, video(1, "a"))
    with pytest.raises(ValueError, match=text):
        plan_bulk(store, spec)
    with pytest.raises(ValueError):
        apply_bulk(store, spec, [vid(1)], NOW)
    assert store.list_drafts() == []


def test_footer_is_appended_after_one_blank_line(store: Store) -> None:
    fill(store, video(1, "Intro text.\n\n\n  "), video(2, ""))
    plan = {c.video_id: c for c in plan_bulk(store, BulkSpec("footer", footer="Follow us!\nSite: x.invalid"))}
    assert plan[vid(1)].description == "Intro text.\n\nFollow us!\nSite: x.invalid"
    assert plan[vid(2)].description == "Follow us!\nSite: x.invalid"
    assert plan[vid(1)].count == 1 and plan[vid(1)].title == "Cactus tip 1"


def test_footer_marker_replaces_from_the_last_marker_line(store: Store) -> None:
    text = "Intro\n--- Links ---\nold a\n--- Links ---\nold b\nold c"
    fill(store, video(1, text), video(2, "Intro only"))
    spec = BulkSpec("footer", footer="--- Links ---\nnew", footer_marker="--- Links ---")
    plan = {c.video_id: c for c in plan_bulk(store, spec)}
    assert plan[vid(1)].description == "Intro\n--- Links ---\nold a\n--- Links ---\nnew"
    assert plan[vid(2)].description == "Intro only\n\n--- Links ---\nnew"  # no marker: appended


def test_footer_marker_matches_a_whole_stripped_line(store: Store) -> None:
    fill(store, video(1, "Intro\nSee --- Links --- here\n   --- Links ---   \nold"))
    spec = BulkSpec("footer", footer="--- Links ---\nnew", footer_marker="--- Links ---")
    assert plan_bulk(store, spec)[0].description == "Intro\nSee --- Links --- here\n--- Links ---\nnew"


def test_footer_already_present_is_left_out(store: Store) -> None:
    fill(store, video(1, "Intro\n\nFollow us!\n\n"), video(2, "Follow us!"), video(3, "Please Follow us!"))
    plan = plan_bulk(store, BulkSpec("footer", footer="Follow us!"))
    assert [c.video_id for c in plan] == [vid(3)]  # only the one that does not end with the block
    marker = BulkSpec("footer", footer="## Links\nnew", footer_marker="## Links")
    fill(store, video(4, "Intro\n## Links\nnew\n"))
    assert vid(4) not in [c.video_id for c in plan_bulk(store, marker)]


def test_visibility_filter(store: Store) -> None:
    fill(store, video(1, "x", privacy="public"), video(2, "x", privacy="unlisted"),
         video(3, "x", privacy="private"))
    spec = BulkSpec("replace", find="x", replace_with="y")
    assert [c.video_id for c in plan_bulk(store, spec)] == [vid(1)]  # public is the default
    everything = plan_bulk(store, BulkSpec("replace", find="x", replace_with="y", visibility="all"))
    assert {c.video_id for c in everything} == {vid(1), vid(2), vid(3)}


def test_date_range_is_inclusive(store: Store) -> None:
    fill(store,
         video(1, "x", published="2026-01-01T00:00:00Z"),
         video(2, "x", published="2026-01-10T23:59:59Z"),
         video(3, "x", published="2026-01-20T10:00:00Z"),
         video(4, "x", published=""))
    def ids(**kw: str) -> set[str]:
        return {c.video_id for c in plan_bulk(store, BulkSpec("replace", find="x", replace_with="y", **kw))}
    assert ids(published_from="2026-01-10", published_to="2026-01-20") == {vid(2), vid(3)}
    assert ids(published_to="2026-01-01") == {vid(1)}
    assert ids(published_from="2026-01-11") == {vid(3)}
    assert ids() == {vid(1), vid(2), vid(3), vid(4)}  # no dates: the video without a date is kept


def test_explicit_ids_limit_the_plan(store: Store) -> None:
    fill(store, video(1, "x"), video(2, "x"), video(3, "x"))
    plan = plan_bulk(store, BulkSpec("replace", find="x", replace_with="y", video_ids=(vid(2), vid(3), "nope")))
    assert {c.video_id for c in plan} == {vid(2), vid(3)}


def test_video_with_a_draft_is_blocked(store: Store) -> None:
    fill(store, video(1, "x one"), video(2, "x two"))
    save_draft(store, vid(1), "Cactus tip 1", "something else x", ["cactus"], now=NOW)
    plan = {c.video_id: c for c in plan_bulk(store, BulkSpec("replace", find="x", replace_with="y"))}
    assert "already has a draft" in plan[vid(1)].blocked
    assert plan[vid(1)].description == "y one"  # worked out from the catalog text, not the draft
    assert plan[vid(2)].blocked == ""
    result = apply_bulk(store, BulkSpec("replace", find="x", replace_with="y"), [vid(1), vid(2)], NOW)
    assert (result.drafted, result.blocked) == (1, 1)
    assert store.get_draft(vid(1)).description == "something else x"  # untouched


def test_error_issues_block_and_warnings_do_not(store: Store) -> None:
    near_limit = "a" * 4995
    fill(store,
         video(1, near_limit + " x"),  # 4997 bytes
         video(2, "See https://example.invalid/old now"),
         video(3, "plain x"))
    # video 1 grows over 5000 bytes, video 2 only loses a link (a warning)
    spec = BulkSpec("replace", find=" x", replace_with=" xyz" * 3)
    plan = {c.video_id: c for c in plan_bulk(store, spec)}
    assert "5,000 bytes" in plan[vid(1)].blocked and plan[vid(1)].issues
    assert plan[vid(3)].blocked == ""
    spec2 = BulkSpec("replace", find="https://example.invalid/old", replace_with="")
    change = plan_bulk(store, spec2)[0]
    assert change.blocked == ""
    assert any(i.code == "link_removed" and i.severity == "warning" for i in change.issues)
    brackets = plan_bulk(store, BulkSpec("replace", find="plain", replace_with="<b>plain</b>"))[0]
    assert "< or >" in brackets.blocked
    apply = apply_bulk(store, spec, [vid(1), vid(3)], NOW)
    assert (apply.drafted, apply.blocked) == (1, 1)
    assert store.get_draft(vid(1)) is None


def test_empty_title_is_blocked(store: Store) -> None:
    fill(store, video(1, "d", title="Gone"))
    change = plan_bulk(store, BulkSpec("replace", find="Gone", replace_with="", fields=("title",)))[0]
    assert "title is empty" in change.blocked


def test_apply_makes_drafts_with_source_note_and_keeps_tags(store: Store) -> None:
    fill(store, video(1, f"Shop {OLD}"), video(2, f"Shop {OLD}"), video(3, f"Shop {OLD}"))
    spec = BulkSpec("replace", find=OLD, replace_with=NEW)
    result = apply_bulk(store, spec, [vid(1), vid(3)], NOW)
    assert (result.drafted, result.blocked, result.not_in_plan) == (2, 0, 0)
    assert [(i.video_id, i.status) for i in result.items] == [(vid(1), "drafted"), (vid(3), "drafted")]
    assert store.get_draft(vid(2)) is None  # not approved
    row = store.get_draft(vid(1))
    assert row.source == "bulk"
    assert row.note == f'Bulk: replace "{OLD}" with "{NEW}"'
    assert row.description == f"Shop {NEW}" and row.title == "Cactus tip 1"
    assert row.tags == ("cactus", "garden") and row.approved_at is None
    assert row.base_description == f"Shop {OLD}"  # the base is the catalog text
    assert row.created_at == NOW.isoformat()


def test_footer_and_delete_notes(store: Store) -> None:
    fill(store, video(1, "Hello x"))
    apply_bulk(store, BulkSpec("footer", footer="Bye"), [vid(1)], NOW)
    assert store.get_draft(vid(1)).note == "Bulk: footer block"
    assert store.get_draft(vid(1)).description == "Hello x\n\nBye"
    store.delete_draft(vid(1))
    apply_bulk(store, BulkSpec("replace", find="x", replace_with="", fields=("title", "description")),
               [vid(1)], NOW)
    assert store.get_draft(vid(1)).note == 'Bulk: remove "x" in title and description'


def test_apply_replans_and_skips_ids_that_changed_or_were_never_in_the_plan(store: Store) -> None:
    fill(store, video(1, "x one"), video(2, "x two"), video(3, "x three"), video(4, "no match"))
    spec = BulkSpec("replace", find="x", replace_with="y")
    approved = [plan.video_id for plan in plan_bulk(store, spec)] + [vid(4), "ghost0000001"]
    # after the preview: video 1 is edited on YouTube (no match any more), video 2 gets a draft
    fill(store, video(1, "edited elsewhere"))
    save_draft(store, vid(2), "Cactus tip 2", "mine", ["cactus"], now=NOW)
    result = apply_bulk(store, spec, approved, NOW)
    status = {i.video_id: i.status for i in result.items}
    assert status == {vid(3): "drafted", vid(2): "blocked", vid(1): "not_in_plan",
                      vid(4): "not_in_plan", "ghost0000001": "not_in_plan"}
    assert (result.drafted, result.blocked, result.not_in_plan) == (1, 1, 3)
    assert store.get_draft(vid(1)) is None and store.get_draft(vid(4)) is None
    assert store.get_draft(vid(2)).description == "mine"


def test_apply_with_no_ids_or_duplicates(store: Store) -> None:
    fill(store, video(1, "x"))
    spec = BulkSpec("replace", find="x", replace_with="y")
    empty = apply_bulk(store, spec, [], NOW)
    assert (empty.drafted, empty.items) == (0, ())
    twice = apply_bulk(store, spec, [vid(1), vid(1)], NOW)
    assert twice.drafted == 1 and len(twice.items) == 1


def test_apply_refuses_more_than_the_limit(store: Store) -> None:
    fill(store, video(1, "x"))
    spec = BulkSpec("replace", find="x", replace_with="y")
    with pytest.raises(ValueError, match="Narrow the dates"):
        apply_bulk(store, spec, [f"id{n:09d}" for n in range(MAX_BULK + 1)], NOW)
    assert store.list_drafts() == []


def test_line_breaks_are_matched_as_newlines(store: Store) -> None:
    fill(store, video(1, "a\r\nb\r\nc"))
    plan = plan_bulk(store, BulkSpec("replace", find="a\nb", replace_with="a b"))
    assert plan[0].description == "a b\nc"
