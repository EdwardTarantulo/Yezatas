"""Bulk edits by asking: words into a checked plan, the guards, planning and drafts.

Model replies are scripted with FakeModelClient; no test touches the network or a real model.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import bulkask
from yezatas.agent import ModelError, ModelReply
from yezatas.bulkask import (ASK_FORMAT, KINDS, MAX_STEPS, PLAN_FORMAT, REQUEST_MAX,
                             SYSTEM_PROMPT, AskError, AskPlan, AskProposal, AskScope, AskStep,
                             Digest, apply_ask, catalog_digest, explain, plan_ask, plan_to_dict,
                             propose, validate_plan)
from yezatas.edits import save_draft
from yezatas.store import Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
OLD = "https://shop.example.com/old"
NEW = "https://shop.example.com/new"


def reply(data: Any, tokens: tuple[int, int, int] = (300, 20, 15)) -> ModelReply:
    text = data if isinstance(data, str) else json.dumps(data)
    return ModelReply(text, (), {}, *tokens)


def step(kind: str, **override: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": kind, "find": "", "replace_with": "",
                            "fields": ["description"], "footer": "", "marker": "",
                            "text": "", "contains": ""}
    data.update(override)
    return data


def replace_step(**override: Any) -> dict[str, Any]:
    return step("replace", **{"find": OLD, "replace_with": NEW, **override})


def plan_data(steps: list[dict[str, Any]], **scope_override: Any) -> dict[str, Any]:
    scope: dict[str, Any] = {"visibility": "public", "published_from": "",
                             "published_to": "", "title_contains": ""}
    scope.update(scope_override)
    return {"steps": steps, "scope": scope, "cannot": ""}


def model_answer(steps: list[dict[str, Any]] | None = None, cannot: str = "",
                 **scope_override: Any) -> dict[str, Any]:
    data = plan_data([replace_step()] if steps is None else steps, **scope_override)
    data["cannot"] = cannot
    return data


def make_video(**override: Any) -> Video:
    base: dict[str, Any] = dict(
        video_id="vid00000001", title="Repotting a Cactus in Five Steps",
        description=f"How to repot a cactus.\n\nSee {OLD} for more.\nRepeated closing line",
        tags=["cactus", "garden"], privacy_status="public",
        published_at="2024-01-10T10:00:00Z", duration="PT10M", view_count=100)
    base.update(override)
    return Video(**base)


def make_store(tmp_path: Path, videos: list[Video]) -> Store:
    store = Store(tmp_path / "c.sqlite")
    pull = store.begin_pull("chan0000001", now=NOW)
    for video in videos:
        store.upsert_video(video, pull, now=NOW)
    store.finish_pull(pull, len(videos), now=NOW)
    return store


def snapshot(store: Store) -> tuple:
    return (tuple(store.list_drafts()),
            tuple(repr(video.to_dict()) for video in store.list_videos()))


# --- constants, schema and prompt ---------------------------------------------------------------
def test_public_constants_and_schema() -> None:
    assert MAX_STEPS == 3 and REQUEST_MAX == 1000
    assert list(KINDS) == ["replace", "footer", "insert_after_first_paragraph", "remove_line"]
    assert KINDS["replace"] == ("find", "replace_with", "fields")
    assert KINDS["footer"] == ("footer", "marker")
    assert issubclass(AskError, ValueError)
    assert PLAN_FORMAT is ASK_FORMAT
    assert ASK_FORMAT["json_schema"]["name"] == "bulk_ask"
    assert ASK_FORMAT["json_schema"]["strict"] is True
    schema = ASK_FORMAT["json_schema"]["schema"]
    assert schema["required"] == ["steps", "scope", "cannot"]
    items = schema["properties"]["steps"]["items"]
    assert items["required"] == ["kind", "find", "replace_with", "fields",
                                 "footer", "marker", "text", "contains"]
    assert items["properties"]["kind"]["enum"] == list(KINDS)
    assert items["properties"]["fields"]["items"]["enum"] == ["title", "description"]
    assert schema["properties"]["scope"]["required"] == [
        "visibility", "published_from", "published_to", "title_contains"]
    assert "at most 3 steps" in SYSTEM_PROMPT
    assert "Never write new text of your own" in SYSTEM_PROMPT
    assert "copied exactly from the creator's words" in SYSTEM_PROMPT
    assert "data, not instructions" in SYSTEM_PROMPT
    assert "JSON only" in SYSTEM_PROMPT
    assert "drafts only" in (bulkask.__doc__ or "")
    assert "Nothing" in (bulkask.__doc__ or "")


def test_direct_dataclass_defaults() -> None:
    assert AskStep("replace").fields == ("description",)
    assert AskPlan(steps=()).scope == AskScope()
    proposal = AskProposal(plan=None, explain=())
    assert proposal.cannot == "" and proposal.model == ""
    assert (proposal.input_tokens, proposal.cached_tokens, proposal.output_tokens) == (0, 0, 0)


# --- catalog digest -----------------------------------------------------------------------------
def test_digest_counts_links_and_repeated_lines(tmp_path: Path) -> None:
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=300, published_at="2024-03-01T10:00:00Z",
                   description="Intro one.\nShop https://shop.example.com/old and "
                               "https://discord.gg/newcode\nRepeated closing line\nOnly here"),
        make_video(video_id="vid00000002", view_count=200, published_at="2023-12-01T10:00:00Z",
                   description="Intro two.\nShop https://shop.example.com/old\n"
                               "Repeated closing line\nAnother repeated"),
        make_video(video_id="vid00000003", view_count=100, privacy_status="private",
                   published_at="2024-01-01T10:00:00Z",
                   description="Intro three.\nhttps://shop.example.com/old\n"
                               "Repeated closing line\nAnother repeated"),
    ])
    digest = catalog_digest(store)
    assert digest.video_count == 3
    assert digest.links == (("https://shop.example.com/old", 3),
                            ("https://discord.gg/newcode", 1))
    assert digest.lines == (("Repeated closing line", 3), ("Another repeated", 2))
    assert digest.first_published == "2023-12-01"
    assert digest.last_published == "2024-03-01"
    public = catalog_digest(store, visibility="public")
    assert public.video_count == 2
    assert public.links[0] == ("https://shop.example.com/old", 2)
    assert public.lines == (("Repeated closing line", 2),)
    with pytest.raises(AskError, match="public only"):
        catalog_digest(store, visibility="private")
    store.close()


def test_digest_excludes_single_and_long_lines_and_writes_nothing(tmp_path: Path) -> None:
    long_line = "x" * 201
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", description=f"One only\n{long_line}\nShared line"),
        make_video(video_id="vid00000002", description=f"Two only\n{long_line}\nShared line"),
    ])
    before = snapshot(store)
    digest = catalog_digest(store)
    assert digest.lines == (("Shared line", 2),)
    assert snapshot(store) == before
    store.close()


def test_digest_caps_links_and_repeated_lines(tmp_path: Path) -> None:
    links = "\n".join(f"https://shop.example.com/item{n}" for n in range(305))
    store = make_store(tmp_path, [make_video(description=links)])
    assert len(catalog_digest(store).links) == 300
    store.close()
    repeated = "\n".join(f"Repeated line {n:03d}" for n in range(205))
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", description=repeated),
        make_video(video_id="vid00000002", description=repeated),
    ])
    # The same database now holds only these two videos: the first video was replaced.
    assert len(catalog_digest(store).lines) == 200
    store.close()


# --- each step kind, planned -------------------------------------------------------------
def test_replace_description_via_validate_and_plan(tmp_path: Path) -> None:
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=200),
        make_video(video_id="vid00000002", view_count=100, description="Nothing to change."),
    ])
    digest = catalog_digest(store)
    request = f"Replace {OLD} with {NEW} in my descriptions."
    plan = validate_plan(plan_data([replace_step()]), request, digest)
    assert plan.steps == (AskStep(kind="replace", find=OLD, replace_with=NEW),)
    assert explain(plan) == (
        f'Replace "{OLD}" with "{NEW}" in descriptions.', "Only public videos.")
    changes = plan_ask(store, plan)
    assert [change.video_id for change in changes] == ["vid00000001"]
    assert changes[0].description == f"How to repot a cactus.\n\nSee {NEW} for more.\nRepeated closing line"
    assert changes[0].count == 1 and changes[0].blocked == ""
    store.close()


def test_replace_title_and_description_counts_steps_not_occurrences(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video(
        title="Cactus Cactus care", description="Cactus and Cactus again.")])
    request = "Change Cactus to Succulent in titles and descriptions."
    data = plan_data([replace_step(find="Cactus", replace_with="Succulent",
                                   fields=["title", "description"])])
    plan = validate_plan(data, request, catalog_digest(store))
    changes = plan_ask(store, plan)
    assert changes[0].title == "Succulent Succulent care"
    assert changes[0].description == "Succulent and Succulent again."
    assert changes[0].count == 1  # one step changed, even with four matches
    assert explain(plan)[0] == 'Replace "Cactus" with "Succulent" in titles and descriptions.'
    store.close()


def test_footer_without_and_with_marker(tmp_path: Path) -> None:
    footer = "--- Links ---\nNew footer https://shop.example.com/new"
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=200,
                   description="Intro.\n--- Links ---\nOld footer line"),
        make_video(video_id="vid00000002", view_count=100, description="Intro only."),
    ])
    request = f"Add this footer:\n{footer}\nThe old block starts with --- Links ---"
    bare = validate_plan(plan_data([step("footer", footer=footer)]), request, catalog_digest(store))
    planned = {change.video_id: change for change in plan_ask(store, bare)}
    assert planned["vid00000002"].description == f"Intro only.\n\n{footer}"
    marked = validate_plan(
        plan_data([step("footer", footer=footer, marker="--- Links ---")]),
        request, catalog_digest(store))
    planned = {change.video_id: change for change in plan_ask(store, marked)}
    assert planned["vid00000001"].description == f"Intro.\n{footer}"
    assert explain(marked)[0] == (
        f'Add the footer "--- Links ---\\nNew footer https://shop.example.com/new" '
        'at the end of descriptions, replacing the block that starts with "--- Links ---".')
    store.close()


def test_insert_after_first_paragraph_cases(tmp_path: Path) -> None:
    text = "Inserted middle line."
    request = f"Add {text} after the first paragraph."
    plan = validate_plan(
        plan_data([step("insert_after_first_paragraph", text=text)]),
        request, Digest(links=(), lines=(), video_count=0))
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=300,
                   description="First paragraph.\n\nSecond paragraph.\n\nThird."),
        make_video(video_id="vid00000002", view_count=200,
                   description="Only paragraph with no blank line."),
        make_video(video_id="vid00000003", view_count=100,
                   description=f"First.\n\n{text}\n\nLast."),
    ])
    planned = {change.video_id: change for change in plan_ask(store, plan)}
    assert planned["vid00000001"].description == (
        f"First paragraph.\n\n{text}\n\nSecond paragraph.\n\nThird.")
    assert planned["vid00000002"].description == f"Only paragraph with no blank line.\n\n{text}"
    assert "vid00000003" not in planned  # the text is already there
    assert explain(plan)[0] == f'Add "{text}" after the first paragraph of descriptions.'
    store.close()


def test_remove_line_removes_and_collapses_blank_lines(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video(
        description="Keep this.\nRemove this sponsor line.\n\n\nKeep that too.")])
    request = "Remove every line that mentions sponsor from my descriptions."
    plan = validate_plan(plan_data([step("remove_line", contains="sponsor")]),
                         request, catalog_digest(store))
    changes = plan_ask(store, plan)
    assert changes[0].description == "Keep this.\n\nKeep that too."
    assert changes[0].count == 1
    assert explain(plan)[0] == 'Remove every description line that contains "sponsor".'
    store.close()


def test_steps_are_applied_in_order(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video(
        description=f"Shop {OLD}\nSponsor line to go\nKeep this.")])
    request = f"Replace {OLD} with {NEW} and remove the Sponsor line."
    data = plan_data([replace_step(), step("remove_line", contains="Sponsor")])
    plan = validate_plan(data, request, catalog_digest(store))
    changes = plan_ask(store, plan)
    assert changes[0].description == f"Shop {NEW}\nKeep this."
    assert changes[0].count == 2
    store.close()


def test_plan_writes_nothing(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    plan = AskPlan(steps=(AskStep(kind="replace", find=OLD, replace_with=NEW),))
    before = snapshot(store)
    assert plan_ask(store, plan)
    assert snapshot(store) == before
    store.close()


# --- scope and explanation ---------------------------------------------------------------
def test_scope_visibility_dates_and_title_filter(tmp_path: Path) -> None:
    def scoped(n: int, **kw: Any) -> Video:
        return make_video(video_id=f"vid{n:08d}", view_count=1000 - n,
                          description=f"Shop {OLD}", **kw)

    store = make_store(tmp_path, [
        scoped(1, title="Cactus Care Basics", published_at="2024-01-01T00:00:00Z"),
        scoped(2, title="My cactus update", published_at="2024-12-31T23:59:59Z"),
        scoped(3, title="Cactus later", published_at="2025-01-01T00:00:00Z"),
        scoped(4, title="Cactus private", privacy_status="private",
               published_at="2024-06-01T00:00:00Z"),
        scoped(5, title="Other topic", published_at="2024-06-01T00:00:00Z"),
    ])
    request = f"In 2024, replace {OLD} with {NEW} for cactus videos."
    data = plan_data([replace_step()], published_from="2024-01-01",
                     published_to="2024-12-31", title_contains="cactus")
    plan = validate_plan(data, request, catalog_digest(store))
    assert plan.scope == AskScope(published_from="2024-01-01", published_to="2024-12-31",
                                  title_contains="cactus")
    assert {change.video_id for change in plan_ask(store, plan)} == {"vid00000001", "vid00000002"}
    everything = validate_plan(
        plan_data([replace_step()], visibility="all", published_from="2024-01-01",
                  published_to="2024-12-31", title_contains="cactus"),
        request, catalog_digest(store))
    assert {change.video_id for change in plan_ask(store, everything)} == {
        "vid00000001", "vid00000002", "vid00000004"}
    store.close()


@pytest.mark.parametrize("scope, sentence", [
    (AskScope(), "Only public videos."),
    (AskScope(visibility="all"), "All videos."),
    (AskScope(published_from="2024-01-01", published_to="2024-12-31"),
     "Only public videos published from 2024-01-01 to 2024-12-31."),
    (AskScope(published_from="2024-01-01"), "Only public videos published from 2024-01-01."),
    (AskScope(published_to="2024-12-31"), "Only public videos published up to 2024-12-31."),
    (AskScope(visibility="all", title_contains="cactus"),
     'All videos with "cactus" in the title.'),
])
def test_scope_sentences(scope: AskScope, sentence: str) -> None:
    plan = AskPlan(steps=(AskStep(kind="remove_line", contains="sponsor"),), scope=scope)
    assert explain(plan) == ('Remove every description line that contains "sponsor".', sentence)


def test_explain_replace_delete_and_footer_without_marker() -> None:
    deleted = AskPlan(steps=(AskStep(kind="replace", find="old line", replace_with=""),))
    assert explain(deleted)[0] == 'Remove "old line" from descriptions.'
    title_only = AskPlan(steps=(AskStep(kind="replace", find="Old", replace_with="New",
                                       fields=("title",)),))
    assert explain(title_only)[0] == 'Replace "Old" with "New" in titles.'
    footer = AskPlan(steps=(AskStep(kind="footer", footer="Follow us!"),))
    assert explain(footer)[0] == 'Add the footer "Follow us!" at the end of descriptions.'


def test_plan_to_dict_round_trips_through_validate(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    request = f"Replace {OLD} with {NEW} in my descriptions."
    plan = validate_plan(plan_data([replace_step()]), request, catalog_digest(store))
    assert plan_to_dict(plan) == plan_data([replace_step()])
    assert validate_plan(plan_to_dict(plan), request, catalog_digest(store)) == plan
    store.close()


# --- the words guard ---------------------------------------------------------------------
def test_invented_replacement_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    invented = "https://shop.example.com/invented"
    data = plan_data([replace_step(replace_with=invented)])
    with pytest.raises(AskError, match="not in your words") as caught:
        validate_plan(data, f"Replace {OLD} in my descriptions.", catalog_digest(store))
    assert invented in str(caught.value)
    store.close()


def test_invented_footer_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    data = plan_data([step("footer", footer="Invented footer line")])
    with pytest.raises(AskError, match="not in your words") as caught:
        validate_plan(data, "Add a footer at the end.", catalog_digest(store))
    assert "Invented footer line" in str(caught.value)
    store.close()


def test_invented_insert_is_refused() -> None:
    data = plan_data([step("insert_after_first_paragraph", text="Invented inserted line.")])
    with pytest.raises(AskError, match="not in your words") as caught:
        validate_plan(data, "Add a line after the first paragraph.",
                      Digest(links=(), lines=(), video_count=0))
    assert "Invented inserted line." in str(caught.value)


def test_find_may_come_only_from_the_digest(tmp_path: Path) -> None:
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=200),
        make_video(video_id="vid00000002", view_count=100),
    ])
    digest = catalog_digest(store)
    by_link = validate_plan(
        plan_data([replace_step()]),
        f"swap my old shop link for {NEW}", digest)
    assert by_link.steps[0].find == OLD
    by_line = validate_plan(
        plan_data([replace_step(find="Repeated closing line", replace_with="")]),
        "Remove the line I end every video with.", digest)
    assert by_line.steps[0].find == "Repeated closing line"
    assert explain(by_line)[0] == 'Remove "Repeated closing line" from descriptions.'
    store.close()


def test_a_year_not_in_the_words_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    data = plan_data([replace_step()], published_from="2024-01-01")
    with pytest.raises(AskError, match="not in your words") as caught:
        validate_plan(data, f"Replace {OLD} with {NEW}.", catalog_digest(store))
    assert "2024" in str(caught.value)
    store.close()


def test_too_many_steps_and_unknown_kind_are_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    digest = catalog_digest(store)
    request = f"Replace {OLD} with {NEW}."
    with pytest.raises(AskError, match="at most 3"):
        validate_plan(plan_data([replace_step()] * 4), request, digest)
    with pytest.raises(AskError, match="not available"):
        validate_plan(plan_data([step("rewrite_everything")]), request, digest)
    with pytest.raises(AskError, match="not available"):
        validate_plan("not a plan", request, digest)  # type: ignore[arg-type]
    store.close()


@pytest.mark.parametrize("bad_step", [
    step("insert_after_first_paragraph", text="   "),
    step("remove_line", contains=""),
    step("footer", footer="  "),
    step("replace", find="", replace_with="x"),
    step("replace", find="same", replace_with="same"),
    step("remove_line", contains="z" * 501),
])
def test_empty_or_oversized_steps_are_refused(bad_step: dict[str, Any]) -> None:
    request = "same x " + "z" * 501
    with pytest.raises(AskError):
        validate_plan(plan_data([bad_step]), request, Digest(links=(), lines=(), video_count=0))


# --- propose: the one model call ---------------------------------------------------------
def test_propose_happy_path_sends_words_and_digest(tmp_path: Path) -> None:
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=200),
        make_video(video_id="vid00000002", view_count=100),
    ])
    request = f"Replace {OLD} with {NEW} in my descriptions."
    client = FakeModelClient([reply(model_answer())])
    proposal = propose(client, store, request)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["response_format"] is ASK_FORMAT
    assert call["messages"][0]["content"] is SYSTEM_PROMPT
    user = call["messages"][1]["content"]
    assert user.startswith(request)
    assert "Videos: 2 videos" in user
    assert f"{OLD} (2 videos)" in user
    assert "Repeated closing line (2 videos)" in user
    assert proposal.plan == AskPlan(steps=(AskStep(kind="replace", find=OLD, replace_with=NEW),))
    assert proposal.explain == (
        f'Replace "{OLD}" with "{NEW}" in descriptions.', "Only public videos.")
    assert proposal.cannot == "" and proposal.model == "muse-spark-1.3"
    assert (proposal.input_tokens, proposal.cached_tokens, proposal.output_tokens) == (300, 20, 15)
    store.close()


def test_propose_cannot_returns_no_plan(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    reason = "That needs rewriting whole descriptions, which these steps cannot do."
    client = FakeModelClient([reply(model_answer(steps=[], cannot=reason))])
    proposal = propose(client, store, "Rewrite all my descriptions in a new voice.")
    assert proposal.plan is None
    assert proposal.explain == ()
    assert proposal.cannot == reason
    assert proposal.model == "muse-spark-1.3"
    assert (proposal.input_tokens, proposal.cached_tokens, proposal.output_tokens) == (300, 20, 15)
    store.close()


def test_propose_model_error_and_bad_json_become_ask_errors(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    failing = FakeModelClient([ModelError("rate_limit", "Slow down, then try again.")])
    with pytest.raises(AskError, match="Slow down, then try again"):
        propose(failing, store, f"Replace {OLD} with {NEW}.")
    assert len(failing.calls) == 1
    with pytest.raises(AskError, match="could not be read"):
        propose(FakeModelClient([reply("not json at all")]), store,
                f"Replace {OLD} with {NEW}.")
    store.close()


@pytest.mark.parametrize("words", ["", "   ", "\n\t "])
def test_propose_empty_request_makes_no_call(tmp_path: Path, words: str) -> None:
    store = make_store(tmp_path, [make_video()])
    client = FakeModelClient([])
    with pytest.raises(AskError, match="Describe the change"):
        propose(client, store, words)
    assert client.calls == []
    store.close()


def test_propose_long_request_makes_no_call(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    client = FakeModelClient([])
    with pytest.raises(AskError, match="at most 1000 characters"):
        propose(client, store, "x" * (REQUEST_MAX + 1))
    assert client.calls == []
    store.close()


# --- apply: drafts only ------------------------------------------------------------------
def test_apply_makes_drafts_only_for_approved_unblocked_videos(tmp_path: Path) -> None:
    store = make_store(tmp_path, [
        make_video(video_id="vid00000001", view_count=400),
        make_video(video_id="vid00000002", view_count=300),
        make_video(video_id="vid00000003", view_count=200),
        make_video(video_id="vid00000004", view_count=100, description="No link here."),
    ])
    plan = AskPlan(steps=(AskStep(kind="replace", find=OLD, replace_with=NEW),))
    save_draft(store, "vid00000003", "Repotting a Cactus in Five Steps", "mine",
               ["cactus", "garden"], now=NOW)
    approved = ["vid00000001", "vid00000001", "vid00000003", "vid00000004", "vid00000009"]
    result = apply_ask(store, plan, approved, now=NOW)
    assert (result.drafted, result.blocked, result.not_in_plan) == (1, 1, 2)
    assert [(item.video_id, item.status) for item in result.items] == [
        ("vid00000001", "drafted"), ("vid00000003", "blocked"),
        ("vid00000004", "not_in_plan"), ("vid00000009", "not_in_plan")]
    assert store.get_draft("vid00000002") is None  # not approved
    row = store.get_draft("vid00000001")
    assert row is not None
    assert row.source == "bulk-ask"
    assert row.note.startswith("Bulk by asking: ")
    assert f'Replace "{OLD}" with "{NEW}" in descriptions.' in row.note
    assert "Only public videos." in row.note
    assert row.description == f"How to repot a cactus.\n\nSee {NEW} for more.\nRepeated closing line"
    assert row.tags == ("cactus", "garden") and row.approved_at is None
    existing = store.get_draft("vid00000003")
    assert existing is not None and existing.description == "mine"  # untouched
    store.close()


def test_apply_with_no_ids_and_over_the_limit(tmp_path: Path) -> None:
    store = make_store(tmp_path, [make_video()])
    plan = AskPlan(steps=(AskStep(kind="replace", find=OLD, replace_with=NEW),))
    empty = apply_ask(store, plan, [], now=NOW)
    assert (empty.drafted, empty.blocked, empty.not_in_plan, empty.items) == (0, 0, 0, ())
    with pytest.raises(AskError, match="Narrow the dates"):
        apply_ask(store, plan, [f"vid{n:08d}" for n in range(501)], now=NOW)
    assert store.list_drafts() == []
    store.close()

