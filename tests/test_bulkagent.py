"""The bulk edit agent: a scripted fake model drives the real tools over a made-up catalog.

No network, no model and no YouTube: nothing here can publish. The store only gets drafts, and only
through `apply`."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from yezatas import agent, bulk, bulkagent, copilot, edits
from yezatas.agent import ModelError, ModelReply, ToolCall
from yezatas.bulkagent import BulkAgentError, BulkSession
from yezatas.store import Store, TranscriptRecord, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
OLD = "https://discord.gg/old"
NEW = "https://discord.gg/new"
REQUEST = f"Swap {OLD} for {NEW} on videos from 2024"
SWAP = {"steps": [{"kind": "replace", "find": OLD, "replace_with": NEW, "fields": ["description"]}],
        "all_matching": {"text": OLD, "published_from": "2024-01-01", "published_to": "2024-12-31"}}
LONG_TITLE = "Cactus basics for beginners: everything you need to start growing a cactus today"


def vid(n: int) -> str:
    return f"vid{n:08d}"


def make_video(n: int, **fields: Any) -> Video:
    base = Video(video_id=vid(n), channel_id="UC_fake", title=f"Video title {n}",
                 description=f"Plain text of video {n}.", tags=["cactus", "garden"],
                 privacy_status="public", published_at=f"2024-0{n}-10T10:00:00Z",
                 duration="PT5M", view_count=100 * n)
    return replace(base, **fields)


def catalog_videos() -> list[Video]:
    """Seven made-up videos. 1, 2, 6 and 4 (private) have the old link and are from 2024; 7 has it too
    but is from 2025; 6 will get a draft; 5 has no link; 3 has a long title."""
    return [
        make_video(1, title="Cactus basics", description=f"Welcome.\nJoin: {OLD}\nThanks for watching."),
        make_video(2, title="Watering tips", description=f"Water slowly.\n{OLD}\n{OLD}"),
        make_video(3, title=LONG_TITLE, description="Mix soil and water."),
        make_video(4, title="Private clip", description=f"Join {OLD}", privacy_status="private"),
        make_video(5, title="Soil mix", description="Mix soil."),
        make_video(6, title="Pots", description=f"Pots.\nJoin: {OLD}"),
        make_video(7, title="Repotting", description=f"Repot in spring.\nJoin: {OLD}",
                   published_at="2025-03-10T10:00:00Z"),
    ]


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pull = s.begin_pull("UC_fake", NOW)
        for v in catalog_videos():
            s.upsert_video(v, pull, NOW)
        s.finish_pull(pull, len(catalog_videos()), NOW)
        s.add_transcript(TranscriptRecord(0, vid(5), "en", "voice-test", "clip.mp4", 70_000, [
            {"start_ms": 0, "end_ms": 5000, "text": "Use 3 parts of sand"}], [], ""), NOW)
        yield s


def ref(store: Store, n: int) -> str:
    return next(r for r, v in copilot.catalog_refs(store.list_videos()).items() if v == vid(n))


def tool_reply(*calls: tuple[str, Any], tokens: tuple[int, int, int] = (0, 0, 0)) -> ModelReply:
    tool_calls = tuple(
        ToolCall(f"call_{i}_{name}", name, args if isinstance(args, str) else json.dumps(args))
        for i, (name, args) in enumerate(calls))
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in tool_calls]}
    return ModelReply(None, tool_calls, message, *tokens)


def answer(text: str) -> ModelReply:
    return ModelReply(text, (), {"role": "assistant", "content": text})


def go(store: Store, script: list[Any], text: str = REQUEST, session: BulkSession | None = None,
       **kwargs: Any) -> tuple[BulkSession, FakeModelClient]:
    session = session or BulkSession()
    client = kwargs.pop("client", None) or FakeModelClient(script)
    bulkagent.run(store, client, session, text, as_of=AS_OF, now=NOW, **kwargs)
    return session, client


def tool_results(client: FakeModelClient, call_index: int) -> list[dict[str, Any]]:
    """The tool results the model got in the messages of model call `call_index`."""
    return [json.loads(m["content"]) for m in client.calls[call_index]["messages"] if m["role"] == "tool"]


def staged_ids(session: BulkSession) -> set[str]:
    return {s.video_id for s in session.staged_list()}


# --- the swap: find_text + apply_steps ------------------------------------------------------------
def test_exact_swap_stages_the_right_videos_and_leaves_the_others(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("find_text", {"text": OLD})),
        tool_reply(("apply_steps", SWAP)),
        tool_reply(("show_staged", {})),
        answer(f"I swapped the old invite for {NEW}."),
    ])
    # public videos from 2024 that have the link: 1, 2 and 6. Not 5 (no link), 7 (2025), 4 (private).
    assert staged_ids(session) == {vid(1), vid(2), vid(6)}
    one = session.get(vid(2))
    assert one is not None
    assert one.description == f"Water slowly.\n{NEW}\n{NEW}"
    assert one.description0 == f"Water slowly.\n{OLD}\n{OLD}" and one.title == one.title0
    assert session.get(vid(7)) is None and session.get(vid(5)) is None
    found = tool_results(client, 1)[0]
    assert found["count"] == 4 and found["occurrences"] == 5  # public ones; counted by code
    assert {m["ref"] for m in found["matches"]} == {ref(store, n) for n in (1, 2, 6, 7)}
    applied = tool_results(client, 2)[1]
    assert applied["matched"] == 3 and applied["changed"] == 3 and applied["refused"] == []
    assert "Searched for \"https://discord.gg/old\": 4 videos." in session.log
    assert "Replaced text in 3 videos." in session.log
    assert session.log[0] == "Read the catalog: 7 videos."
    assert session.closing == f"I swapped the old invite for {NEW}."


def test_find_text_is_exact_and_counts_by_code(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("find_text", {"text": OLD.upper()}),
                   ("find_text", {"text": "join:", "in": ["description"]}),
                   ("find_text", {"text": "Join:", "visibility": "all"})),
        answer("Done looking."),
    ])
    first, second, third = tool_results(client, 1)
    assert first["count"] == 0 and second["count"] == 0  # capital letters count
    assert third["count"] == 3 and third["matches"][0]["line"].startswith("Join:")
    assert "Searched for \"https://discord.gg/old\"" not in " ".join(session.log)


def test_visibility_defaults_to_public_and_all_includes_private(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["x"]}],
                                    "all_matching": {"text": "Join"}})),
        tool_reply(("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["y"]}],
                                    "all_matching": {"text": "Join", "visibility": "all"}})),
        answer("ok"),
    ])
    assert staged_ids(session) == {vid(1), vid(4), vid(6), vid(7)}
    assert session.get(vid(4)).tags == ("cactus", "garden", "y")  # type: ignore[union-attr]
    assert session.get(vid(1)).tags == ("cactus", "garden", "x", "y")  # type: ignore[union-attr]


# --- every step kind -------------------------------------------------------------------------------
def step(kind: str, **fields: Any) -> bulk.Step:
    return bulk.step_from_dict({"kind": kind, **fields})


@pytest.mark.parametrize("kind, fields, title, description, tags, expected", [
    ("replace", {"find": "old", "replace_with": "new", "fields": ["title", "description"]},
     "an old title", "old and old", ("a",), ("an new title", "new and new", ("a",))),
    ("replace", {"find": "old", "replace_with": ""}, "old", "old text", (), ("old", " text", ())),
    ("footer", {"footer": "Links:\nhttps://x.example/a"}, "T", "Body", (),
     ("T", "Body\n\nLinks:\nhttps://x.example/a", ())),
    ("footer", {"footer": "Links:\nhttps://x.example/a"}, "T", "Body\n\nLinks:\nhttps://x.example/a", (),
     ("T", "Body\n\nLinks:\nhttps://x.example/a", ())),
    ("insert_after_first_paragraph", {"text": "Hello"}, "T", "First.\n\nSecond.", (),
     ("T", "First.\n\nHello\n\nSecond.", ())),
    ("insert_after_first_paragraph", {"text": "Hello"}, "T", "Only one", (), ("T", "Only one\n\nHello", ())),
    ("remove_line", {"contains": "promo"}, "T", "a\npromo here\nb", (), ("T", "a\nb", ())),
    ("add_first_line", {"text": "New channel!"}, "T", "Body", (), ("T", "New channel!\nBody", ())),
    ("add_first_line", {"text": "New channel!"}, "T", "New channel!\nBody", (),
     ("T", "New channel!\nBody", ())),
    ("title_prefix", {"text": "[Tutorial]"}, "Pots", "d", (), ("[Tutorial] Pots", "d", ())),
    ("title_prefix", {"text": "[Tutorial]"}, "[Tutorial] Pots", "d", (), ("[Tutorial] Pots", "d", ())),
    ("title_suffix", {"text": "| Garden"}, "Pots", "d", (), ("Pots | Garden", "d", ())),
    ("title_suffix", {"text": "| Garden"}, "Pots | Garden", "d", (), ("Pots | Garden", "d", ())),
    ("add_tags", {"tags": ["Cactus", "new"]}, "T", "d", ("cactus",), ("T", "d", ("cactus", "new"))),
    ("remove_tags", {"tags": ["CACTUS"]}, "T", "d", ("cactus", "garden"), ("T", "d", ("garden",))),
])
def test_each_step_kind(kind: str, fields: dict[str, Any], title: str, description: str,
                        tags: tuple[str, ...], expected: tuple[Any, ...]) -> None:
    assert bulk.apply_step(step(kind, **fields), title, description, tags) == expected


@pytest.mark.parametrize("raw, message", [
    ({"kind": "teleport"}, "needs a \"kind\""),
    ("replace", "must be an object"),
    ({"kind": "replace", "find": "", "replace_with": "x"}, "text to find"),
    ({"kind": "replace", "find": "a", "replace_with": "a"}, "are the same"),
    ({"kind": "replace", "find": "a", "replace_with": "b", "fields": ["tags"]}, "title, the description"),
    ({"kind": "replace", "find": "a", "replace_with": "b", "text": "stray"}, "does not take"),
    ({"kind": "add_tags", "tags": []}, "list of text"),
    ({"kind": "add_tags", "tags": ["a,b"]}, "comma"),
    ({"kind": "title_prefix", "text": "a\nb"}, "one line"),
    ({"kind": "title_suffix", "text": "x" * 101}, "limit"),
    ({"kind": "add_first_line", "text": ""}, "Enter the text"),
    ({"kind": "remove_line", "contains": ""}, "Enter the text"),
    ({"kind": "footer", "footer": ""}, "footer text"),
])
def test_bad_steps_raise_plain_sentences(raw: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        bulk.step_from_dict(raw)


def test_unused_step_fields_may_be_empty() -> None:
    parsed = bulk.step_from_dict({"kind": "title_prefix", "text": "A", "find": "", "tags": [],
                                  "fields": None, "contains": ""})
    assert parsed == bulk.Step("title_prefix", text="A")


def test_new_step_kinds_run_through_apply_steps_on_top_of_each_other(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("apply_steps", {
            "steps": [{"kind": "title_prefix", "text": "[Guide]"}, {"kind": "add_tags", "tags": ["guide"]},
                      {"kind": "add_first_line", "text": "New here? Subscribe."}],
            "refs": [ref(store, 5)]})),
        tool_reply(("apply_steps", {"steps": [{"kind": "title_suffix", "text": "(2024)"}],
                                    "refs": [ref(store, 5)]})),
        answer("Done."),
    ], text="Prefix [Guide] to Soil mix, a tag guide, a first line New here? Subscribe. and (2024)")
    change = session.get(vid(5))
    assert change is not None
    assert change.title == "[Guide] Soil mix (2024)"
    assert change.description == "New here? Subscribe.\nMix soil."
    assert change.tags == ("cactus", "garden", "guide")
    assert change.title0 == "Soil mix" and change.tags0 == ("cactus", "garden")
    assert "Added a prefix to the titles of 1 video." in session.log
    assert "Added a suffix to the titles of 1 video." in session.log
    assert "Added tags to 1 video." in session.log and "Added a first line to 1 video." in session.log


def test_an_unchanged_video_is_counted_not_staged(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["cactus"]}],
                                    "refs": [ref(store, 1)]})),
        answer("Nothing to do."),
    ], text="Make sure every video has the cactus tag")
    result = tool_results(client, 1)[0]
    assert result["changed"] == 0 and result["unchanged"] == 1 and staged_ids(session) == set()


# --- edit_videos and the guards ---------------------------------------------------------------------
def edit(store: Store, n: int, **fields: Any) -> dict[str, Any]:
    return {"ref": ref(store, n), **fields}


def test_an_invented_number_is_refused_returned_fixed_and_then_staged(store: Store) -> None:
    long_ref = ref(store, 3)
    session, client = go(store, [
        tool_reply(("edit_videos", {"edits": [{"ref": long_ref, "title": "Cactus basics in 99 seconds",
                                               "reason": "Shorter."}]})),
        tool_reply(("edit_videos", {"edits": [{"ref": long_ref, "title": "Cactus basics for beginners",
                                               "reason": "Shorter."}]})),
        answer("I shortened the title."),
    ], text="Shorten every title over 60 characters")
    first = tool_results(client, 1)[0]["results"][0]
    assert first["ok"] is False and "99" in " ".join(first["problems"])
    assert tool_results(client, 2)[1]["results"][0]["ok"] is True
    change = session.get(vid(3))
    assert change is not None and change.title == "Cactus basics for beginners"
    assert change.reason == "Shorter."
    assert any(line.startswith("Wrote new titles for 1 video") for line in session.log)
    assert any("1 was refused by the checks and sent back to the model" in line for line in session.log)


def test_a_number_in_the_creators_words_is_allowed(store: Store) -> None:
    session, _ = go(store, [
        tool_reply(("edit_videos", {"edits": [edit(store, 3, title="Cactus basics in 99 seconds")]})),
        answer("Done."),
    ], text="Make V3 shorter, something like 99 seconds in the title")
    assert session.get(vid(3)) is not None


def test_links_from_the_creators_words_pass_and_invented_links_do_not(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("edit_videos", {"edits": [
            edit(store, 5, description=f"Mix soil.\nJoin {NEW}"),
            edit(store, 3, description="Mix soil and water.\nJoin https://discord.gg/invented")]})),
        answer("One is staged."),
    ], text=f"Add {NEW} to Soil mix")
    results = tool_results(client, 1)[0]["results"]
    assert results[0]["ok"] is True
    assert results[1]["ok"] is False and "discord.gg/invented" in results[1]["problems"][0]
    assert staged_ids(session) == {vid(5)}


def test_a_video_link_is_allowed_only_after_a_tool_showed_it(store: Store) -> None:
    other = f"https://www.youtube.com/watch?v={vid(2)}"
    soil = edit(store, 5, description=f"Mix soil.\nWatch next: {other}")
    session, client = go(store, [
        tool_reply(("edit_videos", {"edits": [soil]})),  # never looked: refused
        tool_reply(("find_videos", {"limit": 10})),
        tool_reply(("edit_videos", {"edits": [soil]})),
        answer("Added."),
    ], text="Add a Watch next link to Soil mix")
    assert tool_results(client, 1)[0]["results"][0]["ok"] is False
    shown = tool_results(client, 2)[1]["videos"]
    assert any(v.get("url") == other for v in shown)
    assert tool_results(client, 3)[2]["results"][0]["ok"] is True
    assert session.get(vid(5)) is not None
    assert "private" not in json.dumps([v.get("url") for v in shown if v["ref"] == ref(store, 4)])


def test_numbers_in_a_title_a_tool_returned_are_allowed(store: Store) -> None:
    s = store
    s.upsert_video(make_video(5, title="Soil mix 2", description="Mix soil."), s.latest_pull()["id"], NOW)
    session, _ = go(s, [
        tool_reply(("find_videos", {"limit": 10})),
        tool_reply(("edit_videos", {"edits": [edit(s, 1, description="Welcome.\nSee also: Soil mix 2")]})),
        answer("ok"),
    ], text="Mention the soil video in V1")
    assert session.get(vid(1)) is not None


def test_validator_errors_are_refused_too(store: Store) -> None:
    _, client = go(store, [
        tool_reply(("edit_videos", {"edits": [
            edit(store, 5, title="A <b>bold</b> title"),
            edit(store, 3, description="x" * 5001),
            edit(store, 1, tags=["a,b"])]})),
        answer("No."),
    ])
    results = tool_results(client, 1)[0]["results"]
    assert [r["ok"] for r in results] == [False, False, False]
    assert "< or >" in results[0]["problems"][0] and "bytes" in results[1]["problems"][0]


def test_the_creators_own_handle_is_allowed_but_other_handles_are_not(store: Store) -> None:
    _, client = go(store, [
        tool_reply(("edit_videos", {"edits": [
            edit(store, 5, description="Mix soil.\nFollow @mygarden"),
            edit(store, 3, description="Mix soil and water.\nFollow @someoneelse")]})),
        answer("Partly."),
    ], text="Add Follow @mygarden to Soil mix")
    ok = [r["ok"] for r in tool_results(client, 1)[0]["results"]]
    assert ok == [True, False]


def test_edit_videos_limits_and_shapes(store: Store) -> None:
    many = [{"ref": ref(store, 5), "title": f"T{i}"} for i in range(bulkagent.MAX_EDITS_PER_CALL + 1)]
    _, client = go(store, [
        tool_reply(("edit_videos", {"edits": many}), ("edit_videos", {"edits": []}),
                   ("edit_videos", {"edits": ["x", {"ref": ref(store, 5)}, {"ref": "V99", "title": "x"},
                                              {"ref": ref(store, 5), "title": 5},
                                              {"ref": ref(store, 5), "bogus": 1}]})),
        answer("Nothing."),
    ])
    too_many, empty, shapes = tool_results(client, 1)
    assert "At most 25 edits" in too_many["error"] and "list of edits" in empty["error"]
    problems = [" ".join(r["problems"]) for r in shapes["results"]]
    assert "must be an object" in problems[0] and "new title" in problems[1]
    assert "No video V99" in problems[2] and "must be text" in problems[3] and "Unknown arguments" in problems[4]


def test_the_staging_cap_follows_max_bulk(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bulk, "MAX_BULK", 2)
    session, client = go(store, [
        tool_reply(("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["z"]}],
                                    "all_matching": {}})),
        answer("Capped."),
    ], text="Add the tag z everywhere")
    result = tool_results(client, 1)[0]
    assert len(session.staged_list()) == 2 and result["changed"] == 2
    assert result["refused"][0]["reason"].startswith("At most 2 videos")
    assert result["refused"][0]["videos"] == 4


def test_a_staged_text_that_is_back_to_the_original_leaves_the_stage(store: Store) -> None:
    session, _ = go(store, [
        tool_reply(("edit_videos", {"edits": [edit(store, 5, title="Soil mix, better")]})),
        tool_reply(("edit_videos", {"edits": [edit(store, 5, title="Soil mix")]})),
        answer("Back."),
    ])
    assert session.get(vid(5)) is None


def test_show_staged_get_video_and_unstage(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("edit_videos", {"edits": [edit(store, 3, title="Cactus basics", reason="Short.")]}),
                   ("apply_steps", {"steps": [{"kind": "add_first_line", "text": "Hi!"}],
                                    "refs": [ref(store, 5)]})),
        tool_reply(("show_staged", {}), ("get_video", {"ref": ref(store, 3)}),
                   ("unstage", {"refs": [ref(store, 5), ref(store, 1)]})),
        tool_reply(("show_staged", {"limit": 5})),
        answer("Looked."),
    ])
    shown, video, gone = tool_results(client, 2)[-3:]
    assert shown["count"] == 2
    rows = {r["ref"]: r for r in shown["staged"]}
    assert rows[ref(store, 3)]["title_after"] == "Cactus basics" and rows[ref(store, 3)]["reason"] == "Short."
    assert rows[ref(store, 5)]["description_lines_added"] == 1
    assert video["staged"]["title"] == "Cactus basics" and video["title"] == LONG_TITLE
    assert gone["unstaged"] == [ref(store, 5)] and gone["not_staged"] == [ref(store, 1)]
    assert tool_results(client, 3)[-1]["count"] == 1
    assert staged_ids(session) == {vid(3)}


def test_find_videos_extras_long_titles_pages_and_staged_flag(store: Store) -> None:
    _, client = go(store, [
        tool_reply(("find_videos", {"title_longer_than": 60}),
                   ("find_videos", {"limit": 2, "offset": 2, "sort": "oldest"}),
                   ("find_videos", {"title_longer_than": "x"})),
        answer("ok"),
    ])
    long, page, bad = tool_results(client, 1)
    assert long["count"] == 1 and long["videos"][0]["title"] == LONG_TITLE and long["videos"][0]["staged"] is False
    assert [v["ref"] for v in page["videos"]] == [ref(store, 3), ref(store, 4)] and page["offset"] == 2
    assert "whole number" in bad["error"]


def test_a_video_with_a_draft_is_staged_but_blocked(store: Store) -> None:
    edits.save_draft(store, vid(6), "Pots (edited)", f"Pots.\nJoin: {OLD}", ["cactus"])
    session, client = go(store, [
        tool_reply(("apply_steps", SWAP)),
        answer("Done."),
    ])
    assert vid(6) in staged_ids(session)
    assert tool_results(client, 1)[0]["staged_but_blocked_by_an_existing_draft"] == 1
    reviewed = {r.staged.video_id: r for r in bulkagent.check_staged(store, session)}
    assert reviewed[vid(6)].blocked == bulk.BLOCKED_HAS_DRAFT
    assert reviewed[vid(1)].blocked == ""


# --- tools never crash the loop -----------------------------------------------------------------------
def test_bad_calls_return_errors_and_the_loop_goes_on(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("get_video", {"ref": "V99"}), ("get_video", "not json"), ("get_video", "[1]"),
                   ("no_such_tool", {}), ("propose_edit", {"ref": "V1"}),
                   ("apply_steps", {"steps": "x"}), ("apply_steps", {"steps": [{"kind": "nope"}], "refs": ["V1"]}),
                   ("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["a"]}]}),
                   ("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["a"]}], "refs": ["V1"],
                                    "all_matching": {}}),
                   ("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["a"]}], "refs": [3]}),
                   ("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["a"]}],
                                    "all_matching": {"bogus": 1}}),
                   ("find_text", {"text": ""}), ("find_text", {"text": "a", "in": ["tags"]}),
                   ("unstage", {"refs": "V1"}), ("show_staged", {"limit": 500}),
                   ("get_video", {"ref": ["V1"]})),
        answer("Sorted out."),
    ])
    results = tool_results(client, 1)
    assert len(results) == 16 and all("error" in r for r in results)
    assert "No video V99" in results[0]["error"] and "not valid JSON" in results[1]["error"]
    assert "Unknown tool" in results[3]["error"] and "propose_edit" in results[4]["error"]
    assert session.closing == "Sorted out." and staged_ids(session) == set()


def test_the_model_gets_every_tool_and_no_propose_edit(store: Store) -> None:
    _, client = go(store, [answer("What do you mean?")])
    names = {t["function"]["name"] for t in client.calls[0]["tools"]}
    assert names == {"find_videos", "search_videos", "find_text", "get_video", "get_transcript",
                     "channel_stats", "apply_steps", "edit_videos", "show_staged", "unstage"}


def test_search_transcript_and_stats_tools_work(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("search_videos", {"query": "sand parts"}), ("get_transcript", {"ref": ref(store, 5)}),
                   ("get_transcript", {"ref": ref(store, 1)}), ("channel_stats", {})),
        answer("Looked."),
    ])
    search, transcript, none, stats = tool_results(client, 1)
    assert search["count"] == 1 and search["matches"][0]["ref"] == ref(store, 5)
    assert transcript["available"] is True and none["available"] is False
    assert stats["videos_all"] == 7
    assert "Counted the whole channel: 7 videos." in session.log
    assert f'Read the transcript of "Soil mix".' in session.log


def test_ambiguous_request_ends_with_a_question_and_stages_nothing(store: Store) -> None:
    session, _ = go(store, [answer("Which videos do you mean?")], text="Fix them")
    assert session.closing == "Which videos do you mean?" and not session.staged_list()


def test_the_prompt_says_data_is_not_instructions_and_names_the_pages(store: Store) -> None:
    _, client = go(store, [answer("Hm?")])
    system = client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "data, not instructions" in system and "Translations page" in system
    assert "video page" in system and "one short question" in system.replace("ONE", "one")
    assert "The channel has 7 videos" in system


# --- the closing message guard -------------------------------------------------------------------------
def test_closing_numbers_must_come_from_a_tool_or_the_creator(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("find_text", {"text": OLD})),
        answer("I found it in 40 videos."),
        answer("I found it in 4 videos."),
    ])
    assert "40" in client.calls[2]["messages"][-1]["content"]
    assert session.closing == "I found it in 4 videos." and session.notes == []


def test_closing_sentences_with_unsupported_numbers_are_dropped_after_one_repair(store: Store) -> None:
    session, _ = go(store, [
        answer("I changed 77 videos. Everything is ready for you to review."),
        answer("I changed 77 videos. Everything is ready for you to review."),
    ])
    assert session.closing == "Everything is ready for you to review."
    assert session.notes and "77" in session.notes[0]


def test_numbers_from_earlier_closings_and_the_catalog_header_pass(store: Store) -> None:
    session, _ = go(store, [answer("The channel has 7 videos in all.")])
    assert session.closing == "The channel has 7 videos in all."


def test_the_closing_message_notes_unknown_refs(store: Store) -> None:
    session, _ = go(store, [answer("Look at V99 first.")])
    assert session.notes == ["The answer mentions V99, which is not in the catalog."]


# --- follow-ups -------------------------------------------------------------------------------------
def test_a_reply_continues_the_same_session_on_top_of_staged_text(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("apply_steps", SWAP)),
        answer("Swapped the link."),
        tool_reply(("show_staged", {}),
                   ("apply_steps", {"steps": [{"kind": "title_suffix", "text": "| Garden"}],
                                    "refs": [ref(store, 1)]})),
        answer("Added the suffix to V1."),
    ])
    first_log = len(session.log)
    bulkagent.run(store, client, session, "Also add | Garden to the first one", as_of=AS_OF, now=NOW)
    assert session.requests == [REQUEST, "Also add | Garden to the first one"]
    one = session.get(vid(1))
    assert one is not None and one.title == "Cactus basics | Garden"
    assert NEW in one.description  # the swap from the first turn is still there
    assert one.title0 == "Cactus basics" and one.description0 == catalog_videos()[0].description
    # the model saw the whole conversation, with the first turn's request and answer
    third_call = client.calls[2]["messages"]
    assert [m["role"] for m in third_call if m["role"] == "user"] == ["user", "user"]
    assert any(m.get("content") == "Swapped the link." for m in third_call)
    assert session.log[first_log] == 'Read your reply: "Also add | Garden to the first one".'
    assert session.closing == "Added the suffix to V1." and session.log[0] == "Read the catalog: 7 videos."
    # the second turn's staged text is also checked with the words of both turns
    urls, _ = session.allowed()
    assert NEW in urls


def test_a_new_turn_clears_the_last_closing_and_notes(store: Store) -> None:
    session, client = go(store, [answer("Look at V99."), answer("Fine now.")])
    assert session.notes
    bulkagent.run(store, client, session, "ok, thanks", as_of=AS_OF)
    assert session.closing == "Fine now." and session.notes == []


def test_a_pulled_again_catalog_is_noted(store: Store) -> None:
    session, client = go(store, [answer("First."), answer("Second.")])
    pull = store.begin_pull("UC_fake", NOW)
    store.upsert_video(make_video(8, title="Brand new", published_at="2023-01-01T10:00:00Z"), pull, NOW)
    store.finish_pull(pull, 8, NOW)
    bulkagent.run(store, client, session, "Again", as_of=AS_OF)
    assert any("pulled again" in line for line in session.log)
    assert "refs may have shifted" in client.calls[1]["messages"][-1]["content"]


# --- limits and the ways a turn ends -------------------------------------------------------------------
def test_max_rounds_ends_the_turn_cleanly_and_keeps_the_stage(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bulkagent, "MAX_ROUNDS", 3)
    script = [tool_reply(("apply_steps", SWAP))] + [tool_reply(("show_staged", {}))] * 5
    session, client = go(store, script)
    assert len(client.calls) == 3
    assert len(session.staged_list()) == 3
    assert session.log[-1].startswith("The agent reached its limit of 3 model calls")
    assert "reply to continue" in session.log[-1] and session.closing == ""
    assert session.notes == [session.log[-1]] and not session.busy
    # and it can go on from there
    client.script[:] = [answer("Finished.")]
    bulkagent.run(store, client, session, "go on", as_of=AS_OF)
    assert session.closing == "Finished." and len(session.staged_list()) == 3


def test_the_cost_ceiling_ends_the_turn_cleanly_and_keeps_the_stage(store: Store) -> None:
    big = (2_000_000, 0, 0)  # 2.5 dollars of input at muse-spark prices
    assert agent.estimate_cost("muse-spark-1.3", *big) > bulkagent.MAX_RUN_COST_USD
    session, client = go(store, [tool_reply(("apply_steps", SWAP), tokens=big),
                                 answer("Never reached.")])
    assert len(client.calls) == 1 and len(session.staged_list()) == 3
    assert "cost limit of $2.00" in session.log[-1] and "kept" in session.log[-1]
    assert session.tokens() == big
    # the ceiling is per session: a reply stops at once
    client.script[:] = [answer("Never reached.")]
    bulkagent.run(store, client, session, "more", as_of=AS_OF)
    assert len(client.calls) == 1 and "cost limit" in session.log[-1]


def test_a_model_without_a_price_is_not_capped(store: Store) -> None:
    client = FakeModelClient([tool_reply(("show_staged", {}), tokens=(9_000_000, 0, 0)), answer("Fine.")],
                             model="local-model")
    session, _ = go(store, [], client=client)
    assert session.closing == "Fine."


def test_the_stop_flag_ends_the_turn_cleanly_and_keeps_the_stage(store: Store) -> None:
    session = BulkSession()

    def stop_while_working(messages: Any, tools: Any, fmt: Any) -> ModelReply:
        session.stop.set()  # the creator pressed Stop during this model call
        return tool_reply(("apply_steps", SWAP))

    session, client = go(store, [stop_while_working, answer("Never reached.")], session=session)
    assert len(client.calls) == 1 and len(session.staged_list()) == 3
    assert session.log[-1].startswith("Stopped at your request") and "reply to continue" in session.log[-1]
    assert session.closing == ""
    # the flag is cleared when the creator replies
    client.script[:] = [answer("Carrying on.")]
    bulkagent.run(store, client, session, "continue", as_of=AS_OF)
    assert session.closing == "Carrying on." and len(session.staged_list()) == 3


def test_a_model_error_keeps_the_stage_and_says_so(store: Store) -> None:
    session, client = go(store, [tool_reply(("apply_steps", SWAP)),
                                 ModelError("rate_limit", "Meta's Model API is busy.")])
    assert len(session.staged_list()) == 3
    assert session.error == "Meta's Model API is busy."
    assert "The model could not continue: Meta's Model API is busy. What is staged is kept." in session.log[-1]
    assert session.log[-1].endswith("Reply to try again.") and not session.busy
    # a reply goes on from there
    client.script[:] = [answer("Back again.")]
    bulkagent.run(store, client, session, "try again", as_of=AS_OF)
    assert session.closing == "Back again." and session.error == ""


def test_an_auth_error_does_not_suggest_a_retry(store: Store) -> None:
    session, _ = go(store, [ModelError("auth", "The key was refused.")])
    assert "Reply to try again" not in session.log[-1] and session.error


def test_bad_requests_raise_before_anything_starts(store: Store, tmp_path: Path) -> None:
    session = BulkSession()
    for text in ("", "   "):
        with pytest.raises(BulkAgentError, match="few words"):
            bulkagent.run(store, FakeModelClient([]), session, text, as_of=AS_OF)
    with pytest.raises(BulkAgentError, match="1,000 characters"):
        bulkagent.run(store, FakeModelClient([]), session, "x" * (bulkagent.REQUEST_MAX + 1), as_of=AS_OF)
    bulkagent.run(store, FakeModelClient([answer("ok")]), BulkSession(), "x" * bulkagent.REQUEST_MAX, as_of=AS_OF)
    assert not session.requests and not session.started
    with Store(tmp_path / "empty.sqlite") as empty, pytest.raises(BulkAgentError, match="no pulled catalog"):
        bulkagent.run(empty, FakeModelClient([]), session, "Do it", as_of=AS_OF)


def test_two_turns_cannot_run_at_once(store: Store, tmp_path: Path) -> None:
    session = BulkSession()
    entered, release = threading.Event(), threading.Event()

    def slow(messages: Any, tools: Any, fmt: Any) -> ModelReply:
        entered.set()
        assert release.wait(10)
        return answer("Done.")

    def work() -> None:
        with Store(tmp_path / "c.sqlite") as own:  # a connection belongs to one thread
            bulkagent.run(own, FakeModelClient([slow]), session, "First", as_of=AS_OF)

    worker = threading.Thread(target=work)
    worker.start()
    assert entered.wait(10)
    with pytest.raises(BulkAgentError, match="already working"):
        bulkagent.run(store, FakeModelClient([]), session, "Second", as_of=AS_OF)
    with pytest.raises(BulkAgentError, match="still working"):
        bulkagent.apply(store, session, [vid(1)])
    release.set()
    worker.join(10)
    assert session.closing == "Done."


def test_the_page_can_read_the_session_while_the_worker_runs(store: Store) -> None:
    session = BulkSession()
    seen: list[tuple[int, int]] = []

    def peek(messages: Any, tools: Any, fmt: Any) -> ModelReply:
        seen.append((len(session.log_from(0)), len(session.staged_list())))
        return answer("Done.")

    go(store, [tool_reply(("apply_steps", SWAP)), peek], session=session)
    assert seen and seen[0][1] == 3 and seen[0][0] >= 2


# --- nothing is written before apply -------------------------------------------------------------------
def snapshot(store: Store) -> list[Any]:
    return [(v.video_id, v.title, v.description, tuple(v.tags)) for v in store.list_videos("video_id")]


def test_nothing_is_written_to_the_store_before_apply(store: Store) -> None:
    before = snapshot(store)
    session, _ = go(store, [tool_reply(("apply_steps", SWAP),
                                       ("edit_videos", {"edits": [edit(store, 3, title="Short")]})),
                            answer("Staged four.")])
    assert len(session.staged_list()) == 4
    assert store.list_drafts() == [] and snapshot(store) == before


def stage_swap(store: Store) -> BulkSession:
    session, _ = go(store, [tool_reply(("apply_steps", SWAP)), answer("Staged.")])
    return session


def test_apply_makes_drafts_for_ticked_ids_only(store: Store) -> None:
    session = stage_swap(store)
    result = bulkagent.apply(store, session, [vid(1), vid(2)], NOW)
    assert (result.drafted, result.blocked, result.not_in_plan) == (2, 0, 0)
    rows = {d.video_id: d for d in store.list_drafts()}
    assert set(rows) == {vid(1), vid(2)}
    assert rows[vid(1)].source == "bulk-agent" and rows[vid(1)].draft.description.count(NEW) == 1
    assert rows[vid(1)].note.startswith("Bulk agent: Replaced ")
    assert rows[vid(1)].approved_at is None  # nothing is approved, so nothing can be pushed
    assert store.get_draft(vid(6)) is None
    assert staged_ids(session) == {vid(6)}  # the drafted ones left the list
    assert session.log[-1] == "Made drafts for 2 videos."


def test_apply_blocks_a_video_that_got_a_draft_after_staging(store: Store) -> None:
    session = stage_swap(store)
    edits.save_draft(store, vid(2), "Watering tips (mine)", f"Water slowly.\n{OLD}\n{OLD}", ["cactus"])
    result = bulkagent.apply(store, session, [vid(1), vid(2)], NOW)
    assert (result.drafted, result.blocked) == (1, 1)
    blocked = next(i for i in result.items if i.status == "blocked")
    assert blocked.video_id == vid(2) and blocked.message == bulk.BLOCKED_HAS_DRAFT
    assert store.get_draft(vid(2)).draft.title == "Watering tips (mine)"  # type: ignore[union-attr]


def test_apply_blocks_a_video_that_changed_since_it_was_read(store: Store) -> None:
    session = stage_swap(store)
    pull = store.begin_pull("UC_fake", NOW)
    store.upsert_video(make_video(1, title="Cactus basics", description=f"Edited in Studio.\n{OLD}"),
                       pull, NOW)
    store.finish_pull(pull, 7, NOW)
    result = bulkagent.apply(store, session, [vid(1), vid(2)], NOW)
    assert (result.drafted, result.blocked) == (1, 1)
    blocked = next(i for i in result.items if i.status == "blocked")
    assert blocked.video_id == vid(1) and blocked.message == bulkagent.BLOCKED_CHANGED
    assert store.get_draft(vid(1)) is None


def test_apply_blocks_a_video_that_is_gone_and_names_unstaged_ids(store: Store) -> None:
    session = stage_swap(store)
    session.put(replace(session.get(vid(1)), video_id="vid_gone"))  # type: ignore[type-var]
    result = bulkagent.apply(store, session, ["vid_gone", vid(5), vid(1)], NOW)
    assert [i.status for i in result.items] == ["blocked", "not_in_plan", "drafted"]
    assert result.items[0].message == bulkagent.BLOCKED_GONE


def test_apply_runs_the_guards_again(store: Store) -> None:
    session = stage_swap(store)
    one = session.get(vid(1))
    assert one is not None
    session.put(replace(one, description=one.description + "\nBuy at https://shop.example/invented"))
    session.put(replace(session.get(vid(2)), title="<b>"))  # type: ignore[type-var]
    result = bulkagent.apply(store, session, [vid(1), vid(2), vid(6)], NOW)
    assert [i.status for i in result.items] == ["blocked", "blocked", "drafted"]
    assert "shop.example/invented" in result.items[0].message and "< or >" in result.items[1].message
    assert set(d.video_id for d in store.list_drafts()) == {vid(6)}


def test_apply_refuses_more_than_max_bulk(store: Store) -> None:
    session = stage_swap(store)
    with pytest.raises(BulkAgentError, match="at most 500"):
        bulkagent.apply(store, session, [f"vid{n}" for n in range(bulk.MAX_BULK + 1)])
    assert store.list_drafts() == []
    # duplicates do not count twice
    result = bulkagent.apply(store, session, [vid(1)] * 600, NOW)
    assert result.drafted == 1


def test_apply_with_nothing_ticked_makes_nothing(store: Store) -> None:
    session = stage_swap(store)
    result = bulkagent.apply(store, session, [], NOW)
    assert result.drafted == 0 and store.list_drafts() == [] and len(session.staged_list()) == 3


def test_the_note_falls_back_to_the_request_when_there_is_no_reason(store: Store) -> None:
    session, _ = go(store, [tool_reply(("edit_videos", {"edits": [edit(store, 3, title="Short")]})),
                            answer("ok")], text="Shorten the long title")
    session.put(replace(session.staged_list()[0], reason=""))
    bulkagent.apply(store, session, [vid(3)], NOW)
    assert store.get_draft(vid(3)).note == "Bulk agent: Shorten the long title"  # type: ignore[union-attr]


def test_review_lists_fields_and_warns_only_about_links_the_creator_did_not_name(store: Store) -> None:
    session = stage_swap(store)
    reviewed = bulkagent.check_staged(store, session)
    assert {r.staged.video_id for r in reviewed} == {vid(1), vid(2), vid(6)}
    assert all(r.staged.fields == ("description",) for r in reviewed)
    assert all(r.warnings == () for r in reviewed)  # the old link is in the creator's words: asked for
    other, _ = go(store, [tool_reply(("apply_steps", {
        "steps": [{"kind": "remove_line", "contains": "Join"}], "refs": [ref(store, 1)]})), answer("ok")],
        text="Remove the Join lines")
    (only,) = bulkagent.check_staged(store, other)
    assert any(OLD in w and "missing now" in w for w in only.warnings)


# --- reasons and long sessions -------------------------------------------------------------------------
def test_a_reason_with_an_invented_number_is_dropped_and_the_model_is_told(store: Store) -> None:
    session, client = go(store, [
        tool_reply(("edit_videos", {"edits": [edit(store, 3, title="Cactus basics",
                                                   reason="Cut the title to 55 characters.")]}),
                   ("edit_videos", {"edits": [edit(store, 5, title="Soil mix, better", reason="Clearer.")]}),
                   ("apply_steps", {"steps": [{"kind": "add_tags", "tags": ["q"]}], "refs": [ref(store, 1)],
                                    "reason": "Added a tag to 77 videos."})),
        answer("Done."),
    ])
    first, second, third = tool_results(client, 1)
    assert "dropped" in first["results"][0]["note"] and "55" in first["results"][0]["note"]
    assert "note" not in second["results"][0] and "dropped" in third["note"]
    assert session.get(vid(3)).reason == "" and session.get(vid(5)).reason == "Clearer."  # type: ignore[union-attr]
    assert session.get(vid(1)).reason == "Added tags: q."  # type: ignore[union-attr]  # written by code


def test_old_big_tool_results_are_shortened_so_a_long_session_fits(store: Store) -> None:
    big = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
           {"role": "tool", "tool_call_id": "a", "content": "x" * (bulkagent.STUB_OVER + 1)},
           {"role": "tool", "tool_call_id": "b", "content": "short"}]
    big += [{"role": "assistant", "content": "ok"}] * bulkagent.KEEP_MESSAGES
    newest = {"role": "tool", "tool_call_id": "c", "content": "y" * (bulkagent.STUB_OVER + 1)}
    big.append(newest)
    bulkagent._compact(big)
    assert "removed to save space" in big[2]["content"] and big[3]["content"] == "short"
    assert big[-1]["content"] == "y" * (bulkagent.STUB_OVER + 1)  # the newest stay whole
