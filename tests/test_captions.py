import json
from types import SimpleNamespace
from typing import Any

import pytest

from yezatas.captions import (
    DOWNLOAD_QUOTA, LIST_QUOTA, MODEL_NAME, QUOTA_MESSAGE, SCOPE_MESSAGE, CaptionChoice,
    CaptionsError, build_record, choose_track, fetch_captions, language_name, parse_srt,
)

V1 = "vidA00001"

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello and welcome.

2
00:00:04,000 --> 00:01:05,250
Today we repot a <i>cactus</i> &amp; a fern.
Second line.
"""


def track(caption_id: str, language: str, kind: str = "standard", **extra: Any) -> dict[str, Any]:
    return {"id": caption_id, "snippet": {"language": language, "trackKind": kind, **extra}}


def http_error(status: int, reason: str = "", message: str = "") -> Exception:
    body = {"error": {"message": message, "errors": [{"reason": reason}] if reason else []}}
    exc = Exception("boom")
    exc.resp = SimpleNamespace(status=status)  # type: ignore[attr-defined]
    exc.content = json.dumps(body).encode()  # type: ignore[attr-defined]
    return exc


class FakeApi:
    def __init__(self, items: list[dict[str, Any]], srt: Any = SRT,
                 list_error: Exception | None = None, download_error: Exception | None = None):
        self.items, self.srt = items, srt
        self.list_error, self.download_error = list_error, download_error
        self.calls: list[tuple[str, str]] = []

    def caption_tracks(self, video_id: str) -> dict[str, Any]:
        self.calls.append(("list", video_id))
        if self.list_error:
            raise self.list_error
        return {"items": self.items}

    def caption_srt(self, caption_id: str) -> Any:
        self.calls.append(("download", caption_id))
        if self.download_error:
            raise self.download_error
        return self.srt


# --- choosing a track ---------------------------------------------------------------------------------
def test_prefers_manual_track_in_the_video_language() -> None:
    listing = {"items": [track("asr-en", "en", "ASR"), track("man-de", "de"), track("man-en", "en")]}
    choice = choose_track(listing, "en")
    assert choice == CaptionChoice("man-en", "en", True, "English · uploaded by you")


def test_any_manual_track_beats_an_automatic_track_in_the_video_language() -> None:
    listing = {"items": [track("asr-en", "en", "ASR"), track("man-de", "de")]}
    choice = choose_track(listing, "en")
    assert choice.caption_id == "man-de"
    assert choice.label == "German · uploaded by you"


def test_automatic_track_is_the_last_choice_and_prefers_the_video_language() -> None:
    listing = {"items": [track("asr-de", "de", "asr"), track("asr-en", "en", "asr")]}
    choice = choose_track(listing, "en-GB")
    assert (choice.caption_id, choice.manual, choice.label) == (
        "asr-en", False, "English · automatic")
    assert choose_track(listing, "").caption_id == "asr-de"  # no language: the API's order


def test_language_match_ignores_region_and_case() -> None:
    listing = {"items": [track("a", "pt"), track("b", "en-US")]}
    assert choose_track(listing, "EN_gb").caption_id == "b"


def test_forced_draft_and_failed_tracks_are_never_chosen() -> None:
    listing = {"items": [track("f", "en", "forced"), track("d", "en", isDraft=True),
                         track("x", "en", status="failed"), track("ok", "en", "ASR")]}
    assert choose_track(listing, "en").caption_id == "ok"


def test_unknown_language_label_uses_the_code() -> None:
    assert choose_track({"items": [track("a", "xx")]}).label == "xx · uploaded by you"
    assert language_name("lt") == "Lithuanian"
    assert language_name("") == "Unknown language"


@pytest.mark.parametrize("listing", [{}, {"items": []}, {"items": [track("f", "en", "forced")]},
                                     {"items": ["junk", {"id": 5}, {"id": "a"}]}])
def test_no_readable_track_is_an_error(listing: dict[str, Any]) -> None:
    with pytest.raises(CaptionsError) as err:
        choose_track(listing, "en")
    assert err.value.kind == "none"
    assert "no captions" in str(err.value)


# --- the SRT ------------------------------------------------------------------------------------------
def test_parse_srt_keeps_the_times_in_the_file() -> None:
    assert parse_srt(SRT) == [
        {"start_ms": 1000, "end_ms": 3500, "text": "Hello and welcome."},
        {"start_ms": 4000, "end_ms": 65250, "text": "Today we repot a cactus & a fern. Second line."},
    ]


def test_parse_srt_handles_bom_crlf_missing_counter_and_hours() -> None:
    text = "﻿00:00:00,000 --> 00:00:02,000\r\nNo counter\r\n\r\n7\r\n100:00:00,500 --> 100:00:01,000\r\nLate\r\n"
    segments = parse_srt(text)
    assert segments[0] == {"start_ms": 0, "end_ms": 2000, "text": "No counter"}
    assert segments[1]["start_ms"] == 360_000_500


def test_parse_srt_skips_cues_without_text_but_keeps_the_others_times() -> None:
    text = "1\n00:00:01,000 --> 00:00:02,000\n<i> </i>\n\n2\n00:00:03,000 --> 00:00:04,000\nHi\n"
    assert parse_srt(text) == [{"start_ms": 3000, "end_ms": 4000, "text": "Hi"}]


@pytest.mark.parametrize("bad", [
    "1\n00:00:01.000 --> 00:00:02,000\nDot\n",  # dot instead of comma
    "1\n00:00:01,000 --> 00:00:02\nNo millis\n",
    "1\n00:00:61,000 --> 00:01:02,000\nBad seconds\n",
    "1\n0:00:01,000 --> 00:00:02,000\nOne-digit hours\n",
    "1\n00:00:01,000 --> abc\nWords\n",
    "1\n00:00:05,000 --> 00:00:02,000\nEnds first\n",
    "1\n00:00:05,000 --> 00:00:06,000\nA\n\n2\n00:00:01,000 --> 00:00:02,000\nB\n",  # out of order
    "1\nHello there\n",  # no time line
    "1\n",
])
def test_parse_srt_rejects_malformed_input(bad: str) -> None:
    with pytest.raises(CaptionsError) as err:
        parse_srt(bad)
    assert err.value.kind == "srt"
    assert "malformed" in str(err.value)


def test_parse_srt_names_the_bad_cue() -> None:
    text = "1\n00:00:01,000 --> 00:00:02,000\nOk\n\n2\n00:00:0x,000 --> 00:00:03,000\nBad\n"
    with pytest.raises(CaptionsError, match="cue 2"):
        parse_srt(text)


@pytest.mark.parametrize("empty", ["", "   \n\n", "﻿"])
def test_parse_srt_rejects_a_file_with_no_cues(empty: str) -> None:
    with pytest.raises(CaptionsError, match="no text"):
        parse_srt(empty)


# --- the record ---------------------------------------------------------------------------------------
def test_build_record_uses_the_caption_times_and_costs_nothing() -> None:
    segments = parse_srt(SRT)
    record = build_record(V1, CaptionChoice("c1", "en-GB", False, "English · automatic"), segments)
    assert record.id == 0 and record.video_id == V1
    assert record.language == "en"
    assert record.model == MODEL_NAME == "YouTube captions"
    assert record.source_name == "YouTube captions (English)"
    assert record.duration_ms == 65250
    assert record.segments == segments and record.problems == []


def test_fetch_captions_lists_then_downloads_the_chosen_track() -> None:
    api = FakeApi([track("asr", "en", "ASR"), track("man", "en")], srt=SRT.encode("utf-8"))
    result = fetch_captions(api, V1, "en")
    assert api.calls == [("list", V1), ("download", "man")]
    assert result.choice.label == "English · uploaded by you"
    assert result.cost == 0.0
    assert result.quota_units == LIST_QUOTA + DOWNLOAD_QUOTA == 250
    assert result.record.segments[0]["text"] == "Hello and welcome."
    assert "2 caption lines" in result.message and "English · uploaded by you" in result.message


def test_fetch_captions_accepts_text_and_works_with_to_srt() -> None:
    from yezatas.subtitles import to_srt

    result = fetch_captions(FakeApi([track("a", "de")]), V1)
    assert "00:00:01,000 --> 00:00:03,500" in to_srt(result.record.segments)


def test_fetch_captions_needs_a_video_id() -> None:
    with pytest.raises(ValueError):
        fetch_captions(FakeApi([]), "  ")


def test_fetch_with_no_tracks_does_not_download() -> None:
    api = FakeApi([])
    with pytest.raises(CaptionsError) as err:
        fetch_captions(api, V1)
    assert err.value.kind == "none"
    assert api.calls == [("list", V1)]


def test_malformed_download_is_an_srt_error() -> None:
    with pytest.raises(CaptionsError) as err:
        fetch_captions(FakeApi([track("a", "en")], srt="WEBVTT\n\n00:01.000 --> 00:02.000\nHi"), V1)
    assert err.value.kind == "srt"


def test_unexpected_download_shape_is_an_api_error() -> None:
    with pytest.raises(CaptionsError) as err:
        fetch_captions(FakeApi([track("a", "en")], srt={"oops": 1}), V1)
    assert err.value.kind == "api"


# --- API errors ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("error, kind, text", [
    (http_error(403, "insufficientPermissions"), "scope", SCOPE_MESSAGE),
    (http_error(401), "scope", SCOPE_MESSAGE),
    (http_error(403, "forbidden"), "scope", SCOPE_MESSAGE),
    (http_error(403, "quotaExceeded"), "quota", QUOTA_MESSAGE),
    (http_error(404, "videoNotFound"), "not_found", "could not find"),
    (http_error(500), "api", "HTTP 500"),
    (ConnectionError("down"), "network", "Could not reach YouTube"),
])
@pytest.mark.parametrize("stage", ["list", "download"])
def test_api_errors_become_plain_english(error: Exception, kind: str, text: str, stage: str) -> None:
    api = FakeApi([track("a", "en")], list_error=error if stage == "list" else None,
                  download_error=error if stage == "download" else None)
    with pytest.raises(CaptionsError) as err:
        fetch_captions(api, V1, "en")
    assert err.value.kind == kind
    assert text in str(err.value)
    assert "boom" not in str(err.value)


def test_scope_message_tells_the_creator_what_to_do() -> None:
    assert SCOPE_MESSAGE == "Sign in again to allow reading captions."
