"""Category check: which YouTube category fits one video, judged by Muse Spark from its title, description
and tags. The model only names a category from YouTube's list and gives one short reason; code checks the
id, cleans the reason and stores the verdict (a failed check too). The category itself is changed by the
creator in YouTube Studio; nothing here talks to YouTube."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from .agent import ModelClient, ModelError
from .packaging import _BadAnswer, _format, _object, _parse_object, _Tally, clean_line
from .seo import CATEGORY_NAMES, video_category_id
from .store import CategoryCheck, Store, Video

DESCRIPTION_CHARS = 2000
REASON_MAX = 200

CATEGORY_FORMAT = _format("category_check", _object({
    "category_id": {"type": "string", "enum": list(CATEGORY_NAMES)},
    "reason": {"type": "string", "description": "One short sentence: why this category fits the video."},
}))

SYSTEM_PROMPT = f"""\
You are the category checker in Yezatas, a tool that helps a YouTube creator. You are given the title, \
description and tags of ONE video and the category it is set to now. Name the YouTube category that fits \
what the video is about, from this list (id: name): \
{"; ".join(f"{key}: {name}" for key, name in CATEGORY_NAMES.items())}.

Rules
- Judge what the video is about, not who might watch it.
- When the current category fits as well as any other, answer the current category.
- Give one short reason in plain words, without quoting the text at length.
- Answer only with the JSON object asked for."""


def _request(video: Video, current_id: str) -> str:
    description = video.description[:DESCRIPTION_CHARS]
    current = CATEGORY_NAMES.get(current_id, "not set")
    return (f"Current category: {current_id or '-'} ({current})\n"
            f"Title: {video.title}\n"
            f"Tags: {', '.join(video.tags) or '(none)'}\n"
            f"Description:\n{description}")


def judge_category(store: Store, client: ModelClient, video_id: str, *, now: datetime | None = None,
                   report: Callable[[str], None] | None = None) -> CategoryCheck:
    """Ask Muse Spark which category fits the video and store the verdict (a failure as "failed").
    An unknown video raises ValueError before any model call."""
    video = store.get_video(video_id)
    if video is None:
        raise ValueError("This video is not in your catalog. Pull your videos first.")
    say = report or (lambda line: None)
    current_id = video_category_id(video)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _request(video, current_id)}]
    tally = _Tally()
    status, error, suggested, reason = "done", "", "", ""
    say("Asking Muse Spark")
    try:
        data = _parse_object(tally.complete(client, messages, CATEGORY_FORMAT).content)
        suggested = data.get("category_id") if isinstance(data.get("category_id"), str) else ""
        if suggested not in CATEGORY_NAMES:
            raise _BadAnswer("The model named a category YouTube does not have. Try again.")
        reason = clean_line(data.get("reason"), REASON_MAX)
    except (ModelError, _BadAnswer) as exc:
        status, error, suggested, reason = "failed", str(exc), "", ""
    record = CategoryCheck(video_id=video_id, checked_at="", model=client.model, status=status,
                           current_id=current_id, suggested_id=suggested, reason=reason, confirmed_at="",
                           error=error, input_tokens=tally.input, output_tokens=tally.output)
    store.save_category_check(record, now)
    stored = store.category_check(video_id)
    assert stored is not None
    return stored
