"""Pinned comment: Muse Spark suggests one short comment for the creator to post under a video and pin.

The model only writes the words. Code drops a suggestion with a link that is not in the description or a
number that is not in the title or description, cuts it to a length and stores it (a failure keeps the
suggestion before it). The creator posts and pins it in YouTube Studio; nothing here talks to YouTube."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from .agent import ModelClient, ModelError
from .comments import _allowed_digits, _first_bad_number, _first_bad_url
from .facts import extract_urls
from .packaging import _BadAnswer, _format, _object, _parse_object, _Tally
from .store import PinnedComment, Store, Video

SUGGESTION_CHARS = 500
DESCRIPTION_CHARS = 3000

PINNED_FORMAT = _format("pinned_comment", _object({
    "text": {"type": "string", "description": "The comment, ready to post."},
}))

SYSTEM_PROMPT = """\
You write the pinned comment for ONE YouTube video in Yezatas, a tool that helps the creator. The creator \
posts it under the video and pins it, so it is the first comment viewers see.

Rules
- One short comment, at most 3 sentences, in the language of the title.
- It adds something: a question viewers can answer, the most useful link from the description, or what to \
watch next.
- Use only links and numbers that are in the title or description. Never invent a link, price, date or count.
- No hashtags, no "like and subscribe", no quotes around the comment.
- Answer only with the JSON object asked for."""


def _request(video: Video, previous: str) -> str:
    parts = [f"Title: {video.title}", f"Description:\n{video.description[:DESCRIPTION_CHARS] or '(none)'}"]
    if previous:
        parts.append(f"Write a different comment from this earlier suggestion:\n{previous}")
    return "\n\n".join(parts)


def suggest_pinned(store: Store, client: ModelClient, video_id: str, *, now: datetime | None = None,
                   report: Callable[[str], None] | None = None) -> PinnedComment:
    """Ask Muse Spark for a pinned comment (different from the stored one) and store it.
    An unknown video raises ValueError before any model call."""
    video = store.get_video(video_id)
    if video is None:
        raise ValueError("This video is not in your catalog. Pull your videos first.")
    say = report or (lambda line: None)
    before = store.pinned_comment(video_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _request(video, before.suggestion if before else "")}]
    tally = _Tally()
    text, error = "", ""
    say("Asking Muse Spark")
    try:
        data = _parse_object(tally.complete(client, messages, PINNED_FORMAT).content)
        raw = data.get("text").strip() if isinstance(data.get("text"), str) else ""
        if not raw:
            raise _BadAnswer("The model's answer was empty. Try again.")
        bad_url = _first_bad_url(raw, set(extract_urls(video.description)))
        if bad_url:
            raise _BadAnswer(f'The suggestion was dropped: it links to "{bad_url}", which is not in the '
                             "description. Try again.")
        bad_number = _first_bad_number(raw, _allowed_digits(video.title, video.description))
        if bad_number:
            raise _BadAnswer(f'The suggestion was dropped: it uses the number "{bad_number}", which is not in '
                             "the title or description. Try again.")
        text = raw[:SUGGESTION_CHARS]
    except (ModelError, _BadAnswer) as exc:
        error = str(exc)
    store.save_pinned_suggestion(video_id, text, client.model, error, tally.input, tally.output, now)
    stored = store.pinned_comment(video_id)
    assert stored is not None
    return stored
