"""Fact guard: catches links, times, @handles and emails that a suggested text invents.

A suggestion may rewrite and reorder, but anything that looks like a fact a reader could act on
(a link, a chapter time, a handle, an address) must already be in the current text, or be a link a
tool returned. Pure and deterministic; the suggestion agent runs it before saving a draft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection
from urllib.parse import parse_qs, urlsplit

from .seo import _URL
from .validate import Draft

_TRAILING = ".,;:!?)]}'\""
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE = re.compile(r"(?<![\w@.])@([\w][\w.-]{1,29})")
_TIME = re.compile(r"(?<![\w:])(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?![\w:])")
_YOUTUBE_HOSTS = ("youtube.com", "m.youtube.com", "music.youtube.com")


@dataclass(frozen=True)
class FactProblem:
    field: str  # "title" | "description"
    kind: str  # "new_link" | "new_timestamp" | "new_handle" | "new_email"
    value: str  # the offending text
    message: str  # one sentence for the model and the creator


def extract_urls(text: str) -> list[str]:
    """URLs in order, without sentence punctuation stuck to the end."""
    return [url.rstrip(_TRAILING) for url in _URL.findall(text)]


def _url_key(url: str) -> str:
    """Comparison key: no scheme, no `www.`, lowercase host; one key per YouTube video id."""
    parts = urlsplit(url if "://" in url else "//" + url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "youtu.be":
        video_id = parts.path.strip("/").split("/")[0]
        if video_id:
            return f"youtube:{video_id}"
    if host in _YOUTUBE_HOSTS and parts.path.rstrip("/") == "/watch":
        ids = parse_qs(parts.query).get("v")
        if ids and ids[0]:
            return f"youtube:{ids[0]}"
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    fragment = f"#{parts.fragment}" if parts.fragment else ""
    return f"{host}{path}{query}{fragment}"


def _seconds(match: re.Match[str]) -> int:
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def _scan(text: str) -> dict[str, dict[str, str]]:
    """Facts found in a text: kind -> {comparison key: the text as written}."""
    found: dict[str, dict[str, str]] = {
        "new_link": {}, "new_email": {}, "new_timestamp": {}, "new_handle": {}}
    for url in extract_urls(text):
        found["new_link"].setdefault(_url_key(url), url)
    rest = _URL.sub(" ", text)
    for email in _EMAIL.findall(rest):
        found["new_email"].setdefault(email.casefold(), email)
    rest = _EMAIL.sub(" ", rest)
    for match in _TIME.finditer(rest):
        found["new_timestamp"].setdefault(str(_seconds(match)), match.group(0))
    for match in _HANDLE.finditer(rest):
        handle = match.group(0).rstrip(".-")
        found["new_handle"].setdefault(handle.casefold(), handle)
    return found


_MESSAGES = {
    "new_link": 'The link "{value}" is not in the current text; do not add links.',
    "new_timestamp": ('The time "{value}" is not in the current text; do not add timestamps '
                      "or chapters."),
    "new_handle": 'The handle "{value}" is not in the current text; do not add @handles.',
    "new_email": 'The email address "{value}" is not in the current text; do not add addresses.',
}


def check_facts(draft: Draft, original: Draft, allowed_urls: Collection[str] = ()) -> list[FactProblem]:
    """Facts in the draft's title or description that the original does not have.

    `allowed_urls` are links the creator owns (returned by a tool); they never count as new.
    """
    known = _scan(original.title + "\n" + original.description)
    for url in allowed_urls:
        known["new_link"].setdefault(_url_key(url.rstrip(_TRAILING)), url)
    problems: list[FactProblem] = []
    reported: set[tuple[str, str]] = set()
    for field_name, text in (("title", draft.title), ("description", draft.description)):
        for kind, items in _scan(text).items():
            for key, value in items.items():
                if key in known[kind] or (kind, key) in reported:
                    continue
                reported.add((kind, key))
                problems.append(FactProblem(field_name, kind, value,
                                            _MESSAGES[kind].format(value=value)))
    return problems
