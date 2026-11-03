"""Fact guard: catches links, times, @handles and emails that a suggested text invents.

A suggestion may rewrite and reorder, but anything that looks like a fact a reader could act on
(a link, a chapter time, a handle, an address, a number) must already be in the current text, or be a link a
tool returned. Pure and deterministic; the suggestion agent runs it before saving a draft.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Collection
from urllib.parse import parse_qs, urlsplit

from .seo import _URL
from .validate import Draft

_TRAILING = ".,;:!?)]}'\""
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_HANDLE = re.compile(r"(?<![\w@.])@([\w][\w.-]{1,29})")
_TIME = re.compile(r"(?<![\w:])(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?![\w:])")
_LIST_MARK = re.compile(r"^[ \t]*\d{1,2}[.)](?=\s)", re.MULTILINE)  # "1." of a numbered list
# A number as written: 49, 4.2, 1,204, 1 204 (also with no-break spaces), in any script's digits.
NUMBER = re.compile(r"(?<![\w#@])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d[\d.,]*\d|\d)")
RATIO = re.compile(r"\w*\d+(?::\d+)+\w*")  # 16:9, a12:30, 5:61: left after the times are taken out
_YOUTUBE_HOSTS = ("youtube.com", "m.youtube.com", "music.youtube.com")


@dataclass(frozen=True)
class FactProblem:
    field: str  # "title" | "description"
    kind: str  # "new_link" | "new_timestamp" | "new_handle" | "new_email" | "new_number"
    value: str  # the offending text
    message: str  # one sentence for the model and the creator


def number_key(number: str) -> str:
    """The value of a number as written, in one spelling: 1,204 = 1.204 = 1 204 = 1204 and
    4,2 = 4.2, but 4.2 is not 42 and 1,204 is not 12.04. Digits of other scripts become 0-9."""
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in number)
    text = re.sub(r"[ \u00a0\u202f]", "", text)
    parts = re.split(r"[.,]", text)
    head, tail = parts[:-1], parts[-1]
    if not head:
        return tail.lstrip("0") or "0"
    seps = re.findall(r"[.,]", text)
    if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3 and parts[0] != "0":
        return "".join(parts).lstrip("0") or "0"
    if len(set(seps)) == 2 and seps[:-1] == [seps[0]] * (len(seps) - 1):
        whole = "".join(head)  # 1,204.5 or 1.204,5
    elif len(parts) == 2:
        whole = parts[0]
    else:
        return "".join(parts)  # 26.10.2026: not one value, compare it whole
    fraction = tail.rstrip("0")
    whole = whole.lstrip("0") or "0"
    return f"{whole}.{fraction}" if fraction else whole


def extract_urls(text: str) -> list[str]:
    """URLs in order, without sentence punctuation stuck to the end."""
    return [url.rstrip(_TRAILING) for url in _URL.findall(text)]


def _url_key(url: str) -> str:
    """Comparison key: no scheme, no `www.`, lowercase host; one key per YouTube video id."""
    try:
        parts = urlsplit(url if "://" in url else "//" + url)
        host = (parts.hostname or "").lower()
    except ValueError:  # e.g. a placeholder such as https://[your-link-here]: not a real address
        return url.casefold()
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
        "new_link": {}, "new_email": {}, "new_timestamp": {}, "new_handle": {}, "new_number": {}}
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
    rest = _LIST_MARK.sub(" ", _HANDLE.sub(" ", RATIO.sub(" ", _TIME.sub(" ", rest))))
    for match in NUMBER.finditer(rest):
        found["new_number"].setdefault(number_key(match.group(0)), match.group(0))
    return found


_MESSAGES = {
    "new_link": 'The link "{value}" is not in the current text; do not add links.',
    "new_timestamp": ('The time "{value}" is not in the current text; do not add timestamps '
                      "or chapters."),
    "new_handle": 'The handle "{value}" is not in the current text; do not add @handles.',
    "new_email": 'The email address "{value}" is not in the current text; do not add addresses.',
    "new_number": ('The number "{value}" is not in the current text; do not add prices, counts, '
                   "percentages, years or other numbers."),
}


def numbers_in(text: str) -> set[str]:
    """The numbers of a text, in the fact guard's spelling (for `allowed_numbers`)."""
    return set(_scan(text)["new_number"])


def check_facts(draft: Draft, original: Draft, allowed_urls: Collection[str] = (),
                allowed_numbers: Collection[str] = ()) -> list[FactProblem]:
    """Facts in the draft's title or description that the original does not have.

    `allowed_urls` are links the creator owns (returned by a tool); they never count as new.
    `allowed_numbers` (from `numbers_in`) are numbers from a real source, such as the transcript.
    """
    known = _scan(original.title + "\n" + original.description)
    for url in allowed_urls:
        known["new_link"].setdefault(_url_key(url.rstrip(_TRAILING)), url)
    known["new_number"].update(dict.fromkeys(allowed_numbers, ""))
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
