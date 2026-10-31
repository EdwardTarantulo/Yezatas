"""The suggestion agent: Muse Spark reads a video's scorecard through tools and proposes a draft.

The model only proposes. Every suggestion passes the edit validator and the fact guard
(`facts.py`) before it becomes a draft, and a draft still needs the creator's approval to reach
YouTube. Each run is recorded in the store's `agent_runs` table: tool calls, result and tokens.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol

from . import edits, seo
from .config import Config, ensure_dirs
from .facts import check_facts, extract_urls
from .store import Store, Video
from .validate import Draft, validate_draft

MODEL_BASE_URL = "https://api.meta.ai/v1"
KEY_ENV = "MODEL_API_KEY"

# USD per 1M tokens: (input, cached input, output). Reasoning tokens are billed as output.
PRICES: dict[str, tuple[float, float, float]] = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}


class AgentError(ValueError):
    """The agent cannot start for this video (no such video, or it already has a draft)."""


class ModelError(RuntimeError):
    """The model API failed. `kind` is "auth", "rate_limit", "network" or "api"."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text from the model


@dataclass(frozen=True)
class ModelReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    message: dict[str, Any]  # the assistant message, ready to append to `messages`
    input_tokens: int = 0  # includes cached_tokens
    cached_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens


class ModelClient(Protocol):
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelReply: ...


@dataclass(frozen=True)
class Change:
    field: str  # "title" | "description" | "tags"
    check_id: str  # the scorecard check it fixes ("" if none)
    reason: str


@dataclass(frozen=True)
class NotFixed:
    check_id: str
    reason: str  # why the agent left it (usually: it would need facts it does not have)


@dataclass(frozen=True)
class Suggestion:
    title: str
    description: str
    tags: tuple[str, ...]
    summary: str
    changes: tuple[Change, ...]
    not_fixed: tuple[NotFixed, ...]


@dataclass(frozen=True)
class SuggestResult:
    run_id: int
    video_id: str
    status: str  # "drafted" | "no_change" | "rejected" | "failed"
    message: str  # one or two sentences for the UI
    suggestion: Suggestion | None = None


def estimate_cost(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float | None:
    """USD cost of a run from its token counts, or None for a model without a known price."""
    price = PRICES.get(model)
    if price is None:
        return None
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * price[0] + cached_tokens * price[1] + output_tokens * price[2]) / 1_000_000


# --- the API key ---------------------------------------------------------------------------------
def key_source(config: Config, env: Mapping[str, str] | None = None) -> str:
    """Where the key comes from: "environment", "saved" or "" (none)."""
    if env is None:
        env = os.environ
    if env.get(KEY_ENV, "").strip():
        return "environment"
    if config.model_key_path is not None and config.model_key_path.is_file():
        return "saved"
    return ""


def load_api_key(config: Config, env: Mapping[str, str] | None = None) -> str | None:
    """The Meta API key: MODEL_API_KEY from the environment, else the saved key file."""
    if env is None:
        env = os.environ
    key = env.get(KEY_ENV, "").strip()
    if key:
        return key
    if config.model_key_path is None:
        return None
    try:
        key = config.model_key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def save_api_key(config: Config, key: str) -> None:
    """Save the key atomically next to the OAuth token (same folder, same protection)."""
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("The key must be one line with no spaces.")
    if config.model_key_path is None:
        raise ValueError("This configuration has no place for the key.")
    ensure_dirs(config)
    fd, tmp_name = tempfile.mkstemp(dir=config.credentials_dir, prefix="model_api_key.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
        if sys.platform != "win32":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, config.model_key_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def delete_api_key(config: Config) -> bool:
    """Delete the saved key. Returns True if there was one."""
    if config.model_key_path is None:
        return False
    try:
        config.model_key_path.unlink()
    except FileNotFoundError:
        return False
    return True


# --- the suggestion run --------------------------------------------------------------------------
_CHANGE_FIELDS = ["title", "description", "tags"]


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


SUGGESTION_SCHEMA: dict[str, Any] = _object({
    "title": {"type": "string"},
    "description": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "summary": {"type": "string", "description": "One sentence: what changed and why."},
    "changes": {"type": "array", "items": _object({
        "field": {"type": "string", "enum": _CHANGE_FIELDS},
        "check_id": {"type": "string", "description": "The scorecard check this fixes."},
        "reason": {"type": "string"},
    })},
    "not_fixed": {"type": "array", "items": _object({
        "check_id": {"type": "string"},
        "reason": {"type": "string", "description": "Why the check was left alone."},
    })},
})

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "suggestion", "strict": True, "schema": SUGGESTION_SCHEMA},
}


def _tool(name: str, description: str, properties: dict[str, Any] | None = None,
          required: tuple[str, ...] = ()) -> dict[str, Any]:
    parameters = {"type": "object", "properties": properties or {}, "required": list(required)}
    return {"type": "function",
            "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS: list[dict[str, Any]] = [
    _tool("get_current_metadata",
          "The video's current title, description and tags, with duration, whether it is a Short, "
          "publish date, views, likes, comments, privacy status and languages."),
    _tool("get_scorecard",
          "The video's SEO scorecard: the score and every applicable check with its points, "
          "what was found and how to fix it. Failed checks come first, biggest loss first."),
    _tool("find_related_videos",
          "Public videos on this channel that share words with a query, with their links. "
          "Use it to link a related video. At most 5 results.",
          {"query": {"type": "string", "description": "Words describing the topic."},
           "limit": {"type": "integer", "description": "How many videos, 1 to 5 (default 5)."}},
          ("query",)),
    _tool("get_channel_profile",
          "How the channel usually writes: its most used tags, how many videos have chapters, and "
          "the description lines that repeat across many videos (the creator's closing lines)."),
    _tool("get_transcript",
          "The video's transcript (each line is a time and the words spoken then), if there is "
          "one."),
]

SYSTEM_PROMPT = """\
You are the suggestion assistant in Yezatas, a tool that helps a YouTube creator improve the \
search text of their videos. You work on ONE video at a time and improve its title, description \
and tags so that viewers and YouTube search can find it.

How to work
- Before you answer, always call get_scorecard and get_current_metadata. The scorecard lists \
checks the video fails. Work through the failed checks, biggest point loss first, and fix those \
that can be fixed with the text you have.
- You may also call find_related_videos (to link another video of the channel), \
get_channel_profile (to match the channel's habits) and get_transcript.
- Write in the language and voice of the current text. Keep the creator's style; do not make it \
louder, longer or more clickbait than it is.

Rules you must never break
- Never invent facts. Do not add links unless find_related_videos or the current text gave them. \
Never write timestamps or chapters yourself, even when get_transcript returns a transcript: the \
Make chapters button adds them with real times. A transcript only tells you what the video \
really says. Do not add new @handles, email addresses, prices, specifications, dates or claims \
that the current text does not support.
- Keep every existing link, chapter line, sponsor or affiliate line and the creator's closing \
lines exactly as they are, unless a check says to change them.
- Stay within YouTube's limits: the title at most 100 characters (aim for 30 to 60); the \
description at most 5,000 bytes; all tags together at most 500 characters; no < or > characters \
anywhere. A tag has no commas.
- Change only what a failed check or a clear search benefit asks for. If the text is already \
good, return it unchanged.
- When a failed check needs facts you do not have (chapters need a transcript, a link needs a \
real URL, a language needs the creator), leave it alone and list it in not_fixed with the reason.

Answer
Your final answer is JSON with the complete new title, description and tags (not a diff), a \
one-sentence summary of what changed and why, the changes (each with the field, the check it \
fixes and the reason) and the not_fixed list.
"""

MAX_RELATED = 5
COMMON_LINE_SHARE = 0.3  # a closing line appears in at least this share of public videos ...
COMMON_LINE_MIN = 2  # ... and in at least this many
NO_TRANSCRIPT = {
    "available": False,
    "reason": "There is no transcript for this video. Do not add chapters or timestamps, and do "
              "not make claims about what the video shows beyond the current text.",
}

TRANSCRIPT_CHAR_LIMIT = 20_000
TRANSCRIPT_NOTE = (
    "You may use this transcript to describe what the video really says. Do not add chapters or "
    "timestamps to the text: the Make chapters button does that, with real times."
)

_SEARCH_WORD = re.compile(r"[^\W\d_]{3,}")


class _ToolError(ValueError):
    """A tool call that cannot be answered; the message goes back to the model."""


def _search_words(text: str) -> set[str]:
    return set(_SEARCH_WORD.findall(text.casefold()))


class _Toolbox:
    """The tools of one run. Remembers the links its tools returned: the fact guard allows them."""

    def __init__(self, store: Store, video: Video) -> None:
        self._store = store
        self._video = video
        self.allowed_urls: set[str] = set()
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "get_current_metadata": self._metadata,
            "get_scorecard": self._scorecard,
            "find_related_videos": self._related,
            "get_channel_profile": self._profile,
            "get_transcript": self._transcript,
        }

    def run(self, name: str, raw_arguments: str) -> tuple[Any, str]:
        """(arguments for the log, JSON result text). Never raises for a bad call."""
        logged: Any = raw_arguments
        try:
            args: dict[str, Any] = {}
            if raw_arguments.strip():
                try:
                    parsed = json.loads(raw_arguments)
                except ValueError:
                    raise _ToolError("The arguments are not valid JSON.") from None
                if not isinstance(parsed, dict):
                    raise _ToolError("The arguments must be a JSON object.")
                args = logged = parsed
            else:
                logged = {}
            handler = self._handlers.get(name)
            if handler is None:
                raise _ToolError(f"Unknown tool \"{name}\". Available tools: "
                                 f"{', '.join(self._handlers)}.")
            result = handler(args)
        except _ToolError as exc:
            result = {"error": str(exc)}
        return logged, json.dumps(result, ensure_ascii=False)

    def _transcript(self, args: dict[str, Any]) -> dict[str, Any]:
        """The latest transcript as `m:ss text` lines (about 20,000 characters at most)."""
        from .transcribe import clock  # imported here: transcribe imports this module

        record = self._store.latest_transcript(self._video.video_id)
        if record is None:
            return dict(NO_TRANSCRIPT)
        lines: list[str] = []
        used = 0
        truncated = False
        for segment in record.segments:
            text = " ".join(str(segment.get("text") or "").split())
            if not text:
                continue
            line = f"{clock(int(segment['start_ms']))} {text}"
            if used + len(line) > TRANSCRIPT_CHAR_LIMIT:
                truncated = True
                break
            lines.append(line)
            used += len(line)
        data: dict[str, Any] = {"available": True, "language": record.language,
                                "segments": lines, "note": TRANSCRIPT_NOTE}
        if truncated:
            data["truncated"] = True
        return data

    def _public_videos(self) -> list[Video]:
        return [v for v in self._store.list_videos() if v.privacy_status == "public"]

    def _metadata(self, args: dict[str, Any]) -> dict[str, Any]:
        v = self._video
        return {
            "title": v.title, "description": v.description, "tags": list(v.tags),
            "duration_seconds": seo.parse_duration(v.duration), "is_short": seo.is_short(v),
            "published_at": v.published_at, "views": v.view_count, "likes": v.like_count,
            "comments": v.comment_count, "privacy_status": v.privacy_status,
            "default_language": v.default_language,
            "default_audio_language": v.default_audio_language,
        }

    def _scorecard(self, args: dict[str, Any]) -> dict[str, Any]:
        card = seo.score_video(self._video, seo.channel_context(self._store.list_videos()))
        ordered = card.failed + [c for c in card.checks if c.applicable and c.passed]
        return {"score": card.score, "checks": [
            {"id": c.check_id, "title": c.title, "points": c.points, "max_points": c.max_points,
             "passed": c.passed, "detail": c.detail, "fix": c.fix} for c in ordered]}

    def _related(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise _ToolError("\"query\" is required: words describing the topic.")
        limit = args.get("limit", MAX_RELATED)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise _ToolError("\"limit\" must be a whole number.")
        limit = max(1, min(limit, MAX_RELATED))
        words = _search_words(query)
        ranked: list[tuple[int, Video]] = []
        for v in self._public_videos():
            if v.video_id == self._video.video_id:
                continue
            shared = len(words & _search_words(" ".join([v.title, *v.tags])))
            if shared:
                ranked.append((shared, v))
        ranked.sort(key=lambda pair: (-pair[0], -pair[1].view_count, pair[1].video_id))
        videos = []
        for _, v in ranked[:limit]:
            url = f"https://www.youtube.com/watch?v={v.video_id}"
            self.allowed_urls.add(url)
            videos.append({"video_id": v.video_id, "title": v.title, "url": url,
                           "views": v.view_count})
        return {"videos": videos}

    def _profile(self, args: dict[str, Any]) -> dict[str, Any]:
        public = self._public_videos()
        tag_counts: dict[str, int] = {}
        tag_names: dict[str, str] = {}
        line_counts: dict[str, int] = {}
        with_chapters = 0
        for v in public:
            for tag in v.tags:
                if tag.strip():
                    tag_names.setdefault(tag.casefold(), tag)
            for key in {t.casefold() for t in v.tags if t.strip()}:
                tag_counts[key] = tag_counts.get(key, 0) + 1
            stamps = seo.parse_chapters(v.description)
            if len(stamps) >= seo.CHAPTERS_MIN_COUNT and stamps[0] == 0:
                with_chapters += 1
            for line in {ln.strip() for ln in v.description.splitlines() if ln.strip()}:
                line_counts[line] = line_counts.get(line, 0) + 1
        top_tags = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        needed = max(COMMON_LINE_MIN, math.ceil(COMMON_LINE_SHARE * len(public)))
        common = sorted(((ln, n) for ln, n in line_counts.items() if n >= needed),
                        key=lambda kv: (-kv[1], kv[0]))[:10]
        for line, _ in common:
            self.allowed_urls.update(extract_urls(line))
        return {
            "video_count": len(public),
            "top_tags": [{"tag": tag_names[k], "count": n} for k, n in top_tags],
            "share_with_chapters": round(with_chapters / len(public), 2) if public else 0.0,
            "common_closing_lines": [{"line": ln, "count": n} for ln, n in common],
        }


_PROGRESS = {
    "get_current_metadata": "Reading the current text",
    "get_scorecard": "Reading the scorecard",
    "find_related_videos": "Looking for related videos",
    "get_channel_profile": "Checking how the channel writes",
    "get_transcript": "Looking for a transcript",
}
_BAD_FORMAT = "The model's answer did not match the expected format."


class _BadAnswer(ValueError):
    """The model's final answer cannot be used; the message is shown to the creator."""


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise _BadAnswer(_BAD_FORMAT)
    return value


def _entries(value: Any, keys: tuple[str, ...]) -> list[dict[str, str]]:
    if value is None:
        return []
    if (not isinstance(value, list)
            or not all(isinstance(e, dict) and all(isinstance(e.get(k), str) for k in keys)
                       for e in value)):
        raise _BadAnswer(_BAD_FORMAT)
    return value


def parse_suggestion(content: str | None) -> Suggestion:
    """The model's JSON answer as a normalized `Suggestion`; raises `_BadAnswer` when unusable."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError:
        # Some answers wrap the object in a sentence; take the outermost {...} if that parses.
        start, end = text.find("{"), text.rfind("}")
        try:
            data = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except ValueError:
            data = None
        if data is None:
            raise _BadAnswer("The model's answer was not valid JSON.") from None
    if (not isinstance(data, dict) or not isinstance(data.get("title"), str)
            or not isinstance(data.get("description"), str)):
        raise _BadAnswer(_BAD_FORMAT)
    tags = [t.strip() for t in _text_list(data.get("tags"))]
    changes = _entries(data.get("changes"), ("field", "check_id", "reason"))
    if any(c["field"] not in _CHANGE_FIELDS for c in changes):
        raise _BadAnswer(_BAD_FORMAT)
    not_fixed = _entries(data.get("not_fixed"), ("check_id", "reason"))
    summary = data.get("summary")
    return Suggestion(
        title=data["title"].strip(),
        description=data["description"].replace("\r\n", "\n").replace("\r", "\n"),
        tags=tuple(t for t in tags if t),
        summary=" ".join(summary.split()) if isinstance(summary, str) else "",
        changes=tuple(Change(c["field"], c["check_id"], c["reason"]) for c in changes),
        not_fixed=tuple(NotFixed(n["check_id"], n["reason"]) for n in not_fixed),
    )


def _suggestion_dict(s: Suggestion) -> dict[str, Any]:
    return {"title": s.title, "description": s.description, "tags": list(s.tags),
            "summary": s.summary, "changes": [asdict(c) for c in s.changes],
            "not_fixed": [asdict(n) for n in s.not_fixed]}


def _problems(video: Video, suggestion: Suggestion, allowed_urls: set[str]) -> list[str]:
    original = Draft(video.title, video.description, tuple(video.tags))
    new = Draft(suggestion.title, suggestion.description, suggestion.tags)
    found = [f"The {i.field}: {i.message}" for i in validate_draft(new, original)
             if i.severity == "error"]
    found += [p.message for p in check_facts(new, original, allowed_urls)]
    return found


@dataclass
class _Tally:
    """Tokens used so far in a run, summed over every model call."""

    input: int = 0
    cached: int = 0
    output: int = 0

    def complete(self, client: ModelClient, messages: list[dict[str, Any]],
                 **kwargs: Any) -> ModelReply:
        reply = client.complete(messages, **kwargs)
        self.input += reply.input_tokens
        self.cached += reply.cached_tokens
        self.output += reply.output_tokens
        return reply

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.input, self.cached, self.output)


def _converse(store: Store, client: ModelClient, video: Video, tally: _Tally,
              log: list[dict[str, Any]], say: Callable[[str], None], max_tool_rounds: int,
              now: datetime | None) -> tuple[str, str, Suggestion | None]:
    """The model conversation and its checks: (status, message, suggestion)."""
    toolbox = _Toolbox(store, video)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Please suggest improvements for this video: \"{video.title}\" (id {video.video_id}). "
            "Read its scorecard and current text with the tools first.")},
    ]
    for _ in range(max_tool_rounds):
        reply = tally.complete(client, messages, tools=TOOLS)
        messages.append(reply.message)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            say(_PROGRESS.get(call.name, "Using a tool"))
            arguments, result = toolbox.run(call.name, call.arguments)
            log.append({"type": "tool", "name": call.name, "arguments": arguments,
                        "result_chars": len(result)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    say("Writing the suggestion")
    messages.append({"role": "user", "content": (
        "Now give your final answer as JSON with the complete new title, description and tags, "
        "a one-sentence summary, the changes and the not_fixed list.")})
    reply = tally.complete(client, messages, response_format=RESPONSE_FORMAT)
    try:
        suggestion = parse_suggestion(reply.content)
        problems = _problems(video, suggestion, toolbox.allowed_urls)
    except _BadAnswer as exc:  # one unusable answer gets the same single repair round
        problems = [f"{exc} Reply with only the JSON object that matches the schema."]
    if problems:
        log.append({"type": "problems", "round": 1, "problems": problems})
        say("Fixing problems the checks found")
        messages.append(reply.message)
        messages.append({"role": "user", "content": (
            "The checks found problems with your answer:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\nReply again with the corrected JSON. Remove what is not allowed; do not "
              "replace it with other invented facts.")})
        reply = tally.complete(client, messages, response_format=RESPONSE_FORMAT)
        suggestion = parse_suggestion(reply.content)
        problems = _problems(video, suggestion, toolbox.allowed_urls)
        if problems:
            log.append({"type": "problems", "round": 2, "problems": problems})
            return ("rejected", "Stopped by the checks, so nothing was drafted: "
                    + " ".join(problems), None)
    if (suggestion.title, suggestion.description, suggestion.tags) == (
            video.title, video.description, tuple(video.tags)):
        return "no_change", "The model found nothing worth changing in this video.", suggestion
    edits.save_draft(store, video.video_id, suggestion.title, suggestion.description,
                     suggestion.tags, source="agent",
                     note=f"Suggested by {client.model}: {suggestion.summary}", now=now)
    return "drafted", "Saved as a draft for you to review.", suggestion


def suggest_for_video(store: Store, client: ModelClient, video_id: str, *,
                      now: datetime | None = None, report: Callable[[str], None] | None = None,
                      max_tool_rounds: int = 6) -> SuggestResult:
    """Ask the model for a better title, description and tags and save them as a draft.

    Raises `AgentError` before a run starts. Everything after that is recorded as a run and
    returned as a status; the draft is never approved or pushed.
    """
    video = store.get_video(video_id)
    if video is None:
        raise AgentError("No such video.")
    if store.get_draft(video_id) is not None:
        raise AgentError("This video already has a draft. Push or discard it first.")
    say = report or (lambda line: None)
    run_id = store.begin_agent_run(video_id, client.model, now)
    log: list[dict[str, Any]] = []
    tally = _Tally()
    suggestion: Suggestion | None = None
    try:
        status, message, suggestion = _converse(store, client, video, tally, log, say,
                                                max_tool_rounds, now)
    except ModelError as exc:
        log.append({"type": "error", "kind": exc.kind, "message": str(exc)})
        status, message = "failed", str(exc)
    except _BadAnswer as exc:
        log.append({"type": "error", "kind": "answer", "message": str(exc)})
        status, message = "failed", str(exc)
    except BaseException:
        store.finish_agent_run(run_id, "failed", "The run stopped with an unexpected error.", log,
                               {}, tally.as_tuple(), now)
        raise
    summary = suggestion.summary if suggestion and suggestion.summary else message
    store.finish_agent_run(run_id, status, summary, log,
                           _suggestion_dict(suggestion) if suggestion else {}, tally.as_tuple(),
                           now)
    return SuggestResult(run_id, video_id, status, message, suggestion)
