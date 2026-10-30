"""Fake YouTube reader and writer with made-up data. No network."""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Sequence

import httplib2
from googleapiclient.errors import HttpError

CHANNEL_ID = "UC_fake_channel"
UPLOADS_ID = "UU_fake_uploads"
CHANNEL_TITLE = "Sample Cactus Garden"


def make_item(n: int, views: int | None = None) -> dict[str, Any]:
    """A videos.list item. Every 7th video hides likes, every 5th has a German localization."""
    stats: dict[str, str] = {"viewCount": str(views if views is not None else n * 10),
                             "commentCount": str(n % 4)}
    if n % 7 != 0:
        stats["likeCount"] = str(n)
    item: dict[str, Any] = {
        "id": f"vid{n:06d}",
        "snippet": {
            "channelId": CHANNEL_ID,
            "title": f"Repotting cactus #{n}",
            "description": f"How to repot cactus number {n}.",
            "tags": ["cactus", "garden"],
            "publishedAt": "2026-01-01T10:00:00Z",
            "defaultLanguage": "en",
            "thumbnails": {"default": {"url": f"https://example.invalid/{n}.jpg"}},
        },
        "contentDetails": {"duration": "PT5M30S"},
        "statistics": stats,
        "status": {"privacyStatus": "public"},
    }
    if n % 5 == 0:
        item["localizations"] = {"de": {"title": f"Kaktus umtopfen #{n}",
                                        "description": f"Anleitung {n}."}}
    return item


class FakeYouTubeReader:
    def __init__(self, video_count: int = 120, page_size: int = 50, has_channel: bool = True,
                 error: Exception | None = None) -> None:
        self.error = error  # raised by channels_mine when set
        self.items = [make_item(n) for n in range(1, video_count + 1)]
        self.page_size = page_size
        self.has_channel = has_channel
        self.calls: list[tuple[str, Any]] = []

    def channels_mine(self, part: str) -> dict[str, Any]:
        self.calls.append(("channels_mine", part))
        if self.error is not None:
            raise self.error
        if not self.has_channel:
            return {"items": []}
        channel: dict[str, Any] = {
            "id": CHANNEL_ID,
            "contentDetails": {"relatedPlaylists": {"uploads": UPLOADS_ID}},
        }
        parts = set(part.split(","))
        if "snippet" in parts:
            channel["snippet"] = {"title": CHANNEL_TITLE}
        if "statistics" in parts:
            channel["statistics"] = {"videoCount": str(len(self.items)), "viewCount": "12345",
                                     "subscriberCount": "321"}
        return {"items": [channel]}

    def playlist_items(self, playlist_id: str, page_token: str | None) -> dict[str, Any]:
        self.calls.append(("playlist_items", (playlist_id, page_token)))
        assert playlist_id == UPLOADS_ID
        start = int(page_token) if page_token else 0
        chunk = self.items[start:start + self.page_size]
        response: dict[str, Any] = {
            "items": [{"contentDetails": {"videoId": i["id"]}} for i in chunk]}
        if start + self.page_size < len(self.items):
            response["nextPageToken"] = str(start + self.page_size)
        return response

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]:
        self.calls.append(("videos", (tuple(ids), part)))
        assert 1 <= len(ids) <= 50
        wanted = set(ids)
        return {"items": [i for i in self.items if i["id"] in wanted]}


def make_http_error(status: int, reason: str, message: str = "Something went wrong.") -> HttpError:
    body = {"error": {"code": status, "message": message,
                      "errors": [{"reason": reason, "message": message}]}}
    return HttpError(httplib2.Response({"status": str(status)}), json.dumps(body).encode())


class FakeYouTubeWriter(FakeYouTubeReader):
    """A fake channel that also accepts videos.update. Records every call; nothing leaves the process.

    Like YouTube, an update replaces the writable snippet: fields left out of the body are deleted.
    The part decides what is read and written: `localizations` is the complete translations map,
    replaced as a whole by an update that asks for that part.
    """

    WRITABLE = ("title", "description", "tags", "categoryId", "defaultLanguage",
                "defaultAudioLanguage")

    def __init__(self, video_count: int = 5) -> None:
        super().__init__(video_count)
        for item in self.items:
            item["snippet"]["categoryId"] = "26"
            item["snippet"]["defaultAudioLanguage"] = "en"
        self.update_errors: dict[str, Exception] = {}  # video id -> raised by update_video
        self.on_update: Callable[[dict[str, Any]], None] | None = None  # called before applying
        self.updates: list[tuple[dict[str, Any], str]] = []

    def item(self, video_id: str) -> dict[str, Any]:
        return next(i for i in self.items if i["id"] == video_id)

    def studio_edit(self, video_id: str, **snippet_fields: Any) -> None:
        """Change the live text, as if the creator edited the video in Studio."""
        self.item(video_id)["snippet"].update(snippet_fields)

    def studio_localizations(self, video_id: str, localizations: dict[str, Any]) -> None:
        """Replace the live translations, as if the creator edited them in Studio."""
        self.item(video_id)["localizations"] = copy.deepcopy(localizations)

    def remove(self, video_id: str) -> None:
        self.items = [i for i in self.items if i["id"] != video_id]

    def videos(self, ids: Sequence[str], part: str) -> dict[str, Any]:
        self.calls.append(("videos", (tuple(ids), part)))
        assert 1 <= len(ids) <= 50
        wanted = set(ids)
        items = [copy.deepcopy(i) for i in self.items if i["id"] in wanted]
        if "localizations" not in part.split(","):
            for item in items:
                item.pop("localizations", None)
        return {"items": items}

    def update_video(self, body: dict[str, Any], part: str) -> dict[str, Any]:
        self.calls.append(("update_video", (body["id"], part)))
        self.updates.append((copy.deepcopy(body), part))
        if self.on_update is not None:
            self.on_update(body)
        if body["id"] in self.update_errors:
            raise self.update_errors[body["id"]]
        item = self.item(body["id"])
        parts = part.split(",")
        response: dict[str, Any] = {"id": body["id"]}
        if "snippet" in parts:
            old = item["snippet"]
            new = {k: v for k, v in old.items() if k not in self.WRITABLE}
            new.update(copy.deepcopy(body["snippet"]))
            item["snippet"] = new
            response["snippet"] = copy.deepcopy(new)
        if "localizations" in parts:
            if body.get("localizations"):
                item["localizations"] = copy.deepcopy(body["localizations"])
            else:
                item.pop("localizations", None)
            response["localizations"] = copy.deepcopy(item.get("localizations", {}))
        return response


class FakeModelClient:
    """A scripted model: no network. Each script item is a ModelReply, an exception to raise, or a
    callable(messages, tools, response_format) returning either. Records every call it got."""

    def __init__(self, script: Sequence[Any], model: str = "muse-spark-1.3") -> None:
        self.model = model
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []  # messages are copied: the agent keeps appending

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 response_format: dict[str, Any] | None = None) -> Any:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools,
                           "response_format": response_format})
        assert self.script, "the fake model ran out of scripted replies"
        item = self.script.pop(0)
        if callable(item):
            item = item(messages, tools, response_format)
        if isinstance(item, Exception):
            raise item
        return item


# --- translation fakes (no network) ----------------------------------------------------------------
class ScriptedTranslator:
    """Translates by prefixing: the title gets "[es] ", each description line too, so links, times
    and tags survive. `outputs[lang]` overrides one language with a Translated, an exception to raise
    or a callable(source_language, target, title, description)."""

    def __init__(self, outputs: dict[str, Any] | None = None, model: str = "spark-open-test") -> None:
        self.model = model
        self.outputs = dict(outputs or {})
        self.calls: list[tuple[str, str, str, str]] = []

    def translate(self, source_language: str, target_language: str, title: str, description: str) -> Any:
        from yezatas.translate import Translated

        self.calls.append((source_language, target_language, title, description))
        item = self.outputs.get(target_language)
        if callable(item):
            item = item(source_language, target_language, title, description)
        if isinstance(item, Exception):
            raise item
        if item is not None:
            return item
        prefix = f"[{target_language}] "
        body = "\n".join(prefix + line if line else line for line in description.split("\n"))
        return Translated(prefix + title, body, self.model, 10, 20)


class ScriptedVerifier:
    """Gives every translation `default` (a Verdict) unless `verdicts[lang]` says otherwise (a Verdict
    or an exception to raise). `up=False` makes available() False."""

    def __init__(self, verdicts: dict[str, Any] | None = None, up: bool = True,
                 model: str = "glimmer-test", default: Any = None) -> None:
        self.model = model
        self.verdicts = dict(verdicts or {})
        self.up = up
        self.default = default
        self.calls: list[tuple[str, str, str, str, str, str]] = []

    def available(self) -> bool:
        return self.up

    def verify(self, source_language: str, target_language: str, source_title: str,
               source_description: str, title: str, description: str) -> Any:
        from yezatas.translate import Verdict

        self.calls.append((source_language, target_language, source_title, source_description,
                           title, description))
        item = self.verdicts.get(target_language, self.default)
        if isinstance(item, Exception):
            raise item
        if item is not None:
            return item
        return Verdict(5, (), source_title, source_description, self.model)


class FakeOpener:
    """Stands in for a urllib opener. Each script item is a dict (sent as JSON), a str (sent as is),
    an exception to raise, or a callable(request) returning one of those. Records the requests as
    (method, url, json body or None, timeout)."""

    def __init__(self, script: Sequence[Any]) -> None:
        self.script = list(script)
        self.requests: list[tuple[str, str, Any, Any]] = []

    def open(self, request: Any, timeout: Any = None) -> Any:
        import io

        if isinstance(request, str):
            method, url, data = "GET", request, None
        else:
            method, url, data = request.get_method(), request.full_url, request.data
        self.requests.append((method, url, json.loads(data) if data else None, timeout))
        assert self.script, "the fake opener ran out of scripted replies"
        item = self.script.pop(0)
        if callable(item):
            item = item(request)
        if isinstance(item, Exception):
            raise item
        raw = item if isinstance(item, str) else json.dumps(item)

        class Response(io.BytesIO):
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *exc: object) -> None:
                self.close()

        return Response(raw.encode("utf-8"))
