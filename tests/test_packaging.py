"""Packaging check: boxes, corner maths, fact guard, the three Meta clients (faked), the review and
the variants. Tiny pictures are made with Pillow; no test touches the network."""

import base64
import io
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from PIL import Image

from fakes import FakeModelClient
from yezatas import packaging
from yezatas.agent import ModelError, ModelReply, estimate_cost
from yezatas.config import load_config
from yezatas.edits import save_draft
from yezatas.packaging import (
    Box, MetaImageClient, MetaSamClient, PackagingError, best_box, best_thumbnail_url, fetch_thumbnail,
    guard_overlay, guard_point, make_variants, measure_image, measurement_numbers,
    overlaps_badge_corner, packaging_tools_from_config, parse_sam_boxes, review_packaging,
    share_of_frame, stored_image_path, thumbnail_url_problem, variant_prompt, words_of,
)
from yezatas.store import Store, Video

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
VID = "vid00000001"
URL = "https://i.ytimg.com/vi/vid00000001/maxresdefault.jpg"
TITLE = "Cactus repotting guide"
DESCRIPTION = "How to repot a cactus without hurting its roots."
KEY = "sk-test-secret-key-123"


# --- helpers --------------------------------------------------------------------------------------
def picture(fmt: str = "JPEG", size: tuple[int, int] = (640, 360), color: Any = (200, 200, 200)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, fmt)
    return out.getvalue()


def reply(data: dict[str, Any], tokens: tuple[int, int, int] = (1000, 0, 100)) -> ModelReply:
    return ModelReply(json.dumps(data), (), {}, tokens[0], tokens[1], tokens[2])


def reading(text: str = "REPOT IT", zone: str = "top-left", subject: str = "cactus",
            face: bool = False) -> ModelReply:
    return reply({"text": text, "text_zone": zone, "subject": subject, "face": face})


def critique(points: list[str] | None = None, ideas: list[tuple[str, str]] | None = None) -> ModelReply:
    ideas = [("Zoom in on the cactus", "REPOT IT"), ("Brighter background", "cactus guide")] \
        if ideas is None else ideas
    return reply({"points": points if points is not None else ["The text is easy to read."],
                  "ideas": [{"instruction": i, "overlay_text": t} for i, t in ideas]})


def sam_box(x1: int = 100, y1: int = 50, x2: int = 300, y2: int = 150, w: int = 640, h: int = 360,
            c: float = 0.9) -> Box:
    return Box(x1, y1, x2, y2, w, h, c)


class FakeSegmenter:
    """Returns boxes per phrase (a list, or an exception to raise); records every call."""

    def __init__(self, boxes: dict[str, Any] | None = None, model: str = "sam-test") -> None:
        self.model = model
        self.boxes = boxes if boxes is not None else {}
        self.calls: list[tuple[int, str, str]] = []

    def detect(self, image_bytes: bytes, mime: str, phrase: str) -> list[Box]:
        self.calls.append((len(image_bytes), mime, phrase))
        item = self.boxes.get(phrase, [])
        if isinstance(item, Exception):
            raise item
        return item


class FakeMaker:
    """Each script item is picture bytes or an exception to raise."""

    def __init__(self, script: list[Any], model: str = "image-test") -> None:
        self.model = model
        self.script = list(script)
        self.prompts: list[str] = []

    def edit(self, image_bytes: bytes, mime: str, instruction: str) -> bytes:
        self.prompts.append(instruction)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        pid = s.begin_pull("chan0000001", now=NOW)
        s.upsert_video(Video(
            video_id=VID, channel_id="chan0000001", title=TITLE, description=DESCRIPTION,
            privacy_status="public", thumbnails={"default": {"url": "https://i.ytimg.com/vi/x/default.jpg"},
                                                  "maxres": {"url": URL}}), pid, now=NOW)
        yield s


@pytest.fixture
def images(tmp_path: Path) -> Path:
    return tmp_path / "thumbnails"


def run_review(store: Store, images: Path, vision: Any, segmenter: Any = None, data: bytes | None = None,
               **kw: Any) -> Any:
    seen = data if data is not None else picture()
    return review_packaging(store, vision, segmenter or FakeSegmenter(), VID, images_dir=images,
                            fetch=lambda url: seen, now=NOW, **kw)


# --- SAM boxes ------------------------------------------------------------------------------------
SAM_TEXT = ("<|box;x1=783;y1=64;x2=2047;y2=1046;w=2048;h=1152;c=0.95|>"
            "<|mask;x=0;y=0;c=0.95;data=796,1024,~!!3@B#L47c(Kjk|>"
            "<|box;x1=10;y1=20;x2=110;y2=220;w=2048;h=1152;c=0.5|>")


def test_parse_sam_boxes_reads_several_boxes_and_skips_masks() -> None:
    boxes = parse_sam_boxes(SAM_TEXT)
    assert boxes == [Box(783, 64, 2047, 1046, 2048, 1152, 0.95), Box(10, 20, 110, 220, 2048, 1152, 0.5)]


def test_parse_sam_boxes_empty_and_junk() -> None:
    assert parse_sam_boxes("") == []
    assert parse_sam_boxes("no boxes here <|mask;x=0;y=0;data=1,2,abc|>") == []
    assert parse_sam_boxes("<|box;x1=a;y1=1;x2=2;y2=2;w=10;h=10;c=0.5|>") == []  # not a number
    assert parse_sam_boxes("<|box;x1=1;y1=1;x2=2;y2=2;w=10|>") == []  # no height
    assert parse_sam_boxes("<|box;x1=5;y1=1;x2=2;y2=2;w=10;h=10;c=1|>") == []  # negative area
    assert parse_sam_boxes("<|box;x1=1;y1=1;x2=2;y2=2;w=0;h=10;c=1|>") == []  # no frame


def test_parse_sam_boxes_confidence_is_optional() -> None:
    assert parse_sam_boxes("<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>")[0].c == 0.0


def test_best_box_is_the_most_certain_one() -> None:
    low, high = sam_box(c=0.3), sam_box(x1=5, c=0.8)
    assert best_box([low, high]) is high
    assert best_box([]) is None


def test_share_of_frame_is_a_whole_percent_of_the_frame() -> None:
    assert share_of_frame(Box(0, 0, 640, 360, 640, 360, 1)) == 100
    assert share_of_frame(Box(0, 0, 320, 180, 640, 360, 1)) == 25
    assert share_of_frame(Box(-50, -50, 900, 900, 640, 360, 1)) == 100  # clamped to the frame
    assert share_of_frame(Box(0, 0, 1, 1, 640, 360, 1)) == 0


def test_badge_corner_overlap_maths() -> None:
    # frame 1000 x 1000: the corner is x > 820 and y > 860
    def box(x1: float, y1: float, x2: float, y2: float) -> Box:
        return Box(x1, y1, x2, y2, 1000, 1000, 1)

    assert overlaps_badge_corner(box(900, 900, 990, 990))  # inside
    assert overlaps_badge_corner(box(0, 0, 1000, 1000))  # covers it
    assert overlaps_badge_corner(box(800, 850, 830, 870))  # one corner of the box just inside
    assert not overlaps_badge_corner(box(0, 0, 820, 1000))  # touches the left edge only
    assert not overlaps_badge_corner(box(0, 0, 1000, 860))  # touches the top edge only
    assert not overlaps_badge_corner(box(0, 0, 400, 400))
    assert not overlaps_badge_corner(box(850, 100, 990, 500))  # right side, but higher up


# --- the thumbnail --------------------------------------------------------------------------------
def test_best_thumbnail_url_prefers_the_largest_https() -> None:
    thumbs = {"default": {"url": "https://i.ytimg.com/d.jpg"}, "high": {"url": "https://i.ytimg.com/h.jpg"},
              "medium": {"url": "https://i.ytimg.com/m.jpg"}}
    assert best_thumbnail_url(Video("v", thumbnails=thumbs)) == "https://i.ytimg.com/h.jpg"
    thumbs["maxres"] = {"url": "http://i.ytimg.com/max.jpg"}  # not https: skipped
    assert best_thumbnail_url(Video("v", thumbnails=thumbs)) == "https://i.ytimg.com/h.jpg"
    thumbs["standard"] = {"url": "https://i.ytimg.com/s.jpg"}
    assert best_thumbnail_url(Video("v", thumbnails=thumbs)) == "https://i.ytimg.com/s.jpg"
    assert best_thumbnail_url(Video("v", thumbnails={})) == ""
    assert best_thumbnail_url(Video("v", thumbnails={"high": {"url": 5}})) == ""


@pytest.mark.parametrize("url", [
    "http://i.ytimg.com/vi/x/a.jpg",
    "https://example.invalid/a.jpg",
    "https://i.ytimg.com.example.invalid/a.jpg",
    "https://i.ytimg.com@example.invalid/a.jpg",
    "https://user:pw@i.ytimg.com/a.jpg",
    "https://i.ytimg.com:8443/a.jpg",
    "file:///etc/passwd",
    "",
])
def test_thumbnail_url_problem_refuses_other_hosts(url: str) -> None:
    assert thumbnail_url_problem(url)


def test_thumbnail_url_problem_accepts_the_youtube_image_host() -> None:
    assert thumbnail_url_problem("https://i.ytimg.com/vi/x/maxresdefault.jpg") == ""
    assert thumbnail_url_problem("https://I.YTIMG.COM:443/vi/x/a.jpg") == ""


class FakeHttp:
    """Stands in for an httpx client: `stream(method, url, ...)` replays the scripted response."""

    def __init__(self, status: int = 200, chunks: list[bytes] | None = None, lines: list[str] | None = None,
                 headers: dict[str, str] | None = None, body: Any = None,
                 error: Exception | None = None) -> None:
        self.status, self.chunks, self.lines = status, chunks or [], lines or []
        self.headers, self.body, self.error = headers or {}, body, error
        self.requests: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kw: Any) -> Any:
        self.requests.append({"method": method, "url": url, **kw})
        if self.error is not None:
            raise self.error
        outer = self

        class Response:
            status_code = outer.status
            headers = outer.headers

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def read(self) -> bytes:
                return b""

            def json(self) -> Any:
                if outer.body is None:
                    raise ValueError("no json")
                return outer.body

            def iter_bytes(self) -> Any:
                yield from outer.chunks

            def iter_lines(self) -> Any:
                yield from outer.lines

        return Response()


def test_fetch_thumbnail_downloads_from_the_image_host() -> None:
    http = FakeHttp(chunks=[b"ab", b"cd"])
    assert fetch_thumbnail(URL, http) == b"abcd"
    assert http.requests[0]["method"] == "GET" and http.requests[0]["url"] == URL


def test_fetch_thumbnail_refuses_other_hosts_without_a_request() -> None:
    http = FakeHttp(chunks=[b"x"])
    with pytest.raises(PackagingError, match="only downloads thumbnails from"):
        fetch_thumbnail("https://example.invalid/a.jpg", http)
    assert http.requests == []


def test_fetch_thumbnail_failures_are_plain_sentences() -> None:
    with pytest.raises(PackagingError, match="HTTP 404"):
        fetch_thumbnail(URL, FakeHttp(status=404))
    with pytest.raises(PackagingError, match="too large"):
        fetch_thumbnail(URL, FakeHttp(headers={"content-length": str(10 * 1024 * 1024)}, chunks=[b"x"]))
    big = [b"x" * (1024 * 1024)] * 6
    with pytest.raises(PackagingError, match="too large"):
        fetch_thumbnail(URL, FakeHttp(chunks=big))
    with pytest.raises(PackagingError, match="empty"):
        fetch_thumbnail(URL, FakeHttp(chunks=[]))
    with pytest.raises(PackagingError, match="took too long"):
        fetch_thumbnail(URL, FakeHttp(error=httpx.ReadTimeout("slow")))
    with pytest.raises(PackagingError, match="Could not download"):
        fetch_thumbnail(URL, FakeHttp(error=httpx.ConnectError("down")))


# --- Pillow measurements --------------------------------------------------------------------------
def test_measure_image_white_and_black_halves() -> None:
    white = measure_image(picture(color=(255, 255, 255)))
    assert (white.width, white.height, white.brightness, white.contrast) == (640, 360, 255, 0)
    assert (white.mime, white.extension) == ("image/jpeg", "jpg")
    half = Image.new("RGB", (100, 100), (0, 0, 0))
    half.paste((255, 255, 255), (0, 0, 50, 100))
    out = io.BytesIO()
    half.save(out, "PNG")
    facts = measure_image(out.getvalue())
    assert facts.mime == "image/png" and facts.brightness == 128 and facts.contrast == 128
    assert isinstance(facts.brightness, int) and isinstance(facts.contrast, int)


def test_measure_image_refuses_things_that_are_not_pictures() -> None:
    with pytest.raises(PackagingError, match="could not be read"):
        measure_image(b"not an image at all")
    with pytest.raises(PackagingError, match="JPEG, PNG or WebP"):
        measure_image(picture("GIF"))
    assert measure_image(picture("WEBP")).extension == "webp"


# --- the fact guard -------------------------------------------------------------------------------
ALLOWED = set(words_of("Cactus repotting guide How to repot a cactus REPOT IT"))


def test_words_of_folds_case_and_strips_punctuation() -> None:
    assert words_of("Repot, IT! (now)") == ["repot", "it", "now"]
    assert words_of("") == []


def test_guard_overlay_keeps_words_from_the_creators_text() -> None:
    assert guard_overlay("REPOT it!", ALLOWED) == ("REPOT it!", "")
    assert guard_overlay("", ALLOWED) == ("", "")
    assert guard_overlay(None, ALLOWED) == ("", "")


def test_guard_overlay_drops_invented_words_and_long_text() -> None:
    text, problem = guard_overlay("Repot FREE cactus", ALLOWED)
    assert text == "" and '"free"' in problem
    text, problem = guard_overlay("how to repot a cactus guide", ALLOWED)  # six words
    assert text == "" and "more than 5 words" in problem
    assert guard_overlay("cactus 100", ALLOWED)[0] == ""  # a number that is not in the text


def test_measurement_numbers_collects_nested_ints_only() -> None:
    found = measurement_numbers({"width": 1280, "subject": {"share_pct": 35, "found": True},
                                 "text_zone": "top-left", "x": [7, {"y": 0}]})
    assert found == {"1280", "35", "7", "0"}  # True is not a number


def test_guard_point_drops_numbers_that_were_not_measured() -> None:
    allowed = {"1280", "35"}
    assert guard_point("The subject takes 35% of the frame.", allowed) == (
        "The subject takes 35% of the frame.", "")
    text, problem = guard_point("The subject takes 90% of the frame.", allowed)
    assert text == "" and '"90"' in problem
    assert guard_point("The picture is 1,280 pixels wide.", allowed)[0] != ""  # separators do not matter
    assert guard_point("No digits here.", set())[0] == "No digits here."
    assert guard_point("Bigger <b>text</b>", set())[0] == "Bigger b text /b"  # no < or >


# --- the Meta clients -----------------------------------------------------------------------------
def delta(text: str) -> str:
    return "data: " + json.dumps({"type": "response.output_text.delta", "delta": text})


def test_sam_client_reads_the_stream_and_sends_the_documented_request() -> None:
    http = FakeHttp(lines=["event: x", delta(SAM_TEXT[:30]), "", "data: not json", delta(SAM_TEXT[30:]),
                           "data: " + json.dumps({"type": "response.completed"})])
    sam = MetaSamClient(KEY, client=http)
    boxes = sam.detect(b"imagebytes", "image/jpeg", "steam iron")
    assert len(boxes) == 2 and boxes[0].c == 0.95
    request = http.requests[0]
    assert request["method"] == "POST" and request["url"] == "https://api.meta.ai/v1/responses"
    body = request["json"]
    assert body["model"] == "sam-3.1" and body["stream"] is True
    assert body["metadata"] == {"include_confidence": "true"}
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "steam iron"}
    assert content[1]["image_url"] == "data:image/jpeg;base64," + base64.b64encode(b"imagebytes").decode()
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"


def test_sam_client_empty_stream_means_no_boxes() -> None:
    http = FakeHttp(lines=[delta(""), "data: " + json.dumps({"type": "response.output_text.done"})])
    assert MetaSamClient(KEY, client=http).detect(b"x", "image/png", "text") == []


@pytest.mark.parametrize("status, kind", [(401, "auth"), (403, "auth"), (429, "rate_limit"),
                                          (500, "api")])
def test_sam_client_maps_http_errors_without_the_key(status: int, kind: str) -> None:
    http = FakeHttp(status=status, body={"error": {"message": f"bad token {KEY} given"}})
    with pytest.raises(ModelError) as caught:
        MetaSamClient(KEY, client=http).detect(b"x", "image/png", "cactus")
    assert caught.value.kind == kind and KEY not in str(caught.value)
    if kind == "api":
        assert "HTTP 500" in str(caught.value) and "***" in str(caught.value)


def test_sam_client_network_failures_and_stream_errors() -> None:
    for error in (httpx.ReadTimeout("slow"), httpx.ConnectError("down")):
        with pytest.raises(ModelError) as caught:
            MetaSamClient(KEY, client=FakeHttp(error=error)).detect(b"x", "image/png", "cactus")
        assert caught.value.kind == "network"
    failed = FakeHttp(lines=["data: " + json.dumps({"type": "response.failed"})])
    with pytest.raises(ModelError) as caught:
        MetaSamClient(KEY, client=failed).detect(b"x", "image/png", "cactus")
    assert caught.value.kind == "api"


def test_clients_never_show_the_key_in_a_repr() -> None:
    sam, maker = MetaSamClient(KEY, client=FakeHttp()), MetaImageClient(KEY, client=object())
    tools = packaging.PackagingTools(FakeModelClient([]), sam, maker)
    for item in (sam, maker, tools):
        assert KEY not in repr(item) and KEY not in str(item)


class FakeResponses:
    def __init__(self, result: Any) -> None:
        self.result, self.calls = result, []

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def image_client(result: Any) -> tuple[MetaImageClient, FakeResponses]:
    responses = FakeResponses(result)
    return MetaImageClient(KEY, client=SimpleNamespace(responses=responses)), responses


def test_image_client_returns_the_first_picture_and_sends_the_edit_request() -> None:
    raw = picture("WEBP")
    output = [SimpleNamespace(type="reasoning"), SimpleNamespace(type="image_generation_call", result=""),
              SimpleNamespace(type="image_generation_call", result=base64.b64encode(raw).decode()),
              SimpleNamespace(type="message")]
    maker, responses = image_client(SimpleNamespace(output=output))
    assert maker.edit(b"orig", "image/jpeg", "Make it brighter") == raw
    call = responses.calls[0]
    assert call["model"] == "muse-image-1.0"
    content = call["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Make it brighter"}
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_image_client_without_a_picture_is_an_api_error() -> None:
    for output in ([SimpleNamespace(type="message")], [SimpleNamespace(type="image_generation_call", result="!!")],
                   None):
        maker, _ = image_client(SimpleNamespace(output=output))
        with pytest.raises(ModelError) as caught:
            maker.edit(b"x", "image/jpeg", "go")
        assert caught.value.kind == "api"


def test_image_client_maps_openai_errors() -> None:
    request = httpx.Request("POST", "https://api.meta.ai/v1/responses")

    def response(status: int) -> httpx.Response:
        return httpx.Response(status, request=request)

    cases = [
        (openai.AuthenticationError("denied", response=response(401), body=None), "auth"),
        (openai.PermissionDeniedError("denied", response=response(403), body=None), "auth"),
        (openai.RateLimitError("slow down", response=response(429), body=None), "rate_limit"),
        (openai.APIConnectionError(request=request), "network"),
        (openai.APITimeoutError(request=request), "network"),
        (openai.InternalServerError(f"boom {KEY[:0]}", response=response(500), body=None), "api"),
    ]
    for error, kind in cases:
        maker, _ = image_client(error)
        with pytest.raises(ModelError) as caught:
            maker.edit(b"x", "image/jpeg", "go")
        assert caught.value.kind == kind and KEY not in str(caught.value)


def test_packaging_tools_need_a_key_and_use_the_standard_model(tmp_path: Path) -> None:
    config = load_config({"YEZATAS_HOME": str(tmp_path / "home")})
    assert packaging_tools_from_config(config, {}) is None
    tools = packaging_tools_from_config(config, {"MODEL_API_KEY": KEY})
    assert tools is not None
    assert tools.vision.model == "muse-spark-1.3" and tools.segmenter.model == "sam-3.1"
    assert tools.maker.model == "muse-image-1.0"
    assert KEY not in repr(tools)
    with pytest.raises(Exception):  # a Contributor model is refused by the configuration itself
        load_config({"YEZATAS_HOME": str(tmp_path / "h"), "YEZATAS_MODEL": "muse-spark-1.3-contributor"})


# --- review_packaging -----------------------------------------------------------------------------
def test_review_measures_reads_and_stores(store: Store, images: Path) -> None:
    vision = FakeModelClient([reading(face=True), critique(
        points=["The subject takes 25% of the frame.", "Bright enough at 200."])])
    sam = FakeSegmenter({"cactus": [sam_box(0, 0, 320, 180, c=0.4), sam_box(0, 0, 320, 180, c=0.9)],
                         "face": [sam_box(560, 300, 640, 360, c=0.7)]})
    seen: list[str] = []
    result = run_review(store, images, vision, sam, report=seen.append)
    assert result.status == "done" and "2 points and 2 variant ideas" in result.message
    review = result.review
    assert review is not None and review.id > 0 and review.status == "done" and review.error == ""
    assert store.latest_packaging_review(VID) == review
    assert review.title == TITLE and review.thumbnail_url == URL
    m = review.measurements
    assert (m["width"], m["height"], m["brightness"], m["contrast"]) == (640, 360, 200, 0)
    assert m["subject"] == {"phrase": "cactus", "found": True, "share_pct": 25, "confidence_pct": 90,
                            "in_badge_corner": False}
    assert m["face"]["share_pct"] == 2 and m["face"]["in_badge_corner"] is True
    assert m["badge_risk"] is True and m["text_zone"] == "top-left"
    assert [c[2] for c in sam.calls] == ["cactus", "face"]
    assert review.reading == {"text": "REPOT IT", "text_zone": "top-left", "subject": "cactus", "face": True}
    assert review.critique["points"] == ["The subject takes 25% of the frame.", "Bright enough at 200."]
    assert [i["overlay_text"] for i in review.critique["ideas"]] == ["REPOT IT", "cactus guide"]
    assert review.problems == []
    assert review.models == {"vision": "muse-spark-1.3", "segmenter": "sam-test"}
    assert (review.input_tokens, review.cached_tokens, review.output_tokens) == (2000, 0, 200)
    assert review.cost_usd == pytest.approx(estimate_cost("muse-spark-1.3", 2000, 0, 200) + 2 * 0.0025)
    assert seen[0] == "Downloading the thumbnail" and seen[-1] == "Saving the check"


def test_review_saves_the_thumbnail_under_a_generated_bare_name(store: Store, images: Path) -> None:
    data = picture()
    result = run_review(store, images, FakeModelClient([reading(), critique()]), data=data)
    name = result.review.image_file
    assert Path(name).name == name and name.startswith("thumb-") and name.endswith(".jpg")
    assert (images / VID / name).read_bytes() == data
    assert stored_image_path(images, VID, name) == images / VID / name
    for bad in ("", "..", "../x.jpg", "a/b.jpg", "a\\b.jpg"):
        with pytest.raises(PackagingError):
            stored_image_path(images, VID, bad)


def test_review_sends_the_picture_and_uses_strict_json_schema(store: Store, images: Path) -> None:
    vision = FakeModelClient([reading(), critique()])
    run_review(store, images, vision)
    first, second = vision.calls
    part = first["messages"][1]["content"][1]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    for call in (first, second):
        fmt = call["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert set(first["response_format"]["json_schema"]["schema"]["properties"]) == {
        "text", "text_zone", "subject", "face"}
    prompt = second["messages"][1]["content"]
    assert TITLE in prompt and '"brightness": 200' in prompt and "REPOT IT" in prompt
    assert "contributor" not in vision.model


def test_review_only_asks_sam_for_the_face_when_there_is_one(store: Store, images: Path) -> None:
    sam = FakeSegmenter()
    run_review(store, images, FakeModelClient([reading(face=False), critique()]), sam)
    assert [c[2] for c in sam.calls] == ["cactus"]
    sam = FakeSegmenter()
    result = run_review(store, images, FakeModelClient([reading(subject=""), critique()]), sam)
    assert sam.calls == [] and "subject" not in result.review.measurements
    assert result.review.cost_usd == pytest.approx(estimate_cost("muse-spark-1.3", 2000, 0, 200))


def test_review_subject_not_found_and_text_in_the_badge_corner(store: Store, images: Path) -> None:
    result = run_review(store, images, FakeModelClient([reading(zone="bottom-right"), critique()]))
    assert result.review.measurements["subject"] == {"phrase": "cactus", "found": False}
    assert result.review.measurements["badge_risk"] is True
    result = run_review(store, images, FakeModelClient([reading(zone="top-right"), critique()]))
    assert result.review.measurements["badge_risk"] is False


def test_review_fact_guard_drops_invented_words_and_numbers(store: Store, images: Path) -> None:
    vision = FakeModelClient([reading(), critique(
        points=["The subject takes 90% of the frame.", "Keep the bold text.", "Brightness is 200."],
        ideas=[("Add a banner", "FREE CACTUS"), ("add  a BANNER", "Repot"), ("Zoom in", "Cactus guide")])])
    review = run_review(store, images, vision).review
    assert review.critique["points"] == ["Keep the bold text.", "Brightness is 200."]
    assert review.critique["ideas"] == [{"instruction": "Add a banner", "overlay_text": ""},
                                        {"instruction": "Zoom in", "overlay_text": "Cactus guide"}]
    assert any('"free"' in p for p in review.problems)
    assert any('"90"' in p for p in review.problems)
    assert any("repeated another idea" in p for p in review.problems)
    assert review.status == "done"


def test_review_points_may_quote_numbers_the_creator_wrote(store: Store, images: Path) -> None:
    save_draft(store, VID, "Cactus repotting guide: 3 mistakes", DESCRIPTION, (), source="manual",
               note="", now=NOW)
    vision = FakeModelClient([reading(), critique(
        points=["The title promises 3 mistakes; show one.", "Name 4 mistakes instead."])])
    review = run_review(store, images, vision).review
    assert review.critique["points"] == ["The title promises 3 mistakes; show one."]
    assert any('"4"' in p for p in review.problems)


def test_review_uses_the_draft_title_and_words(store: Store, images: Path) -> None:
    save_draft(store, VID, "Cactus rescue plan", DESCRIPTION, (), source="manual", note="", now=NOW)
    vision = FakeModelClient([reading(), critique(ideas=[("Zoom", "rescue plan"), ("Dark", "guide")])])
    review = run_review(store, images, vision).review
    assert review.title == "Cactus rescue plan"
    assert [i["overlay_text"] for i in review.critique["ideas"]] == ["rescue plan", ""]  # "guide" is old


def test_review_text_words_may_come_from_the_thumbnail_text(store: Store, images: Path) -> None:
    vision = FakeModelClient([reading(text="NEVER KILL AGAIN"), critique(ideas=[("Bold", "NEVER AGAIN"),
                                                                             ("Dim", "kill")])])
    ideas = run_review(store, images, vision).review.critique["ideas"]
    assert [i["overlay_text"] for i in ideas] == ["NEVER AGAIN", "kill"]


def test_review_sam_trouble_is_a_problem_not_a_failure(store: Store, images: Path) -> None:
    sam = FakeSegmenter({"cactus": ModelError("network", "Could not reach the segmentation service.")})
    result = run_review(store, images, FakeModelClient([reading(), critique()]), sam)
    assert result.status == "done" and "subject" not in result.review.measurements
    assert any("could not be measured" in p for p in result.review.problems)


def test_review_refuses_without_storing(store: Store, images: Path) -> None:
    vision = FakeModelClient([])
    unknown = review_packaging(store, vision, FakeSegmenter(), "vid00000404", images_dir=images,
                               fetch=lambda url: b"")
    assert unknown.status == "failed" and "not in your catalog" in unknown.message and unknown.review is None
    store.upsert_video(Video(video_id="vid00000002", thumbnails={}), store.begin_pull("c", now=NOW), now=NOW)
    none = review_packaging(store, vision, FakeSegmenter(), "vid00000002", images_dir=images,
                            fetch=lambda url: b"")
    assert none.status == "failed" and "no thumbnail" in none.message
    store.upsert_video(Video(video_id="vid00000003", thumbnails={"high": {"url": "https://example.invalid/a.jpg"}}),
                       store.begin_pull("c", now=NOW), now=NOW)
    other = review_packaging(store, vision, FakeSegmenter(), "vid00000003", images_dir=images,
                             fetch=lambda url: pytest.fail("must not download"))
    assert other.status == "failed" and "only downloads thumbnails" in other.message
    assert all(store.latest_packaging_review(v) is None for v in ("vid00000404", "vid00000002", "vid00000003"))
    assert vision.calls == []


@pytest.mark.parametrize("kind", ["rate_limit", "network", "api"])
def test_review_model_failure_stores_a_failed_review(store: Store, images: Path, kind: str) -> None:
    vision = FakeModelClient([reading(), ModelError(kind, "The model said no.")])
    result = run_review(store, images, vision)
    assert result.status == "failed" and result.message == "The model said no."
    review = store.latest_packaging_review(VID)
    assert review is not None and review.status == "failed" and review.error == "The model said no."
    assert review.reading["subject"] == "cactus" and review.image_file == "" and review.critique == {}
    assert review.input_tokens == 1000  # the first call was paid for
    assert not images.exists()


def test_review_auth_failure_raises_and_stores_nothing(store: Store, images: Path) -> None:
    for vision, sam in (
            (FakeModelClient([ModelError("auth", "Key refused.")]), FakeSegmenter()),
            (FakeModelClient([reading()]), FakeSegmenter({"cactus": ModelError("auth", "Key refused.")})),
            (FakeModelClient([reading(), ModelError("auth", "Key refused.")]), FakeSegmenter())):
        with pytest.raises(ModelError) as caught:
            run_review(store, images, vision, sam)
        assert caught.value.kind == "auth"
        assert store.latest_packaging_review(VID) is None
    assert not images.exists()


def test_review_bad_answers_and_bad_pictures_are_failed_reviews(store: Store, images: Path) -> None:
    junk = ModelReply("sorry, I cannot", (), {})
    result = run_review(store, images, FakeModelClient([junk]))
    assert result.status == "failed" and "could not be read" in result.message
    result = run_review(store, images, FakeModelClient([reading(), ModelReply("[1, 2]", (), {})]))
    assert result.status == "failed"
    result = run_review(store, images, FakeModelClient([]), data=b"not a picture")
    assert result.status == "failed" and "could not be read as a picture" in result.message


def test_review_download_failure_is_a_failed_review(store: Store, images: Path) -> None:
    def fetch(url: str) -> bytes:
        raise PackagingError("Could not download the thumbnail.")

    result = review_packaging(store, FakeModelClient([]), FakeSegmenter(), VID, images_dir=images,
                              fetch=fetch, now=NOW)
    assert result.status == "failed" and result.review is not None and result.review.status == "failed"


def test_review_reading_is_cleaned(store: Store, images: Path) -> None:
    odd = reply({"text": "  A\n B ", "text_zone": "everywhere", "subject": "big red steam iron machine",
                 "face": "yes"})
    review = run_review(store, images, FakeModelClient([odd, critique()])).review
    assert review.reading == {"text": "A B", "text_zone": "none", "subject": "big red steam", "face": False}


def test_review_key_never_appears_in_stored_text(store: Store, images: Path) -> None:
    sam = MetaSamClient(KEY, client=FakeHttp(status=500, body={"error": {"message": f"oops {KEY}"}}))
    result = run_review(store, images, FakeModelClient([reading(), critique()]), sam)
    review = result.review
    assert KEY not in json.dumps([review.problems, review.error, review.measurements])
    assert any("***" in p for p in review.problems)


# --- make_variants --------------------------------------------------------------------------------
def make_review(store: Store, images: Path, **kw: Any) -> int:
    vision = FakeModelClient([reading(), critique(**kw)])
    sam = FakeSegmenter({"cactus": [sam_box(0, 0, 320, 180)]})
    return run_review(store, images, vision, sam).review.id


def made_picture(color: Any = (30, 120, 60)) -> bytes:
    return picture("WEBP", (2048, 1152), color)


def read_back(text: str) -> ModelReply:
    return reading(text=text, zone="top-left")


def test_variants_make_two_checked_jpegs(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([made_picture(), made_picture((60, 60, 200))])
    vision = FakeModelClient([read_back("REPOT IT"), read_back("cactus guide")])
    sam = FakeSegmenter({"cactus": [sam_box(0, 0, 640, 180, c=0.8)]})
    seen: list[str] = []
    result = make_variants(store, maker, vision, sam, review_id, images_dir=images, now=NOW,
                           report=seen.append)
    assert result.status == "done" and result.message == "Made 2 variant ideas."
    first, second = result.variants
    assert store.list_thumbnail_variants(review_id) == [first, second]
    assert (first.status, first.text_ok, first.text_read) == ("ok", True, "REPOT IT")
    assert (second.status, second.text_ok) == ("ok", True)
    for variant in (first, second):
        path = images / VID / variant.image_file
        assert Path(variant.image_file).name == variant.image_file and variant.image_file.endswith(".jpg")
        with Image.open(path) as saved:
            assert saved.format == "JPEG" and saved.size == (1280, 720)
        assert variant.review_id == review_id and variant.model == "image-test"
        assert variant.measurements["subject"]["before_share_pct"] == 25
        assert variant.measurements["subject"]["after"]["share_pct"] == 50
        expected = 0.01 + 0.0025 + estimate_cost("muse-spark-1.3", 1000, 0, 100)
        assert variant.cost_usd == pytest.approx(expected)
    assert first.image_file != second.image_file
    assert len(seen) >= 2


def test_variant_prompt_has_the_guard_and_the_exact_text() -> None:
    prompt = variant_prompt("Zoom in on the cactus", "REPOT IT")
    assert "Zoom in on the cactus" in prompt and packaging.GUARD_SENTENCE in prompt
    assert 'exactly this text on the picture: "REPOT IT"' in prompt
    assert packaging.KEEP_TEXT_SENTENCE in variant_prompt("Zoom in", "")


def test_variants_send_the_stored_thumbnail_and_the_prompts(store: Store, images: Path) -> None:
    review_id = make_review(store, images, ideas=[("Zoom in", "REPOT IT"), ("Dim the room", "")])
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([read_back("REPOT IT"), read_back("REPOT IT")])
    make_variants(store, maker, vision, FakeSegmenter(), review_id, images_dir=images)
    assert 'exactly this text on the picture: "REPOT IT"' in maker.prompts[0]
    assert packaging.KEEP_TEXT_SENTENCE in maker.prompts[1]
    read = vision.calls[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert read.startswith("data:image/jpeg;base64,")


def test_variant_with_other_text_is_flagged(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([read_back("REPOT YT"), read_back("Cactus  GUIDE!")])
    result = make_variants(store, maker, vision, FakeSegmenter(), review_id, images_dir=images)
    first, second = result.variants
    assert (first.status, first.text_ok, first.text_read) == ("flagged", False, "REPOT YT")
    assert (second.status, second.text_ok) == ("ok", True)  # case, spaces and punctuation do not count
    assert result.status == "done" and "1 with text to check" in result.message


def test_variant_text_may_have_the_creators_own_words_around_it(store: Store, images: Path) -> None:
    # a brand printed on the product is read too; it is fine when the title or description has it
    review_id = make_review(store, images)
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([read_back("Cactus roots REPOT IT"), read_back("BUY NOW cactus guide")])
    first, second = make_variants(store, maker, vision, FakeSegmenter(), review_id,
                                  images_dir=images).variants
    assert (first.status, first.text_ok) == ("ok", True)
    assert (second.status, second.text_ok) == ("flagged", False)  # "buy now" is nobody's words


def test_text_matches_needs_the_whole_text_in_order() -> None:
    assert packaging.text_matches("Acme REPLACE YOUR IRON", "replace your iron", {"acme"})
    assert not packaging.text_matches("YOUR REPLACE IRON", "replace your iron", set())
    assert not packaging.text_matches("REPLACE YOUR", "replace your iron", set())


def test_variant_that_keeps_the_text_is_checked_against_the_original_text(store: Store, images: Path) -> None:
    review_id = make_review(store, images, ideas=[("Dim", ""), ("Bright", "")])
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([read_back("REPOT IT"), read_back("")])
    first, second = make_variants(store, maker, vision, FakeSegmenter(), review_id,
                                  images_dir=images).variants
    assert first.text_ok is True and second.text_ok is False and second.status == "flagged"


def test_variant_text_is_not_checked_when_nothing_was_requested(store: Store, images: Path) -> None:
    review_id = run_review(store, images, FakeModelClient([reading(text=""), critique(ideas=[("Dim", "")])])).review.id
    variants = make_variants(store, FakeMaker([made_picture()]), FakeModelClient([read_back("SURPRISE")]),
                             FakeSegmenter(), review_id, images_dir=images).variants
    assert variants[0].text_ok is None and variants[0].status == "ok" and variants[0].text_read == "SURPRISE"


def test_one_bad_variant_does_not_fail_the_other(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([ModelError("rate_limit", "The Meta API is limiting requests."), made_picture()])
    vision = FakeModelClient([read_back("cactus guide")])
    result = make_variants(store, maker, vision, FakeSegmenter(), review_id, images_dir=images)
    bad, good = result.variants
    assert (bad.status, bad.image_file, bad.cost_usd) == ("failed", "", 0.0)
    assert bad.error == "The Meta API is limiting requests."
    assert good.status == "ok" and good.image_file
    assert result.status == "done" and result.message == "Made 1 variant idea."


def test_a_picture_that_is_not_an_image_fails_that_variant_only(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([b"garbage", made_picture()])
    result = make_variants(store, maker, FakeModelClient([read_back("cactus guide")]), FakeSegmenter(),
                           review_id, images_dir=images)
    bad, good = result.variants
    assert bad.status == "failed" and "not a picture" in bad.error and bad.cost_usd == pytest.approx(0.01)
    assert good.status == "ok"


def test_all_variants_failing_is_a_failed_result(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([ModelError("network", "Offline."), ModelError("network", "Offline.")])
    result = make_variants(store, maker, FakeModelClient([]), FakeSegmenter(), review_id, images_dir=images)
    assert result.status == "failed" and result.message == "Offline." and len(result.variants) == 2


def test_variants_auth_stops_everything(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([ModelError("auth", "Key refused."), made_picture()])
    with pytest.raises(ModelError) as caught:
        make_variants(store, maker, FakeModelClient([]), FakeSegmenter(), review_id, images_dir=images)
    assert caught.value.kind == "auth" and len(maker.prompts) == 1
    assert store.list_thumbnail_variants(review_id) == []
    # a refused key while reading the text back also stops
    maker = FakeMaker([made_picture(), made_picture()])
    with pytest.raises(ModelError):
        make_variants(store, maker, FakeModelClient([ModelError("auth", "Key refused.")]), FakeSegmenter(),
                      review_id, images_dir=images)
    assert store.list_thumbnail_variants(review_id) == [] and len(maker.prompts) == 1


def test_a_failed_text_check_or_measurement_keeps_the_picture(store: Store, images: Path) -> None:
    review_id = make_review(store, images)
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([ModelError("rate_limit", "Slow down."), ModelReply("nonsense", (), {})])
    sam = FakeSegmenter({"cactus": ModelError("network", "Offline.")})
    first, second = make_variants(store, maker, vision, sam, review_id, images_dir=images).variants
    assert first.status == "flagged" and first.image_file and first.text_ok is None
    assert "could not be checked" in first.error and "Slow down." in first.error
    assert second.status == "flagged" and "could not be read" in second.error
    assert first.measurements == {}


def test_variants_refuse_without_a_usable_review(store: Store, images: Path) -> None:
    maker, vision, sam = FakeMaker([]), FakeModelClient([]), FakeSegmenter()
    assert make_variants(store, maker, vision, sam, 999, images_dir=images).status == "failed"
    failed = run_review(store, images, FakeModelClient([ModelError("api", "Down.")])).review
    result = make_variants(store, maker, vision, sam, failed.id, images_dir=images)
    assert result.status == "failed" and "no ideas" in result.message
    no_ideas = run_review(store, images, FakeModelClient([reading(), critique(ideas=[])])).review
    assert make_variants(store, maker, vision, sam, no_ideas.id, images_dir=images).status == "failed"
    ok = make_review(store, images)
    review = store.get_packaging_review(ok)
    (images / VID / review.image_file).unlink()
    result = make_variants(store, maker, vision, sam, ok, images_dir=images)
    assert result.status == "failed" and "Run the check again" in result.message
    assert maker.prompts == []


def test_variant_files_never_use_names_from_the_model(store: Store, images: Path) -> None:
    review_id = make_review(store, images, ideas=[("Save it as ../../evil.jpg", "REPOT IT"), ("Dim", "")])
    maker = FakeMaker([made_picture(), made_picture()])
    vision = FakeModelClient([read_back("REPOT IT"), read_back("REPOT IT")])
    variants = make_variants(store, maker, vision, FakeSegmenter(), review_id, images_dir=images).variants
    assert all(v.image_file.startswith("variant-") for v in variants)
    assert sorted(p.name for p in (images / VID).iterdir() if p.name.startswith("variant-")) == sorted(
        v.image_file for v in variants)
    assert not (images.parent / "evil.jpg").exists()


def test_replace_is_not_needed_for_frozen_records() -> None:
    # PackagingResult / VariantsResult are plain frozen records
    result = packaging.VariantsResult("done", "ok")
    assert replace(result, message="x").message == "x" and result.variants == ()
