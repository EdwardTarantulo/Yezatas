"""Packaging check in the web UI: the Packaging card, the check and variants jobs, the picture routes and the
feed preview. The three Meta clients and the thumbnail download are fakes; nothing touches the network, and
nothing is ever sent to YouTube."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog
from yezatas import agent, edits, packaging
from yezatas.agent import ModelError, ModelReply
from yezatas.modelclient import AUTH_MESSAGE
from yezatas.packaging import Box, PackagingError, PackagingTools
from yezatas.store import Store
from yezatas.web import views
from yezatas.web.app import NO_PACKAGING_KEY

V1, V2 = "vid00000001", "vid00000002"
MAXRES = f"https://i.ytimg.com/vi/{V1}/maxresdefault.jpg"
DESCRIPTION = "How to repot a cactus without hurting its roots."
KEEP_SENTENCE = "Yezatas does not upload thumbnails; download one and upload it in YouTube Studio."
AI_NOTE = "AI-made image: check that the product still looks like the real one"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


# --- fakes ---------------------------------------------------------------------------------------
def picture(fmt: str = "JPEG", size: tuple[int, int] = (640, 360)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(out, fmt)
    return out.getvalue()


def reply(data: dict[str, Any]) -> ModelReply:
    return ModelReply(json.dumps(data), (), {}, 1000, 0, 100)


def reading(text: str = "REPOT IT", zone: str = "top-left", subject: str = "cactus",
            face: bool = False) -> ModelReply:
    return reply({"text": text, "text_zone": zone, "subject": subject, "face": face})


def critique(points: list[str] | None = None,
             ideas: list[tuple[str, str]] | None = None) -> ModelReply:
    ideas = ([("Zoom in on the cactus", "REPOT IT"), ("Brighter background", "cactus guide")]
             if ideas is None else ideas)
    return reply({"points": points if points is not None else ["The text is easy to read."],
                  "ideas": [{"instruction": i, "overlay_text": t} for i, t in ideas]})


class FakeSegmenter:
    def __init__(self) -> None:
        self.model = "sam-test"
        self.calls: list[str] = []

    def detect(self, image_bytes: bytes, mime: str, phrase: str) -> list[Box]:
        self.calls.append(phrase)
        return [Box(100, 50, 300, 150, 640, 360, 0.9)] if phrase == "cactus" else []


class FakeMaker:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.model = "image-test"
        self.script = list(script) if script is not None else [picture("PNG", (512, 288))] * 2
        self.prompts: list[str] = []

    def edit(self, image_bytes: bytes, mime: str, instruction: str) -> bytes:
        self.prompts.append(instruction)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Tools:
    """The injected packaging factory. The default script is a check followed by two variants whose
    text reads back as asked."""

    def __init__(self, script: list[Any] | None = None, maker: FakeMaker | None = None) -> None:
        self.vision = FakeModelClient(
            script if script is not None
            else [reading(), critique(), reading("REPOT IT"), reading("cactus guide")])
        self.segmenter = FakeSegmenter()
        self.maker = maker or FakeMaker()

    def __call__(self, config: Any) -> PackagingTools:
        return PackagingTools(self.vision, self.segmenter, self.maker)  # type: ignore[arg-type]


class Fetch:
    def __init__(self, error: Exception | None = None, gate: threading.Event | None = None) -> None:
        self.urls: list[str] = []
        self.error = error
        self.gate = gate
        self.entered = threading.Event()

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        return picture()


def make_pk_ui(tmp_path: Path, tools: Any = "default", fetch: Any = None, others: int = 0,
               **kwargs: Any) -> tuple[Any, Any, Fetch]:
    tools = Tools() if tools == "default" else tools
    fetch = fetch or Fetch()
    ui = make_ui(tmp_path, packaging_tools=tools if tools is not None else (lambda cfg: None),
                 packaging_fetch=fetch, **kwargs)
    videos = [
        make_video(1, video_id=V1, title="Cactus repotting guide", description=DESCRIPTION, tags=[],
                   privacy_status="public", view_count=500, duration="PT1H2M3S",
                   thumbnails={"maxres": {"url": MAXRES},
                               "medium": {"url": f"https://i.ytimg.com/vi/{V1}/mqdefault.jpg"}}),
        make_video(2, video_id=V2, title="Second video", description="Plain.", tags=[],
                   privacy_status="public", view_count=100),
    ]
    views_of = [300, 9000, 50, 7000, 4000, 8000, 10, 6000, 5000]
    for i in range(others):
        videos.append(make_video(10 + i, video_id=f"vid{10 + i:08d}", title=f"Other {i}",
                                 description=f"Other video number {i}.", tags=[],
                                 privacy_status="public", view_count=views_of[i % len(views_of)]))
    seed_catalog(ui.config, videos)
    return ui, tools, fetch


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any, video_id: str = V1) -> str:
    return ui.client.get(f"/videos/{video_id}").text


def card(html: str) -> str:
    """The Packaging card, up to the end of the card."""
    return html.split('id="thumbnail"')[1].split("</section>")[0]


def fact(section: str, label: str) -> str:
    match = re.search(rf"<dt>{re.escape(label)}</dt><dd>(.*?)</dd>", section)
    assert match, f"no fact {label!r}"
    return match.group(1)


def run_check(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/packaging")
    ui.web.packaging_job.wait()
    return resp


def run_variants(ui: Any, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/packaging/variants")
    ui.web.variants_job.wait()
    return resp


def seed_review(ui: Any, tools: Tools, video_id: str = V1) -> int:
    """A stored check, made directly (no job), with its picture on disk."""
    with Store(ui.config.catalog_db) as store:
        result = packaging.review_packaging(store, tools.vision, tools.segmenter, video_id,
                                            images_dir=ui.web.images_dir, fetch=lambda url: picture())
    assert result.review is not None and result.status == "done", result.message
    return result.review.id


def seed_variants(ui: Any, tools: Tools, review_id: int) -> None:
    with Store(ui.config.catalog_db) as store:
        result = packaging.make_variants(store, tools.maker, tools.vision, tools.segmenter, review_id,
                                         images_dir=ui.web.images_dir)
    assert result.variants, result.message


def stored_variants(ui: Any) -> list[Any]:
    with Store(ui.config.catalog_db) as store:
        review = store.latest_packaging_review(V1)
        return store.list_thumbnail_variants(review.id) if review else []


# --- the card ------------------------------------------------------------------------------------
def test_the_card_renders_before_any_check(tmp_path: Path) -> None:
    ui, _, fetch = make_pk_ui(tmp_path)
    html = page(ui)
    section = card(html)
    assert "Thumbnail: title and thumbnail together" in section
    assert "Cactus repotting guide" in section
    assert f"https://i.ytimg.com/vi/{V1}/mqdefault.jpg" in section  # the current thumbnail
    assert f'action="/videos/{V1}/packaging"' in section and 'name="csrf"' in section
    assert re.search(r'class="button primary">Research thumbnail<', section)
    assert section.count(KEEP_SENTENCE) == 1
    assert "packaging-result" not in section and "Generate 2" not in section and "Variant ideas" not in section
    assert "packaging-no-key" not in section
    assert f'href="/videos/{V1}/packaging"' in section  # preview link
    assert fetch.urls == []


def test_a_check_runs_the_job_and_the_card_shows_what_was_found(tmp_path: Path) -> None:
    ui, tools, fetch = make_pk_ui(tmp_path)
    resp = run_check(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#thumbnail"
    assert fetch.urls == [MAXRES]
    snap = ui.web.packaging_job.snapshot()
    assert snap.status == "done" and snap.result.review.status == "done"
    section = card(page(ui))
    assert 'id="packaging-done"' in section and "Checked the packaging: 1 point and 2 variant ideas." in section
    assert fact(section, "Size") == "640 x 360 pixels"
    assert fact(section, "Brightness") == "200 out of 255 on average"
    assert fact(section, "Contrast").startswith("0 ")
    assert fact(section, "Subject share") == "9% of the frame"
    assert fact(section, "Face share") == "no face seen"
    assert fact(section, "Text or subject in the duration-badge corner") == "no"
    assert fact(section, "Text zone") == "top-left"
    assert fact(section, "Text on the thumbnail") == "“REPOT IT”"
    assert fact(section, "Subject") == "cactus" and fact(section, "Face") == "no"
    assert "<li>The text is easy to read.</li>" in section
    assert "Zoom in on the cactus" in section and "Brighter background" in section
    assert "Text on the picture: &ldquo;REPOT IT&rdquo;" in section
    assert "Title that was checked: Cactus repotting guide" in section
    assert "muse-spark-1.3, sam-test" in section and "Cost " in section
    assert re.search(r'class="button primary">Generate 2 thumbnail options<', section)
    assert "packaging-problems" not in section
    with Store(ui.config.catalog_db) as store:
        review = store.latest_packaging_review(V1)
    assert views.format_cost(review.cost_usd) in section
    assert (ui.web.images_dir / V1 / review.image_file).is_file()


def test_text_in_the_badge_corner_is_reported(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, Tools([reading(zone="bottom-right"), critique()]))
    run_check(ui)
    section = card(page(ui))
    assert fact(section, "Text or subject in the duration-badge corner") == "yes"
    assert fact(section, "Text zone") == "bottom-right"


def test_the_fact_guard_problems_are_listed_and_a_dropped_text_keeps_the_current_one(
        tmp_path: Path) -> None:
    script = [reading(), critique(points=["Zoom in to 9999 pixels.", "Keep the bright colours."],
                                  ideas=[("Brighter background", "free shipping"),
                                         ("Darker background", "cactus")])]
    ui, _, _ = make_pk_ui(tmp_path, Tools(script))
    run_check(ui)
    section = card(page(ui))
    problems = section.split('id="packaging-problems"')[1].split("</ul>")[0]
    assert "9999" in problems and "is not in the title" in problems
    assert "Zoom in to 9999" not in section.split("packaging-problems")[0]
    assert "<li>Keep the bright colours.</li>" in section
    assert "Keeps the current text." in section  # the dropped overlay text


def test_a_failed_check_shows_its_sentence_and_no_facts(tmp_path: Path) -> None:
    fetch = Fetch(error=PackagingError("The thumbnail could not be downloaded (HTTP 404)."))
    ui, _, _ = make_pk_ui(tmp_path, fetch=fetch)
    run_check(ui)
    section = card(page(ui))
    assert 'class="flash error" role="status" id="packaging-done"' in section
    assert "The thumbnail could not be downloaded (HTTP 404)." in section
    assert "packaging-facts" not in section and "Generate 2" not in section
    assert nf(ui, "/packaging/done", data={"video_id": V1}).status_code == 303
    section = card(page(ui))  # dismissed: the stored failure still shows once
    assert 'id="packaging-error"' in section and "HTTP 404" in section
    assert "packaging-done" not in section


def test_an_unexpected_error_gets_a_plain_sentence(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, fetch=Fetch(error=KeyError("x")))
    run_check(ui)
    snap = ui.web.packaging_job.snapshot()
    assert snap.status == "failed" and snap.message == "The packaging check stopped unexpectedly (KeyError)."
    section = card(page(ui))
    assert 'id="packaging-failed"' in section and "stopped unexpectedly (KeyError)" in section


def test_a_refused_key_fails_the_job_and_stores_nothing(tmp_path: Path) -> None:
    refused = ModelError("auth", AUTH_MESSAGE)
    ui, _, _ = make_pk_ui(tmp_path, Tools([refused]))
    run_check(ui)
    snap = ui.web.packaging_job.snapshot()
    assert snap.status == "failed" and snap.message == AUTH_MESSAGE and snap.error_kind == "auth"
    with Store(ui.config.catalog_db) as store:
        assert store.latest_packaging_review(V1) is None
    assert AUTH_MESSAGE in card(page(ui))
    assert nf(ui, "/packaging/done", data={"video_id": V1}).status_code == 303
    assert 'id="packaging-failed"' not in card(page(ui))


def test_dismissing_a_finished_check_hides_the_message_but_keeps_the_result(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path)
    run_check(ui)
    resp = nf(ui, "/packaging/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#thumbnail"
    section = card(page(ui))
    assert "packaging-done" not in section and "packaging-facts" in section
    assert nf(ui, "/packaging/done", data={}).headers["location"] == "/"


def test_without_a_key_the_post_says_so_and_starts_nothing(tmp_path: Path) -> None:
    ui, _, fetch = make_pk_ui(tmp_path, tools=None)
    html = page(ui)
    assert 'id="packaging-no-key"' in card(html) and 'href="/setup#step-ai"' in card(html)
    assert re.search(r'class="button primary" disabled>Research thumbnail<', card(html))
    resp = nf(ui, f"/videos/{V1}/packaging")
    assert resp.status_code == 303
    assert NO_PACKAGING_KEY in page(ui)
    assert ui.web.packaging_job.snapshot().status == "idle" and fetch.urls == []
    assert NO_PACKAGING_KEY == "There is no Meta API key. Add one in Configuration. Nothing was sent."


def test_the_worker_itself_refuses_when_the_key_disappears(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, tools=None)
    assert ui.web.start_packaging(V1)
    ui.web.packaging_job.wait()
    snap = ui.web.packaging_job.snapshot()
    assert snap.status == "failed" and snap.message == NO_PACKAGING_KEY and snap.error_kind == "auth"
    assert NO_PACKAGING_KEY in card(page(ui))


def test_an_unknown_video_is_a_404_for_both_posts(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path)
    assert nf(ui, "/videos/nosuchvideo/packaging").status_code == 404
    assert nf(ui, "/videos/nosuchvideo/packaging/variants").status_code == 404
    assert ui.web.packaging_job.snapshot().status == "idle"


POSTS = [f"/videos/{V1}/packaging", f"/videos/{V1}/packaging/variants", "/packaging/done",
         "/packaging/variants/done"]


@pytest.mark.parametrize("path", POSTS)
def test_every_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, tools, fetch = make_pk_ui(tmp_path)
    assert ui.client.post(path, data={}, follow_redirects=False).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong"}, follow_redirects=False).status_code == 403
    assert ui.web.packaging_job.snapshot().status == "idle"
    assert ui.web.variants_job.snapshot().status == "idle"
    assert fetch.urls == [] and tools.vision.calls == []


def test_a_running_check_shows_its_message_and_blocks_a_second_one(tmp_path: Path) -> None:
    gate = threading.Event()
    ui, _, fetch = make_pk_ui(tmp_path, fetch=Fetch(gate=gate))
    nf(ui, f"/videos/{V1}/packaging")
    assert fetch.entered.wait(5)
    html = page(ui)
    assert 'data-poll="1"' in html
    assert 'id="packaging-msg"' in card(html) and "Downloading the thumbnail" in card(html)
    assert ui.client.get("/api/status").json()["packaging"]["status"] == "running"
    other = page(ui, V2)
    assert 'data-poll="1"' in other and 'id="packaging-busy"' in card(other)
    assert re.search(r'class="button primary" disabled>Research thumbnail<', card(other))
    nf(ui, f"/videos/{V2}/packaging")
    assert "already running" in page(ui, V2)
    gate.set()
    ui.web.packaging_job.wait()
    assert ui.client.get("/api/status").json()["packaging"]["status"] == "done"
    assert 'data-poll="0"' in page(ui)


def test_a_check_that_takes_too_long_is_given_up_on(tmp_path: Path) -> None:
    gate = threading.Event()
    now = [0.0]
    ui, _, fetch = make_pk_ui(tmp_path, fetch=Fetch(gate=gate), clock=lambda: now[0])
    nf(ui, f"/videos/{V1}/packaging")
    assert fetch.entered.wait(5)
    now[0] = 10_000.0
    snap = ui.web.packaging_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "timeout" and "5 minutes" in snap.message
    gate.set()
    ui.web.packaging_job.wait()
    assert ui.web.packaging_job.snapshot().status == "failed"  # the late result is ignored


# --- variants -------------------------------------------------------------------------------------
def test_the_variants_button_needs_a_check_with_ideas(tmp_path: Path) -> None:
    ui, tools, _ = make_pk_ui(tmp_path)
    resp = nf(ui, f"/videos/{V1}/packaging/variants")
    assert resp.status_code == 303 and "no packaging check with ideas" in page(ui)
    assert ui.web.variants_job.snapshot().status == "idle" and tools.maker.prompts == []
    # a check without ideas has no button either
    ui2, _, _ = make_pk_ui(tmp_path / "b", Tools([reading(), critique(ideas=[])]))
    run_check(ui2)
    assert "Generate 2" not in card(page(ui2))
    nf(ui2, f"/videos/{V1}/packaging/variants")
    assert "no packaging check with ideas" in page(ui2)
    assert ui2.web.variants_job.snapshot().status == "idle"


def test_making_variants_stores_them_and_the_card_shows_each_one(tmp_path: Path) -> None:
    ui, tools, _ = make_pk_ui(tmp_path)
    run_check(ui)
    resp = run_variants(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#thumbnail"
    snap = ui.web.variants_job.snapshot()
    assert snap.status == "done" and snap.result.message == "Made 2 variant ideas."
    assert len(tools.maker.prompts) == 2
    section = card(page(ui))
    assert 'id="variants-done"' in section and "Made 2 variant ideas." in section
    first, second = (section.split(f'id="variant-{v.id}"')[1].split("</li>")[0]
                     for v in stored_variants(ui))
    assert section.count(AI_NOTE) == 2
    assert "Text matches" in first and "Subject share: 9% → 9%" in first
    assert "Text asked for: &ldquo;REPOT IT&rdquo;" in first
    assert 'src="/videos/%s/packaging/variants/1.jpg"' % V1 in first
    assert 'href="/videos/%s/packaging/variants/1.jpg?download=1">Download .jpg<' % V1 in first
    assert 'href="/videos/%s/packaging?variant=1">Preview in a feed<' % V1 in first
    assert "Text matches" in second and "cactus guide" in second
    assert "Generate 2 more thumbnail options" in section


def test_a_variant_whose_text_differs_is_flagged_with_what_was_read(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, Tools([reading(), critique(), reading("REPOT IT"),
                                           reading("WRONG TEXT")]))
    run_check(ui)
    run_variants(ui)
    section = card(page(ui))
    assert "Text differs: reads “WRONG TEXT”" in section
    assert 'class="pill warn">Text to check<' in section
    assert ui.web.variants_job.snapshot().result.message == "Made 2 variant ideas (1 with text to check)."


def test_an_unchecked_text_says_so(tmp_path: Path) -> None:
    script = [reading(text="", zone="none"), critique(ideas=[("Brighter", ""), ("Darker", "")]),
              reading(""), reading("")]
    ui, _, _ = make_pk_ui(tmp_path, Tools(script))
    run_check(ui)
    run_variants(ui)
    section = card(page(ui))
    assert section.count("Text not checked") == 2 and section.count("Keeps the current text.") >= 2


def test_a_failed_variant_has_its_sentence_and_no_picture_or_download(tmp_path: Path) -> None:
    sentence = "The image maker did not send a picture. Try again."
    ui, _, _ = make_pk_ui(tmp_path, Tools(
        [reading(), critique(), reading("cactus guide")],
        FakeMaker([ModelError("api", sentence), picture("PNG", (512, 288))])))
    run_check(ui)
    run_variants(ui)
    failed, ok = stored_variants(ui)
    assert failed.status == "failed" and ok.status == "ok"
    section = card(page(ui))
    one = section.split(f'id="variant-{failed.id}"')[1].split("</li>")[0]
    assert sentence in one and 'class="pill problem">Failed<' in one
    assert "<img" not in one and "Download" not in one and "Preview" not in one
    assert "Download .jpg" in section
    assert ui.client.get(f"/videos/{V1}/packaging/variants/{failed.id}.jpg").status_code == 404


def test_a_refused_key_while_making_variants_fails_the_job(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, Tools([reading(), critique()],
                                          FakeMaker([ModelError("auth", AUTH_MESSAGE)])))
    run_check(ui)
    run_variants(ui)
    snap = ui.web.variants_job.snapshot()
    assert snap.status == "failed" and snap.message == AUTH_MESSAGE and snap.error_kind == "auth"
    assert 'id="variants-failed"' in card(page(ui)) and stored_variants(ui) == []
    assert nf(ui, "/packaging/variants/done", data={"video_id": V1}).status_code == 303
    assert 'id="variants-failed"' not in card(page(ui))


def test_variants_without_a_key_say_so(tmp_path: Path) -> None:
    ui, tools, _ = make_pk_ui(tmp_path)
    seed_review(ui, tools)
    ui.web.packaging_tools = lambda cfg: None
    resp = nf(ui, f"/videos/{V1}/packaging/variants")
    assert resp.status_code == 303 and NO_PACKAGING_KEY in page(ui)
    assert ui.web.variants_job.snapshot().status == "idle"
    assert ui.web.start_variants(V1, 1)
    ui.web.variants_job.wait()
    assert ui.web.variants_job.snapshot().message == NO_PACKAGING_KEY


def test_the_variants_job_blocks_a_second_run(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()
    held = FakeMaker()
    real_edit = held.edit

    def slow_edit(image_bytes: bytes, mime: str, instruction: str) -> bytes:
        entered.set()
        assert gate.wait(10)
        return real_edit(image_bytes, mime, instruction)

    held.edit = slow_edit  # type: ignore[method-assign]
    ui, tools, _ = make_pk_ui(tmp_path, Tools(maker=held))
    seed_review(ui, tools)
    nf(ui, f"/videos/{V1}/packaging/variants")
    assert entered.wait(5)
    html = page(ui)
    assert 'id="variants-msg"' in card(html) and 'data-poll="1"' in html
    assert ui.client.get("/api/status").json()["variants"]["status"] == "running"
    nf(ui, f"/videos/{V1}/packaging/variants")
    assert "already being made" in page(ui)
    gate.set()
    ui.web.variants_job.wait()
    assert ui.web.variants_job.snapshot().status == "done"


def test_the_status_api_lists_both_jobs(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    idle = {"status": "idle", "message": "", "error_kind": ""}
    assert status["packaging"] == idle and status["variants"] == idle
    run_check(ui)
    run_variants(ui)
    status = ui.client.get("/api/status").json()
    assert status["packaging"]["status"] == "done" and status["variants"]["status"] == "done"


def test_the_polling_script_watches_both_jobs() -> None:
    js = (Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert '"packaging"' in js and '"variants"' in js


# --- the picture route ----------------------------------------------------------------------------
def with_variants(tmp_path: Path, **kwargs: Any) -> tuple[Any, Any]:
    ui, tools, _ = make_pk_ui(tmp_path, **kwargs)
    seed_variants(ui, tools, seed_review(ui, tools))
    return ui, tools


def test_a_variant_is_served_as_a_jpeg(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path)
    resp = ui.client.get(f"/videos/{V1}/packaging/variants/1.jpg")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
    assert "content-disposition" not in resp.headers
    with Image.open(io.BytesIO(resp.content)) as image:
        assert image.format == "JPEG" and image.size == (1280, 720)


def test_the_download_link_sends_an_attachment(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path)
    resp = ui.client.get(f"/videos/{V1}/packaging/variants/2.jpg?download=1")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
    assert resp.headers["content-disposition"] == f'attachment; filename="{V1}-variant-2.jpg"'
    plain = ui.client.get(f"/videos/{V1}/packaging/variants/2.jpg?download=0")
    assert "content-disposition" not in plain.headers


def test_pictures_need_the_browser_session(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path)
    bare = make_ui(tmp_path / "bare", signed_in_browser=False).client
    assert bare.get(f"/videos/{V1}/packaging/variants/1.jpg").status_code == 403


def test_a_variant_of_another_video_is_a_404(tmp_path: Path) -> None:
    ui, tools = with_variants(tmp_path)
    other = Tools([reading(), critique(), reading("REPOT IT"), reading("cactus guide")])
    seed_variants(ui, other, seed_review(ui, other, V2))  # variants 3 and 4 belong to V2
    assert ui.client.get(f"/videos/{V2}/packaging/variants/3.jpg").status_code == 200
    assert ui.client.get(f"/videos/{V1}/packaging/variants/3.jpg").status_code == 404
    assert ui.client.get(f"/videos/{V2}/packaging/variants/1.jpg").status_code == 404
    assert ui.client.get(f"/videos/{V2}/packaging/variants/1.jpg?download=1").status_code == 404


@pytest.mark.parametrize("variant", ["0", "99", "-1", "abc", "1.5", "01x", "١", "9" * 13,
                                     "99999999999999999999999", "%20", "1%20"])
def test_a_bad_variant_id_is_a_404(tmp_path: Path, variant: str) -> None:
    ui, _ = with_variants(tmp_path)
    assert ui.client.get(f"/videos/{V1}/packaging/variants/{variant}.jpg").status_code == 404


@pytest.mark.parametrize("path", [
    f"/videos/{V1}/packaging/variants/..%2f1.jpg",
    f"/videos/{V1}/packaging/variants/%2e%2e%2f%2e%2e%2fcatalog.sqlite.jpg",
    f"/videos/{V1}/packaging/variants/..%5c1.jpg",
    f"/videos/{V1}/packaging/variants/1.jpg/../../x.jpg",
    f"/videos/..%2f..%2f{V1}/packaging/variants/1.jpg",
    f"/videos/..%5c{V1}/packaging/variants/1.jpg",
    f"/videos/{V1}/packaging/variants/thumb-abc.jpg",
    f"/videos/{V1}/packaging/variants/%00.jpg",
])
def test_path_tricks_never_reach_a_file(tmp_path: Path, path: str) -> None:
    ui, _ = with_variants(tmp_path)
    assert ui.client.get(path).status_code == 404


def test_a_file_name_that_is_a_path_in_the_database_is_never_served(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path)
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(picture())
    with sqlite3.connect(ui.config.catalog_db) as conn:
        conn.execute("UPDATE thumbnail_variants SET image_file = ? WHERE id = 1",
                     ("../../../secret.jpg",))
        conn.execute("UPDATE thumbnail_variants SET image_file = ? WHERE id = 2", (str(secret),))
    conn.close()
    assert ui.client.get(f"/videos/{V1}/packaging/variants/1.jpg").status_code == 404
    assert ui.client.get(f"/videos/{V1}/packaging/variants/2.jpg").status_code == 404


def test_a_missing_file_is_a_404(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path)
    for path in (ui.web.images_dir / V1).glob("variant-*.jpg"):
        path.unlink()
    assert ui.client.get(f"/videos/{V1}/packaging/variants/1.jpg").status_code == 404


# --- the preview -----------------------------------------------------------------------------------
def desktop(html: str) -> str:
    return html.split('id="preview-desktop"')[1].split("</section>")[0]


def titles(section: str) -> list[str]:
    return re.findall(r'<h3 class="yt-title">(?:<span class="visually-hidden">Your video: </span>)?'
                      r'(.*?)</h3>', section)


def test_the_preview_shows_the_video_among_the_top_videos_by_views(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, others=9)
    with Store(ui.config.catalog_db) as store:  # a private and an unlisted video are never shown
        pull = store.begin_pull("UC_fake_channel")
        store.upsert_video(make_video(30, video_id="vid00000030", title="Private one",
                                      privacy_status="private", view_count=99999), pull)
        store.upsert_video(make_video(31, video_id="vid00000031", title="Unlisted one",
                                      privacy_status="unlisted", view_count=88888), pull)
        store.finish_pull(pull, 2)
    resp = ui.client.get(f"/videos/{V1}/packaging")
    assert resp.status_code == 200
    html = resp.text
    assert "<h1" in html and "Preview in a feed" in html
    mine = "Cactus repotting guide"
    assert titles(desktop(html)) == ["Other 1", mine, "Other 5", "Other 3", "Other 7", "Other 8", "Other 4",
                                     "Other 0"]
    phone = html.split('id="preview-phone"')[1].split("</section>")[0]
    assert titles(phone) == ["Other 1", mine, "Other 5", "Other 3"]
    search = html.split('id="preview-search"')[1].split("</section>")[0]
    assert titles(search) == [mine, "Other 1", "Other 5", "Other 3"]
    assert "Private one" not in html and "Unlisted one" not in html and "Second video" not in html
    assert "9K views" in desktop(html) and "8K views" in desktop(html)
    assert f'href="/videos/{V1}#thumbnail"' in html


def test_the_preview_uses_the_current_thumbnails_and_the_duration_badge(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, others=2)
    with Store(ui.config.catalog_db) as store:
        pull = store.begin_pull("UC_fake_channel")
        store.upsert_video(make_video(12, video_id="vid00000012", title="No length", duration="",
                                      privacy_status="public", view_count=1), pull)
        store.finish_pull(pull, 1)
    html = ui.client.get(f"/videos/{V1}/packaging").text
    cards = re.findall(r'<article class="yt-card.*?</article>', desktop(html), re.S)
    mine = next(c for c in cards if "Cactus repotting guide" in c)
    assert f'src="{MAXRES}"' in mine and '<span class="yt-badge">1:02:03</span>' in mine
    assert 'class="yt-card mine"' in mine
    plain = next(c for c in cards if "Other 0" in c)
    assert 'src="https://i.ytimg.com/vi/vid000010/mqdefault.jpg"' in plain
    assert '<span class="yt-badge">5:30</span>' in plain
    assert "yt-badge" not in next(c for c in cards if "No length" in c)
    assert 'id="preview-note"' in html and "Showing the thumbnail YouTube shows now." in html
    assert "Variant 1" not in html and "<script>" not in html and "style=" not in html


def test_the_preview_alone_says_so(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        pull = store.begin_pull("UC_fake_channel")
        store.upsert_video(make_video(2, video_id=V2, privacy_status="private"), pull)
        store.finish_pull(pull, 1)
    html = ui.client.get(f"/videos/{V1}/packaging").text
    assert 'id="preview-alone"' in html and titles(desktop(html)) == ["Cactus repotting guide"]


def test_the_preview_shows_a_chosen_variant_and_a_picker(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path, others=3)
    html = ui.client.get(f"/videos/{V1}/packaging?variant=2").text
    mine = re.search(r'<article class="yt-card mine".*?</article>', desktop(html), re.S).group(0)
    assert f'src="/videos/{V1}/packaging/variants/2.jpg"' in mine
    assert f'href="/videos/{V1}/packaging">Current thumbnail<' in html
    assert re.search(rf'class="chip active" href="/videos/{V1}/packaging\?variant=2" aria-current="page">'
                     r"Variant 2<", html)
    assert f'href="/videos/{V1}/packaging?variant=1">Variant 1<' in html
    assert AI_NOTE in html and "Yezatas does not upload thumbnails" in html
    search_mine = html.split('id="preview-search-mine"')[1].split("</article>")[0]
    assert f'/videos/{V1}/packaging/variants/2.jpg' in search_mine
    plain = ui.client.get(f"/videos/{V1}/packaging").text
    assert 'class="chip active" href="/videos/%s/packaging" aria-current="page">Current thumbnail<' % V1 in plain
    assert "Variant 2" in plain  # the picker lists the variants of the newest check


@pytest.mark.parametrize("variant", ["99", "abc", "-1", "0", "", "3", "%00", "9" * 30])
def test_a_bad_or_foreign_variant_falls_back_with_a_note(tmp_path: Path, variant: str) -> None:
    ui, _ = with_variants(tmp_path)
    other = Tools([reading(), critique(), reading("REPOT IT"), reading("cactus guide")])
    seed_variants(ui, other, seed_review(ui, other, V2))  # 3 and 4 belong to V2
    resp = ui.client.get(f"/videos/{V1}/packaging?variant={variant}")
    html = resp.text
    assert resp.status_code == 200
    assert "That variant is not available for this video, so the current thumbnail is shown." in html
    assert f"/packaging/variants/{variant}.jpg" not in html
    assert f'src="{MAXRES}"' in html
    assert re.search(r'class="chip active" href="/videos/%s/packaging" aria-current' % V1, html)


def test_a_failed_variant_is_not_previewed(tmp_path: Path) -> None:
    tools = Tools([reading(), critique(), reading("cactus guide")],
                  FakeMaker([ModelError("api", "The image maker did not send a picture. Try again."),
                             picture("PNG", (512, 288))]))
    ui, _, _ = make_pk_ui(tmp_path, tools)
    seed_variants(ui, tools, seed_review(ui, tools))
    html = ui.client.get(f"/videos/{V1}/packaging?variant=1").text
    assert "That variant is not available" in html and "variants/1.jpg" not in html
    assert f'href="/videos/{V1}/packaging?variant=2">Variant 1<' in html  # only the made picture is offered


def test_the_preview_uses_the_draft_title(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path, others=1)
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "Draft cactus title", DESCRIPTION, [])
    html = ui.client.get(f"/videos/{V1}/packaging").text
    assert "Draft cactus title" in desktop(html) and "The title is the one in your draft." in html


def test_the_preview_of_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, _, _ = make_pk_ui(tmp_path)
    assert ui.client.get("/videos/nosuchvideo/packaging").status_code == 404


def test_the_preview_calls_no_model_and_downloads_nothing(tmp_path: Path) -> None:
    ui, tools, fetch = make_pk_ui(tmp_path, others=3)
    ui.client.get(f"/videos/{V1}/packaging")
    assert tools.vision.calls == [] and tools.maker.prompts == [] and fetch.urls == []


def test_the_preview_without_a_catalog_goes_to_setup(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, packaging_tools=Tools(), packaging_fetch=Fetch())
    resp = ui.client.get(f"/videos/{V1}/packaging", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"


# --- nothing is uploaded --------------------------------------------------------------------------
def test_there_is_no_upload_button_or_form_on_either_page(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path, others=2)
    run_check(ui)
    video_page = page(ui)
    section = card(video_page)
    actions = set(re.findall(r'<form[^>]*action="([^"]*)"', section))
    assert actions == {f"/videos/{V1}/packaging", "/packaging/done",
                       f"/videos/{V1}/packaging/variants"} | ({"/packaging/variants/done"}
                                                              & actions)
    assert 'type="file"' not in section
    preview = ui.client.get(f"/videos/{V1}/packaging?variant=1").text
    assert set(re.findall(r'<form[^>]*action="([^"]*)"', preview)) == {"/videos"}  # only the search box
    for html in (section, preview):
        lowered = html.lower()
        for phrase in ("set as thumbnail", "use this thumbnail", "upload to youtube", "publish",
                       "thumbnails.set"):
            assert phrase not in lowered
    assert section.count(KEEP_SENTENCE) == 1


def test_no_key_or_file_path_reaches_the_pages(tmp_path: Path) -> None:
    ui, _ = with_variants(tmp_path, others=1)
    agent.save_api_key(ui.config, "FAKE-META-KEY")
    html = page(ui) + ui.client.get(f"/videos/{V1}/packaging?variant=1").text
    assert "FAKE-META-KEY" not in html
    assert str(ui.web.images_dir) not in html
    assert not re.search(r"(variant|thumb)-[0-9a-f]{12}", html)  # file names are never shown, only ids


# --- small pieces ----------------------------------------------------------------------------------
def test_format_duration() -> None:
    assert views.format_duration(None) == "" and views.format_duration(0) == ""
    assert views.format_duration(7) == "0:07" and views.format_duration(247) == "4:07"
    assert views.format_duration(3723) == "1:02:03" and views.format_duration(36000) == "10:00:00"


def test_top_other_videos_orders_by_views_then_id() -> None:
    catalog = [make_video(n, video_id=f"vid{n:08d}", view_count=v, privacy_status="public")
               for n, v in ((1, 10), (2, 50), (3, 50), (4, 5))]
    catalog.append(make_video(5, video_id="vid00000005", view_count=999, privacy_status="private"))
    ids = [v.video_id for v in views.top_other_videos("vid00000002", catalog)]
    assert ids == ["vid00000003", "vid00000001", "vid00000004"]
    assert len(views.top_other_videos("none", catalog, limit=2)) == 2
