"""Translated subtitles in the web UI: the block in the transcript card, the subtitles job, the result
rows and the .srt download. The translator and the checker are fakes; nothing touches the network or
Ollama, and nothing is ever sent to YouTube."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from webhelpers import make_ui, make_video, seed_catalog, seed_transcript
from yezatas import agent, translate
from yezatas.store import Store
from yezatas.translate import Completion, TranslateError

V1, V2 = "vidsb0001", "vidsb0002"
TEXTS = ["Hello and welcome.", "Today the plant costs 610 dollars.", "Thanks for watching!"]


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)
    monkeypatch.delenv(translate.TRANSLATOR_KEY_ENV, raising=False)


def tags(messages: list[dict[str, str]]) -> list[tuple[int, str]]:
    return [(int(n), t) for n, t in re.findall(r'<s n="(\d+)">(.*?)</s>', messages[-1]["content"],
                                               re.DOTALL)]


def tagged(items: list[tuple[int, str]]) -> str:
    return "\n".join(f'<s n="{n}">{t}</s>' for n, t in items)


class FakeModel:
    """A translator: every segment keeps its words behind a "~". `script` items (a str, an exception
    or a callable(messages)) are used first, one per call (use `by_language` when the order of the
    calls is not fixed); `gate` holds every call until it opens."""

    def __init__(self, script: list[Any] | None = None, gate: threading.Event | None = None,
                 model: str = "spark-test", by_language: dict[str, Any] | None = None) -> None:
        self.model = model
        self.script = list(script or [])
        self.by_language = dict(by_language or {})  # target code -> the reply for that language
        self.gate = gate
        self.entered = threading.Event()
        self.calls: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        with self._lock:
            self.calls.append(messages)
            item = self.script.pop(0) if self.script else None
            for code, reply in self.by_language.items():
                if f"into {translate.LANGUAGES[code]}." in messages[0]["content"]:
                    item = reply
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if item is None:
            item = tagged([(n, "~" + t) for n, t in tags(messages)])
        return Completion(item, 10, 20)


class FakeChecker:
    """A checker: translates back by removing the "~" and scores 5 unless `verdicts` (one per
    comparison, in order) says otherwise. `up=False` makes it unavailable."""

    def __init__(self, verdicts: list[Any] | None = None, up: bool = True,
                 model: str = "glimmer-test") -> None:
        self.model = model
        self.up = up
        self.verdicts = list(verdicts or [])
        self.calls: list[Any] = []

    def available(self) -> bool:
        return self.up

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> Any:
        self.calls.append(messages)
        if schema is None:
            return tagged([(n, t.removeprefix("~")) for n, t in tags(messages)])
        item = self.verdicts.pop(0) if self.verdicts else {"score": 5, "issues": []}
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)


class Tools:
    """The injected `translate_tools`."""

    def __init__(self, translator: Any = "default", checker: Any = "default") -> None:
        self.translator = FakeModel() if translator == "default" else translator
        self.checker = FakeChecker() if checker == "default" else checker

    def __call__(self, config: Any) -> Any:
        return self.translator, self.checker


def make_sb_ui(tmp_path: Path, tools: Tools | None = None, *, transcripts: bool = True,
               language: str = "en", **kwargs: Any):
    tools = tools or Tools()
    ui = make_ui(tmp_path, translate_tools=tools, **kwargs)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus repotting", description="Plain.", tags=[],
                   privacy_status="public", default_language="en"),
        make_video(2, video_id=V2, title="Second video", description="Plain.", tags=[],
                   privacy_status="public", default_language="en"),
    ])
    if transcripts:
        seed_transcript(ui.config, V1, segments=3, texts=TEXTS, language=language)
        seed_transcript(ui.config, V2, segments=3, texts=TEXTS, language=language)
    return ui, tools


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any, video_id: str = V1) -> str:
    return ui.client.get(f"/videos/{video_id}").text


def block(html: str) -> str:
    """The subtitles block of the transcript card, up to the end of the card."""
    return html.split('id="subtitles"')[1].split("</section>")[0]


def row(html: str, lang: str) -> str:
    rest = html.split(f'id="sub-{lang}"')[1]
    return re.split(r'<li class="tr-item"|</ul>\s*<p class="hint">The numbers', rest)[0]


def run_subtitles(ui: Any, targets: list[str] | None = None, video_id: str = V1) -> Any:
    resp = nf(ui, f"/videos/{video_id}/subtitles", data={"targets": targets or ["es", "de"]})
    ui.web.subtitles_job.wait()
    return resp


def stored(ui: Any, video_id: str = V1) -> dict[str, Any]:
    with Store(ui.config.catalog_db) as store:
        transcript = store.latest_transcript(video_id)
        assert transcript is not None
        return store.latest_subtitle_translations(video_id, transcript.id)


# --- the block ----------------------------------------------------------------------------------
def test_the_block_is_hidden_without_a_transcript(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, transcripts=False)
    html = page(ui)
    assert 'id="subtitles-need-transcript"' in html and 'id="subtitles-form"' not in html
    assert nf(ui, f"/videos/{V1}/subtitles", data={"targets": ["es"]}).status_code == 303
    assert "no transcript" in page(ui) and ui.web.subtitles_job.snapshot().status == "idle"


def test_the_block_sits_under_chapters_and_says_nothing_is_uploaded(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    html = page(ui)
    section = block(html)
    assert html.index("Chapters and subtitles") < html.index('id="subtitles"')
    assert '<h2 id="h-subtitles">Subtitles in other languages</h2>' in html
    assert '<details class="tool" id="tool-subtitles" data-keep>' in html
    assert "Yezatas does not upload subtitles to YouTube" in section
    assert "YouTube Studio" in section and "every time is copied from the transcript" in section
    assert "spark-test translates" in section and "glimmer-test on this computer" in section
    assert f'action="/videos/{V1}/subtitles"' in section
    assert re.search(r'class="button primary">Translate subtitles<', section)
    assert "subtitles-no-key" not in section and "subtitles-no-checker" not in section


def test_default_languages_are_ticked_and_the_transcript_language_is_not_offered(
        tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, language="es")
    section = block(page(ui))
    for code in translate.DEFAULT_TARGETS:
        if code != "es":
            assert f'name="targets" value="{code}" checked' in section
    assert 'value="es"' not in section  # the spoken language is not a choice
    assert 'name="targets" value="it">' in section and 'value="it" checked' not in section
    assert "Translate the subtitles (Spanish) into" in section


def test_a_chinese_transcript_blocks_both_chinese_targets(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, language="zh")
    section = block(page(ui))
    assert 'value="zh-Hans"' not in section and 'value="zh-Hant"' not in section
    assert 'name="targets" value="en"' in section and "(Chinese)" in section
    resp = nf(ui, f"/videos/{V1}/subtitles", data={"targets": ["zh-Hans"]})
    assert resp.status_code == 303 and "language of the transcript (Chinese)" in page(ui)
    assert ui.web.subtitles_job.snapshot().status == "idle"


def test_without_a_translator_the_button_is_disabled_and_setup_is_linked(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(translator=None))
    section = block(page(ui))
    assert 'id="subtitles-no-key"' in section and 'href="/configuration#step-translate"' in section
    assert re.search(r'class="button primary" disabled>Translate subtitles<', section)


def test_without_the_checker_the_block_says_results_will_be_unverified(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(checker=FakeChecker(up=False)))
    section = block(page(ui))
    assert 'id="subtitles-no-checker"' in section and 'marked "Unverified"' in section
    assert re.search(r'class="button primary">Translate subtitles<', section)  # still possible


def test_the_page_has_no_inline_script_or_style(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui)
    html = page(ui)
    assert "<script>" not in html and "style=" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)


# --- starting the job ---------------------------------------------------------------------------
def test_translating_runs_the_job_and_stores_the_results(tmp_path: Path) -> None:
    ui, tools = make_sb_ui(tmp_path)
    resp = run_subtitles(ui, ["de", "es"])
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#subtitles"
    found = stored(ui)
    assert set(found) == {"es", "de"} and all(r.status == "verified" for r in found.values())
    assert len(tools.translator.calls) == 2 and all("into " in c[0]["content"] for c in tools.translator.calls)
    section = block(page(ui))
    assert 'id="subtitles-done"' in section
    assert "Subtitles finished: 2 verified." in section
    assert 'action="/subtitles/done"' in section
    assert section.index('id="sub-es"') < section.index('id="sub-de"')  # LANGUAGES order
    es = row(section, "es")
    assert "Spanish" in es and 'class="pill done">Verified' in es and "Score 5/5" in es
    assert f'href="/videos/{V1}/subtitles/es.srt"' in es and "Download .srt" in es
    assert "spark-test, checked by glimmer-test" in es and "20 out" in es
    assert "Back-translation" in es and "1: Hello and welcome." in es
    assert 'id="subtitles-done"' not in block(page(ui, V2))  # only on its own video


def test_the_boxes_start_ticked_for_the_languages_that_have_a_result(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui, ["it"])
    section = block(page(ui))
    assert 'name="targets" value="it" checked' in section
    assert 'name="targets" value="es" checked' not in section


def test_translating_again_replaces_the_earlier_result(tmp_path: Path) -> None:
    ui, tools = make_sb_ui(tmp_path)
    run_subtitles(ui, ["es"])
    first = stored(ui)["es"].id
    run_subtitles(ui, ["es"])
    assert stored(ui)["es"].id > first and len(tools.translator.calls) == 2
    assert block(page(ui)).count('id="sub-es"') == 1


def test_the_finished_job_is_dismissed_with_the_button(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui, ["es"])
    assert 'id="subtitles-done"' in page(ui)
    resp = nf(ui, "/subtitles/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#subtitles"
    html = page(ui)
    assert 'id="subtitles-done"' not in html and 'id="sub-es"' in html  # the results stay
    assert nf(ui, "/subtitles/done", data={}).headers["location"] == "/"


def test_a_failed_language_has_its_sentence_and_no_download(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(FakeModel(by_language={"fr": "no tags here"})))
    run_subtitles(ui, ["es", "fr"])
    section = block(page(ui))
    fr = row(section, "fr")
    assert 'class="pill problem">Failed' in fr and "Download .srt" not in fr
    assert "The translator&#39;s answer is missing segments 1, 2, 3." in fr
    assert "Download .srt" in row(section, "es")
    assert "Subtitles finished: 1 verified, 1 failed." in section
    assert 'class="flash ok"' in section


def test_all_languages_failing_shows_an_error_flash(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(FakeModel(script=["x", "x"])))
    run_subtitles(ui, ["es"])
    section = block(page(ui))
    assert 'class="flash error" role="status" id="subtitles-done"' in section
    assert "Download .srt" not in section


def test_a_refused_key_fails_the_job_with_a_dismiss_button_and_stores_nothing(
        tmp_path: Path) -> None:
    refused = TranslateError("auth", translate.AUTH_MESSAGE)
    ui, _ = make_sb_ui(tmp_path, Tools(FakeModel(script=[refused] * 4)))
    run_subtitles(ui, ["es"])
    snap = ui.web.subtitles_job.snapshot()
    assert snap.status == "failed" and snap.message == translate.AUTH_MESSAGE
    section = block(page(ui))
    assert 'id="subtitles-failed"' in section and translate.AUTH_MESSAGE in section
    assert 'action="/subtitles/done"' in section
    assert translate.AUTH_MESSAGE not in page(ui, V2)
    assert stored(ui) == {}
    nf(ui, "/subtitles/done", data={"video_id": V1})
    assert 'id="subtitles-failed"' not in page(ui)


def test_the_job_without_a_translator_fails_with_a_clear_message(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(translator=None))
    assert ui.web.start_subtitles(V1, ["es"], "sid")
    ui.web.subtitles_job.wait()
    snap = ui.web.subtitles_job.snapshot()
    assert snap.status == "failed"
    assert snap.message == "There is no translation key. Add one in Configuration. Nothing was sent."


def test_a_post_without_a_translator_flashes_and_starts_nothing(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(translator=None))
    nf(ui, f"/videos/{V1}/subtitles", data={"targets": ["es"]})
    html = page(ui)
    assert "There is no translation key yet." in html and "Nothing was sent." in html
    assert ui.web.subtitles_job.snapshot().status == "idle"


def test_a_crash_in_the_job_is_reported_without_details(tmp_path: Path) -> None:
    def boom(config: Any) -> Any:
        raise RuntimeError("secret detail")

    ui, _ = make_sb_ui(tmp_path, boom)
    assert ui.web.start_subtitles(V1, ["es"], "sid")
    ui.web.subtitles_job.wait()
    snap = ui.web.subtitles_job.snapshot()
    assert snap.status == "failed" and "secret detail" not in snap.message
    assert "stopped unexpectedly (RuntimeError)" in snap.message


def test_a_job_the_core_refuses_shows_its_plain_sentence(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, transcripts=False)
    assert ui.web.start_subtitles(V1, ["es"], "sid")
    ui.web.subtitles_job.wait()
    assert ui.web.subtitles_job.snapshot().message == (
        "This video has no transcript yet. Transcribe it first.")


# --- validation ---------------------------------------------------------------------------------
def test_invalid_form_values_are_refused_with_a_flash_and_nothing_starts(tmp_path: Path) -> None:
    ui, tools = make_sb_ui(tmp_path)
    cases = [
        ({}, "Pick at least one language"),
        ({"targets": ["es", "klingon"]}, "not available"),
        ({"targets": ["es", "en"]}, "language of the transcript (English)"),
        ({"targets": ["en"]}, "Untick it"),
    ]
    for data, message in cases:
        resp = nf(ui, f"/videos/{V1}/subtitles", data=data)
        assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#subtitles"
        html = page(ui)
        assert 'class="flash error"' in html and message in html, message
    assert tools.translator.calls == [] and ui.web.subtitles_job.snapshot().status == "idle"


def test_a_regional_transcript_language_blocks_its_base_language(tmp_path: Path) -> None:
    ui, tools = make_sb_ui(tmp_path, language="en-GB")
    assert 'name="targets" value="en"' not in block(page(ui))
    nf(ui, f"/videos/{V1}/subtitles", data={"targets": ["en"]})
    assert "Untick it" in page(ui) and tools.translator.calls == []


def test_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    assert nf(ui, "/videos/nosuchvideo/subtitles", data={"targets": ["es"]}).status_code == 404


# --- busy ---------------------------------------------------------------------------------------
def test_a_running_job_shows_progress_polls_and_blocks_a_second_start(tmp_path: Path) -> None:
    gate = threading.Event()
    model = FakeModel(gate=gate)
    ui, _ = make_sb_ui(tmp_path, Tools(model))
    try:
        nf(ui, f"/videos/{V1}/subtitles", data={"targets": ["es"]})
        assert model.entered.wait(10)
        html = page(ui)
        section = block(html)
        assert 'id="subtitles-msg"' in section and "Translating subtitles into 1 language" in section
        assert 'class="spinner"' in section and 'name="targets"' not in section
        assert 'data-poll="1"' in html
        assert ui.client.get("/api/status").json()["subtitles"]["status"] == "running"
        other = page(ui, V2)
        assert 'id="subtitles-busy"' in block(other) and 'data-poll="1"' in other
        assert re.search(r'class="button primary" disabled>Translate subtitles<', block(other))
        nf(ui, f"/videos/{V2}/subtitles", data={"targets": ["de"]})
        assert "already being translated" in page(ui, V2)
        assert len(model.calls) == 1
    finally:
        gate.set()
        ui.web.subtitles_job.wait()
    assert ui.client.get("/api/status").json()["subtitles"]["status"] == "done"
    assert set(stored(ui)) == {"es"} and stored(ui, V2) == {}


def test_the_button_is_off_while_the_video_is_being_transcribed(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    gate = threading.Event()
    entered = threading.Event()

    def slow(report: Any) -> Any:
        entered.set()
        gate.wait(10)

    assert ui.web.transcribe_job.start(slow)
    ui.web._transcribe_video = V1
    try:
        assert entered.wait(10)
        assert re.search(r'class="button primary" disabled>Translate subtitles<',
                         block(page(ui)))
    finally:
        gate.set()
        ui.web.transcribe_job.wait()


def test_the_subtitles_job_does_not_block_other_jobs(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    assert ui.web.subtitles_job is not ui.web.translate_job
    assert ui.web.subtitles_job is not ui.web.transcribe_job


# --- results ------------------------------------------------------------------------------------
def test_a_flagged_row_shows_the_code_problems_and_the_checkers_issues(tmp_path: Path) -> None:
    def drop_digits(messages: list[dict[str, str]]) -> str:
        return tagged([(n, "~" + re.sub(r"\d", "", t)) for n, t in tags(messages)])

    checker = FakeChecker(verdicts=[{"score": 5, "issues": []},
                                    {"score": 3, "issues": ["Segment 2 lost the price."]}])
    ui, _ = make_sb_ui(tmp_path, Tools(FakeModel(by_language={"de": drop_digits}), checker))
    run_subtitles(ui, ["es", "de"])
    section = block(page(ui))
    es, de = row(section, "es"), row(section, "de")
    assert 'class="pill done">Verified' in es and "Problem:" not in es
    assert 'class="pill warn">Flagged' in de and "Score 3/5" in de
    assert '<strong>Problem:</strong> The number &#34;610&#34; in segment 2 is missing' in de
    assert "<strong>Checker:</strong> Segment 2 lost the price." in de
    assert "Download .srt" in de  # flagged ones can be downloaded after a careful read
    assert "careful read" in section


def test_an_unverified_row_shows_the_checkers_note(tmp_path: Path) -> None:
    checker = FakeChecker(verdicts=[TranslateError("answer", "The checker sent no back-translation.")])
    ui, _ = make_sb_ui(tmp_path, Tools(checker=checker))
    run_subtitles(ui, ["es"])
    es = row(block(page(ui)), "es")
    assert 'class="pill unverified">Unverified' in es
    assert "The checker could not check this one: The checker sent no back-translation." in es
    assert "Score" not in es and "Back-translation" not in es


def test_results_without_the_checker_are_unverified_with_no_score(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(checker=FakeChecker(up=False)))
    run_subtitles(ui, ["es"])
    section = block(page(ui))
    es = row(section, "es")
    assert "Unverified" in es and "Score" not in es and "Download .srt" in es
    assert "(the checker was not available)" in section


# --- the download -------------------------------------------------------------------------------
def test_the_srt_has_the_transcripts_times_and_the_right_headers(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui, ["es"])
    resp = ui.client.get(f"/videos/{V1}/subtitles/es.srt")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-subrip; charset=utf-8"
    assert resp.headers["content-disposition"] == f'attachment; filename="{V1}.es.srt"'
    assert resp.text == ("1\n00:00:00,000 --> 00:00:50,000\n~Hello and welcome.\n\n"
                         "2\n00:00:50,000 --> 00:01:40,000\n~Today the plant costs 610 dollars.\n\n"
                         "3\n00:01:40,000 --> 00:02:30,000\n~Thanks for watching!\n")
    # the original-language file is untouched
    assert "~" not in ui.client.get(f"/videos/{V1}/transcript.srt").text


def test_a_regional_language_code_downloads_by_its_own_name(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui, ["zh-Hans"])
    resp = ui.client.get(f"/videos/{V1}/subtitles/zh-Hans.srt")
    assert resp.status_code == 200
    assert resp.headers["content-disposition"] == f'attachment; filename="{V1}.zh-Hans.srt"'
    assert f'href="/videos/{V1}/subtitles/zh-Hans.srt"' in page(ui)


def test_the_srt_is_a_404_for_failed_unknown_and_missing(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, Tools(FakeModel(script=["x", "x"])))
    assert ui.client.get(f"/videos/{V1}/subtitles/es.srt").status_code == 404  # nothing yet
    run_subtitles(ui, ["es"])  # failed
    assert stored(ui)["es"].status == "failed"
    assert ui.client.get(f"/videos/{V1}/subtitles/es.srt").status_code == 404
    assert ui.client.get(f"/videos/{V1}/subtitles/de.srt").status_code == 404  # never translated
    assert ui.client.get(f"/videos/{V1}/subtitles/klingon.srt").status_code == 404
    assert ui.client.get(f"/videos/{V1}/subtitles/..%2F..%2Fx.srt").status_code == 404
    assert ui.client.get("/videos/nosuchvideo/subtitles/es.srt").status_code == 404
    assert ui.client.get(f"/videos/{V2}/subtitles/es.srt").status_code == 404  # another video


def test_subtitles_of_an_older_transcript_are_not_served_or_shown(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    run_subtitles(ui, ["es"])
    assert ui.client.get(f"/videos/{V1}/subtitles/es.srt").status_code == 200
    seed_transcript(ui.config, V1, segments=1, texts=["A newer transcript."])
    assert ui.client.get(f"/videos/{V1}/subtitles/es.srt").status_code == 404
    section = block(page(ui))
    assert 'id="sub-es"' not in section and "Download .srt" not in section
    assert 'name="targets" value="es" checked' in section  # back to the defaults


def test_the_srt_needs_a_signed_in_browser(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path, signed_in_browser=False)
    resp = ui.client.get(f"/videos/{V1}/subtitles/es.srt")
    assert resp.status_code in (401, 403) and "Hello" not in resp.text


# --- status, safety, script ---------------------------------------------------------------------
def test_the_status_api_reports_the_subtitles_job(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    assert ui.client.get("/api/status").json()["subtitles"]["status"] == "idle"
    run_subtitles(ui, ["es"])
    status = ui.client.get("/api/status").json()["subtitles"]
    assert status["status"] == "done" and status["error_kind"] in ("", None)


@pytest.mark.parametrize("path", [f"/videos/{V1}/subtitles", "/subtitles/done"])
def test_every_new_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, tools = make_sb_ui(tmp_path)
    data = {"targets": ["es"], "video_id": V1}
    assert ui.client.post(path, data=data).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong", **data}).status_code == 403
    assert tools.translator.calls == [] and ui.web.subtitles_job.snapshot().status == "idle"


def test_nothing_in_the_subtitles_flow_reaches_youtube(tmp_path: Path) -> None:
    from fakes import FakeYouTubeWriter

    writer = FakeYouTubeWriter(0)
    ui, _ = make_sb_ui(tmp_path, writer_factory=lambda creds: writer)
    run_subtitles(ui, ["es", "de"])
    ui.client.get(f"/videos/{V1}/subtitles/es.srt")
    assert writer.calls == []
    with Store(ui.config.catalog_db) as store:
        assert store.list_pushes() == [] and store.get_draft(V1) is None


def test_no_key_appears_in_the_page(tmp_path: Path) -> None:
    ui, _ = make_sb_ui(tmp_path)
    agent.save_api_key(ui.config, "FAKE-META-KEY")
    run_subtitles(ui, ["es"])
    html = page(ui)
    assert "FAKE-META-KEY" not in html
    assert "through Meta's Model API" in block(html)


def test_the_polling_script_watches_the_subtitles_job() -> None:
    js = (Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert '"subtitles"' in js
