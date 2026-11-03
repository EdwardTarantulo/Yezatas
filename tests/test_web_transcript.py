"""Transcripts and chapters in the web UI: the video page card, the transcribe and chapters jobs,
the SRT download and the Setup line. Transcribing and the model are fakes; nothing touches the
network or ffmpeg."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeModelClient
from webhelpers import make_ui, make_video, seed_catalog, seed_transcript
from yezatas import agent, chapters, transcribe
from yezatas.agent import ModelReply
from yezatas.store import Store, TranscriptRecord
from yezatas.transcribe import TranscribeError, TranscribeResult

V1, V2, SHORT = "vidts0001", "vidts0002", "vidts0003"
DESCRIPTION = "Repotting a cactus."
FAKE_KEY = "FAKE-KEY-0123456789abcdef"


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: "ffmpeg")


class FakeTranscribe:
    """The injected `transcribe`: records the call, stores a transcript, or fails / waits."""

    def __init__(self, error: BaseException | None = None, gate: threading.Event | None = None,
                 source_name: str = "clip.mp4") -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error
        self.gate = gate
        self.source_name = source_name

    def __call__(self, store: Store, video_id: str, path: str, language: str, report: Any,
                 keywords: tuple[str, ...] = ()) -> TranscribeResult:
        self.calls.append({"video_id": video_id, "path": path, "language": language,
                           "keywords": tuple(keywords)})
        report("Reading the audio")
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        parts = [{"start_ms": i * 50_000, "end_ms": (i + 1) * 50_000,
                  "text": f"Piece {i} talks about topic {i}."} for i in range(6)]
        record = TranscriptRecord(0, video_id, language, "voice-test", Path(path).name, 300_000,
                                  parts, [], "")
        stored = store.get_transcript(store.add_transcript(record))
        assert stored is not None
        return TranscribeResult(stored, "Transcribed 5:00 of audio in 6 pieces.", 0.015)


def chapters_with(script: list[Any]) -> Any:
    client = FakeModelClient(script)

    def run(store: Store, video_id: str, report: Any) -> chapters.ChaptersResult:
        return chapters.suggest_chapters(store, client, video_id, report=report)

    run.client = client  # type: ignore[attr-defined]
    return run


def reply(chosen: list[tuple[int, str]]) -> ModelReply:
    content = json.dumps({"chapters": [{"segment": s, "title": t} for s, t in chosen],
                          "summary": "Divided by topic."})
    return ModelReply(content, (), {"role": "assistant", "content": content})


GOOD = [(0, "Intro"), (2, "Choosing a pot"), (4, "Aftercare")]


def make_ts_ui(tmp_path: Path, *, fake: Any = "default", chapters_fn: Any = None,
               language: str | None = "en", audio_language: str | None = None,
               description: str = DESCRIPTION, **kwargs: Any):
    fake = FakeTranscribe() if fake == "default" else fake
    ui = make_ui(tmp_path, transcribe=fake, chapters=chapters_fn, **kwargs)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus repotting", description=description, tags=["cactus"],
                   privacy_status="public", default_language=language,
                   default_audio_language=audio_language),
        make_video(2, video_id=V2, title="Second video", description="Plain text.", tags=[],
                   privacy_status="public"),
        make_video(3, video_id=SHORT, title="A short one", description="Plain.", tags=[],
                   privacy_status="public", duration="PT50S"),
    ])
    return ui, fake


def media(tmp_path: Path, name: str = "My Clip.mp4") -> Path:
    folder = tmp_path / "private-folder-xyz"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_bytes(b"not really a video")
    return path


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def card(html: str) -> str:
    return html.split('id="transcript"')[1].split("</section>")[0]


def page(ui: Any, video_id: str = V1) -> str:
    return ui.client.get(f"/videos/{video_id}").text


def start(ui: Any, path: Path | str, video_id: str = V1, language: str = "en", wait: bool = True,
          **extra: str) -> Any:
    resp = nf(ui, f"/videos/{video_id}/transcribe",
              data={"path": str(path), "language": language, **extra})
    if wait:
        ui.web.transcribe_job.wait()
    return resp


def store_of(ui: Any) -> Store:
    return Store(ui.config.catalog_db)


# --- the card -----------------------------------------------------------------------------------
def test_the_card_has_the_form_the_price_and_sits_after_translations(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    html = page(ui)
    section = card(html)
    assert '<h2 id="h-transcript">Transcript and chapters</h2>' in section
    assert f'action="/videos/{V1}/transcribe"' in section
    assert 'name="path"' in section and 'name="language"' in section and 'name="keywords"' in section
    assert "Price: about $0.18 per hour of audio." in section
    assert '<option value="en" selected>English</option>' in section
    assert '<select id="ts-language" name="language">' in section
    assert "Latest transcript" not in section and "Make chapters" not in section
    assert html.index('id="edit"') < html.index('id="translations"') < html.index('id="transcript"')
    assert "<script>" not in html and "style=" not in html and "onclick" not in html


def test_the_card_is_clear_that_only_sound_is_sent(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    section = card(page(ui))
    assert "only the sound is sent" in section
    assert "Nothing goes to YouTube until you approve and push it." in section


def test_without_ffmpeg_the_card_explains_and_has_no_form(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    ui, _ = make_ts_ui(tmp_path)
    section = card(page(ui))
    assert 'id="transcribe-no-ffmpeg"' in section and "ffmpeg was not found" in section
    assert "<form" not in section and 'name="path"' not in section


def test_without_ffmpeg_an_existing_transcript_is_still_shown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1)
    section = card(page(ui))
    assert 'id="transcribe-no-ffmpeg"' in section and 'id="transcript-text"' in section
    assert 'name="path"' not in section and 'id="download-srt"' in section


def test_a_missing_meta_key_is_a_hint_with_a_link_to_setup(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    section = card(page(ui))
    assert 'id="transcribe-no-key"' in section and 'href="/setup#step-ai"' in section
    agent.save_api_key(ui.config, FAKE_KEY)
    section = card(page(ui))
    assert 'id="transcribe-no-key"' not in section and FAKE_KEY not in section


@pytest.mark.parametrize("language, audio, expect", [
    ("en", None, "en"), ("de", None, "de"), ("en-GB", None, "en"), ("fr", "es", "es"),
    ("lt", "pt-BR", "pt"),
])
def test_the_spoken_language_is_preselected_from_the_video(tmp_path: Path, language: str,
                                                           audio: str | None, expect: str) -> None:
    ui, _ = make_ts_ui(tmp_path, language=language, audio_language=audio)
    section = card(page(ui))
    assert f'<option value="{expect}" selected>{transcribe.ASR_LANGUAGES[expect]}</option>' in section
    assert 'value="" selected' not in section and 'id="ts-language-hint"' not in section


def test_an_unsupported_video_language_is_said_and_never_guessed(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, language="lt")
    section = card(page(ui))
    assert 'id="ts-language-hint"' in section and "Lithuanian" in section
    assert "is not one transcription supports" in section
    assert '<select id="ts-language" name="language" required>' in section
    assert '<option value="" selected>Choose the spoken language</option>' in section
    assert " selected>English</option>" not in section


def test_a_video_without_a_language_asks_for_one(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, language=None)
    section = card(page(ui))
    assert "has no language set" in section and '<option value="" selected>' in section


def test_every_supported_language_is_offered(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    section = card(page(ui))
    for code, name in transcribe.ASR_LANGUAGES.items():
        assert f'<option value="{code}"' in section and f">{name}</option>" in section
    assert 'value="lt"' not in section


# --- a stored transcript ------------------------------------------------------------------------
def test_a_transcript_shows_its_facts_lines_problems_and_buttons(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1, source_name="Holiday clip.mp4",
                    problems=["Could not transcribe 1:40-1:50: The service is busy."])
    section = card(page(ui))
    assert "<strong>Holiday clip.mp4</strong>" in section
    assert "voice-test" in section and "5:00 of audio" in section and "cost about $0.015" in section
    assert "English" in section
    assert 'class="issue warning">Could not transcribe 1:40-1:50: The service is busy.' in section
    lines = re.search(r'<pre class="tx-lines">(.*?)</pre>', section, re.S)
    assert lines
    assert lines.group(1).splitlines()[:3] == [
        "0:00 Piece 0 talks about topic 0.", "0:50 Piece 1 talks about topic 1.",
        "1:40 Piece 2 talks about topic 2."]
    assert "<details" in section and "Read the transcript (6 lines)" in section
    assert re.search(r'class="button primary">Make chapters<', section)
    assert 'id="chapters-blocked"' not in section
    assert f'href="/videos/{V1}/transcript.srt"' in section and "Download subtitles (.srt)" in section
    assert ">Transcribe again<" in section


def test_empty_pieces_keep_their_numbering_out_of_the_lines(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1, texts=["One.", "", "Three.", "", "Five.", "Six."])
    section = card(page(ui))
    assert "Read the transcript (4 lines)" in section
    assert "1:40 Three." in section and "0:50" not in section.split('class="tx-lines"')[1]


def test_make_chapters_is_disabled_with_the_reason_on_a_short_video(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, SHORT, segments=2)
    section = card(page(ui, SHORT))
    assert re.search(r'class="button primary" disabled aria-describedby="chapters-blocked">Make chapters<',
                     section)
    assert 'id="chapters-blocked"' in section and "longer than a minute" in section


def test_make_chapters_is_disabled_when_the_description_has_timestamps(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, description="Intro.\n0:00 Start\n1:00 Middle\n2:00 End")
    seed_transcript(ui.config, V1)
    section = card(page(ui))
    assert "disabled" in section.split("Make chapters")[0].rsplit("<button", 1)[1]
    assert "already has timestamps" in section
    assert "will not show them" not in section


def test_timestamps_youtube_will_not_show_are_named_on_the_chapters_card(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, description="Intro.\n0:00 Start\n0:14 Box\n0:23 Look\n2:00 End")
    seed_transcript(ui.config, V1)
    section = card(page(ui))
    assert "disabled" in section.split("Make chapters")[0].rsplit("<button", 1)[1]
    assert "but YouTube will not show them as chapters" in section
    assert "1 chapter is shorter than 10 seconds: 0:14 (9 s)." in section


def test_a_draft_with_timestamps_blocks_chapters_even_if_the_video_has_none(tmp_path: Path) -> None:
    from yezatas import edits

    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1)
    with store_of(ui) as store:
        edits.save_draft(store, V1, "Cactus repotting", "Intro.\n0:00 A\n1:00 B\n2:00 C", ["cactus"])
    section = card(page(ui))
    assert "already has timestamps" in section and 'id="chapters-blocked"' in section


def test_a_video_without_a_draft_but_with_a_transcript_offers_chapters(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1)
    assert 'id="chapters-blocked"' not in card(page(ui))


# --- starting the transcription -----------------------------------------------------------------
def test_transcribing_runs_the_job_and_stores_the_transcript(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    path = media(tmp_path)
    resp = start(ui, f'  "{path}"  ', language="de", keywords="Opuntia, Echeveria ,, opuntia")
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#transcript"
    assert fake.calls == [{"video_id": V1, "path": str(path), "language": "de",
                           "keywords": ("Opuntia", "Echeveria")}]
    with store_of(ui) as store:
        record = store.latest_transcript(V1)
    assert record is not None and record.source_name == "My Clip.mp4" and record.language == "de"
    assert ui.web.transcribe_job.snapshot().status == "done"


def test_without_keywords_the_function_is_called_with_exactly_five_arguments(tmp_path: Path) -> None:
    seen: list[tuple[Any, ...]] = []

    def five(store: Store, video_id: str, path: str, language: str, report: Any) -> TranscribeResult:
        seen.append((video_id, path, language))
        raise TranscribeError("api", "Stopped for the test.")

    ui, _ = make_ts_ui(tmp_path, fake=five)
    path = media(tmp_path)
    start(ui, path)
    assert seen == [(V1, str(path), "en")]


def test_the_finished_job_shows_its_result_until_dismissed(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    start(ui, media(tmp_path))
    section = card(page(ui))
    assert 'id="transcribe-done"' in section
    assert "Transcribed 5:00 of audio in 6 pieces. The cost was about $0.015." in section
    assert 'id="transcript-text"' in section and 'id="download-srt"' in section
    other = card(page(ui, V2))
    assert 'id="transcribe-done"' not in other and "Latest transcript" not in other
    resp = nf(ui, "/transcribe/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#transcript"
    assert 'id="transcribe-done"' not in card(page(ui))


def test_the_status_api_reports_the_transcribe_and_chapters_jobs(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    status = ui.client.get("/api/status").json()
    assert status["transcribe"]["status"] == "idle" and status["chapters"]["status"] == "idle"
    start(ui, media(tmp_path))
    assert ui.client.get("/api/status").json()["transcribe"]["status"] == "done"


def test_a_running_job_shows_progress_and_polls(tmp_path: Path) -> None:
    gate = threading.Event()
    ui, fake = make_ts_ui(tmp_path, fake=FakeTranscribe(gate=gate))
    try:
        start(ui, media(tmp_path), wait=False)
        for _ in range(200):
            if fake.calls:
                break
            threading.Event().wait(0.01)
        assert ui.client.get("/api/status").json()["transcribe"]["status"] == "running"
        html = page(ui)
        section = card(html)
        assert 'id="transcribe-msg"' in section and "Reading the audio" in section
        assert 'class="spinner"' in section and 'name="path"' not in section
        assert 'data-poll="1"' in html
        busy = card(page(ui, V2))
        assert 'id="transcribe-busy"' in busy and 'name="path"' in busy
        again = start(ui, media(tmp_path), video_id=V2, wait=False)
        assert again.status_code == 303
        assert "A transcription is already running" in page(ui, V2)
        assert len(fake.calls) == 1
    finally:
        gate.set()
        ui.web.transcribe_job.wait()


def test_a_failed_job_shows_its_sentence_with_a_dismiss_button(tmp_path: Path) -> None:
    error = TranscribeError("auth", "The transcription API key was refused. Check it in Setup.")
    ui, _ = make_ts_ui(tmp_path, fake=FakeTranscribe(error=error))
    start(ui, media(tmp_path))
    snap = ui.web.transcribe_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth"
    html = page(ui)
    section = card(html)
    assert 'id="transcribe-failed"' in section and "The transcription API key was refused." in section
    assert 'action="/transcribe/done"' in section and 'name="path"' in section
    nf(ui, "/transcribe/done", data={"video_id": V1})
    assert 'id="transcribe-failed"' not in card(page(ui))


def test_an_unexpected_error_gives_a_generic_sentence_with_the_class_name_only(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    ui.web._transcribe = FakeTranscribe(error=KeyError("secret detail 123"))
    start(ui, media(tmp_path))
    message = ui.web.transcribe_job.snapshot().message
    assert message == "The transcription stopped unexpectedly (KeyError)."
    assert "secret detail" not in page(ui)


def test_a_value_error_sentence_passes_through(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, fake=FakeTranscribe(error=ValueError("Pull your videos first.")))
    start(ui, media(tmp_path))
    assert ui.web.transcribe_job.snapshot().message == "Pull your videos first."


def test_the_folder_of_the_file_is_taken_out_of_an_error_sentence(tmp_path: Path) -> None:
    path = media(tmp_path)

    def failing(store: Store, video_id: str, given: str, language: str, report: Any) -> Any:
        raise TranscribeError("ffmpeg", f"ffmpeg could not read the audio of this file: "
                                        f"Error opening input {given}: Invalid data")

    ui, _ = make_ts_ui(tmp_path, fake=failing)
    start(ui, path)
    html = page(ui)
    message = ui.web.transcribe_job.snapshot().message
    assert "My Clip.mp4" in message and str(path.parent) not in message
    assert str(path.parent) not in html and "private-folder-xyz" not in html


# --- refusals before any job --------------------------------------------------------------------
def test_a_missing_file_is_a_flash_not_a_job_and_the_path_is_not_echoed(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    ghost = tmp_path / "private-folder-xyz" / "ghost.mp4"
    resp = start(ui, ghost)
    assert resp.status_code == 303
    html = page(ui)
    assert "There is no file at that path (looked for ghost.mp4)." in html
    assert "Nothing was sent." in html
    assert "private-folder-xyz" not in html
    assert fake.calls == [] and ui.web.transcribe_job.snapshot().status == "idle"


def test_a_wrong_file_type_and_a_folder_are_refused(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    notes = media(tmp_path, "notes.txt")
    start(ui, notes)
    assert "notes.txt is not a video or audio file" in page(ui)
    start(ui, notes.parent)
    assert "That path is a folder, not a video file." in page(ui)
    start(ui, "   ")
    assert "Type the path of the video file" in page(ui)
    assert fake.calls == [] and ui.web.transcribe_job.snapshot().status == "idle"
    assert "private-folder-xyz" not in page(ui)


def test_a_very_long_path_is_refused(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    start(ui, "a" * 1500 + ".mp4")
    assert "too long to be a file path" in page(ui) and fake.calls == []


def test_an_unsupported_or_missing_language_is_refused(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    path = media(tmp_path)
    start(ui, path, language="lt")
    assert "Lithuanian and some other languages are not supported" in page(ui)
    start(ui, path, language="")
    assert "Choose the language that is spoken" in page(ui)
    assert fake.calls == []


def test_bad_keywords_are_refused(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    path = media(tmp_path)
    start(ui, path, keywords=",".join(f"name{i}" for i in range(21)))
    assert "at most 20 names" in page(ui)
    start(ui, path, keywords="x" * 41)
    assert "at most 40 characters" in page(ui)
    start(ui, path, keywords=",".join(f"name{i}" for i in range(20)))
    assert len(fake.calls) == 1 and len(fake.calls[0]["keywords"]) == 20


def test_without_ffmpeg_posting_is_refused_with_the_install_hint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    ui, fake = make_ts_ui(tmp_path)
    start(ui, media(tmp_path))
    assert "Install ffmpeg and restart Yezatas." in page(ui)
    assert fake.calls == [] and ui.web.transcribe_job.snapshot().status == "idle"


def test_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    resp = start(ui, media(tmp_path), video_id="nosuchvideo")
    assert resp.status_code == 404 and fake.calls == []


# --- the default functions ----------------------------------------------------------------------
def test_the_default_transcribe_without_a_key_stops_before_sending_anything(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    start(ui, media(tmp_path))
    snap = ui.web.transcribe_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth"
    assert snap.message == "No Meta API key is set. Add one in Setup. Nothing was sent."


def test_the_default_transcribe_gives_the_key_only_to_the_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(agent.KEY_ENV, FAKE_KEY)
    made: list[str] = []
    runs: list[dict[str, Any]] = []

    class FakeClient:
        model = "voice-test"

        def __init__(self, key: str) -> None:
            made.append(key)

    def fake_run(store: Store, client: Any, video_id: str, path: Any, language: str, **kw: Any) -> Any:
        runs.append({"client": client, "video_id": video_id, "language": language, **kw})
        raise TranscribeError("api", "Stopped for the test.")

    monkeypatch.setattr(transcribe, "MetaAsrClient", FakeClient)
    monkeypatch.setattr(transcribe, "transcribe_video", fake_run)
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    start(ui, media(tmp_path), keywords="Opuntia")
    assert made == [FAKE_KEY]
    assert runs[0]["video_id"] == V1 and runs[0]["keywords"] == ("Opuntia",)
    assert FAKE_KEY not in ui.web.transcribe_job.snapshot().message
    assert FAKE_KEY not in page(ui)


def test_the_default_chapters_without_a_key_fails_in_plain_words(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    seed_transcript(ui.config, V1)
    nf(ui, f"/videos/{V1}/chapters")
    ui.web.chapters_job.wait()
    snap = ui.web.chapters_job.snapshot()
    assert snap.status == "failed" and snap.message == "No Meta API key is set. Add one in Configuration."


# --- making chapters ----------------------------------------------------------------------------
def test_making_chapters_saves_a_draft_and_the_card_links_to_changes(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, chapters_fn=chapters_with([reply(GOOD)]))
    seed_transcript(ui.config, V1)
    resp = nf(ui, f"/videos/{V1}/chapters")
    ui.web.chapters_job.wait()
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#transcript"
    with store_of(ui) as store:
        draft = store.get_draft(V1)
    assert draft is not None and draft.source == "chapters"
    assert draft.description == "Repotting a cactus.\n\nChapters:\n0:00 Intro\n1:40 Choosing a pot\n3:20 Aftercare"
    section = card(page(ui))
    assert 'id="chapters-done"' in section and "Saved 3 chapters as a draft" in section
    assert f'href="/changes#d-{V1}"' in section
    assert re.findall(r"<li>(.*?)</li>", section.split('id="chapters-list"')[1].split("</ul>")[0]) == [
        "0:00 Intro", "1:40 Choosing a pot", "3:20 Aftercare"]
    nf(ui, "/chapters/done", data={"video_id": V1})
    assert 'id="chapters-done"' not in card(page(ui))
    assert ui.client.get("/api/status").json()["chapters"]["status"] == "done"


def test_the_model_sees_numbered_segments_and_the_times_come_from_the_transcript(tmp_path: Path) -> None:
    fn = chapters_with([reply([(0, "Intro"), (2, "Choosing a pot"), (4, "Aftercare")])])
    ui, _ = make_ts_ui(tmp_path, chapters_fn=fn)
    seed_transcript(ui.config, V1)
    nf(ui, f"/videos/{V1}/chapters")
    ui.web.chapters_job.wait()
    prompt = json.dumps(fn.client.calls[0]["messages"])
    assert "[2] 1:40 Piece 2 talks about topic 2." in prompt
    with store_of(ui) as store:
        draft = store.get_draft(V1)
    assert draft is not None and "1:40 Choosing a pot" in draft.description


def test_a_chapters_result_that_failed_is_shown_as_an_error(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, chapters_fn=chapters_with([reply([(0, "Intro")]), reply([(0, "Intro")])]))
    seed_transcript(ui.config, V1)
    nf(ui, f"/videos/{V1}/chapters")
    ui.web.chapters_job.wait()
    section = card(page(ui))
    assert 'class="flash error" role="status" id="chapters-done"' in section
    assert "could not pick at least 3 usable chapters" in section
    assert "Review the draft" not in section
    with store_of(ui) as store:
        assert store.get_draft(V1) is None


def test_chapters_are_refused_without_a_transcript_or_when_blocked(tmp_path: Path) -> None:
    called: list[str] = []

    def fn(store: Store, video_id: str, report: Any) -> Any:
        called.append(video_id)
        raise AssertionError("must not run")

    ui, _ = make_ts_ui(tmp_path, chapters_fn=fn)
    nf(ui, f"/videos/{V1}/chapters")
    assert "Transcribe the video first." in page(ui)
    seed_transcript(ui.config, SHORT, segments=2)
    nf(ui, f"/videos/{SHORT}/chapters")
    assert "longer than a minute" in page(ui, SHORT)
    assert called == [] and ui.web.chapters_job.snapshot().status == "idle"


def test_chapters_for_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path, chapters_fn=chapters_with([]))
    assert nf(ui, "/videos/nosuchvideo/chapters").status_code == 404


def test_a_running_chapters_job_shows_progress_and_blocks_a_second_start(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(store: Store, video_id: str, report: Any) -> Any:
        report("Choosing the chapters")
        entered.set()
        assert gate.wait(10)
        return chapters.ChaptersResult("no_change", "The description already matches.", (), 0)

    ui, _ = make_ts_ui(tmp_path, chapters_fn=slow)
    seed_transcript(ui.config, V1)
    seed_transcript(ui.config, V2)
    try:
        nf(ui, f"/videos/{V1}/chapters")
        assert entered.wait(10)
        html = page(ui)
        section = card(html)
        assert 'id="chapters-msg"' in section and "Choosing the chapters" in section
        assert 'data-poll="1"' in html
        assert ui.client.get("/api/status").json()["chapters"]["status"] == "running"
        other = card(page(ui, V2))
        assert 'id="chapters-busy"' in other
        nf(ui, f"/videos/{V2}/chapters")
        assert "already being made" in page(ui, V2)
    finally:
        gate.set()
        ui.web.chapters_job.wait()
    assert 'class="flash info"' in card(page(ui))


# --- the subtitle download ----------------------------------------------------------------------
def test_the_srt_download_has_the_right_headers_and_body(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1, segments=2, texts=["Hello there.", "Second piece."])
    resp = ui.client.get(f"/videos/{V1}/transcript.srt")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-subrip"
    assert resp.headers["content-disposition"] == f'attachment; filename="{V1}.srt"'
    assert resp.text == ("1\n00:00:00,000 --> 00:00:50,000\nHello there.\n\n"
                         "2\n00:00:50,000 --> 00:01:40,000\nSecond piece.\n")


def test_the_srt_uses_the_latest_transcript(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    seed_transcript(ui.config, V1, segments=1, texts=["Old words."])
    seed_transcript(ui.config, V1, segments=1, texts=["New words."])
    body = ui.client.get(f"/videos/{V1}/transcript.srt").text
    assert "New words." in body and "Old words." not in body


def test_the_srt_is_a_404_when_there_is_no_transcript(tmp_path: Path) -> None:
    ui, _ = make_ts_ui(tmp_path)
    assert ui.client.get(f"/videos/{V1}/transcript.srt").status_code == 404
    assert ui.client.get("/videos/nosuchvideo/transcript.srt").status_code == 404


def test_the_srt_needs_a_signed_in_browser(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, signed_in_browser=False, transcribe=FakeTranscribe())
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    seed_transcript(ui.config, V1)
    resp = ui.client.get(f"/videos/{V1}/transcript.srt")
    assert resp.status_code in (401, 403) and "Piece 0" not in resp.text


# --- safety -------------------------------------------------------------------------------------
POSTS = [f"/videos/{V1}/transcribe", "/transcribe/done", f"/videos/{V1}/chapters", "/chapters/done"]


@pytest.mark.parametrize("path", POSTS)
def test_every_new_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    fn = chapters_with([reply(GOOD)])
    ui, fake = make_ts_ui(tmp_path, chapters_fn=fn)
    seed_transcript(ui.config, V1)
    data = {"path": str(media(tmp_path)), "language": "en", "video_id": V1}
    assert ui.client.post(path, data={}).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong", **data}).status_code == 403
    assert fake.calls == [] and fn.client.calls == []
    assert ui.web.transcribe_job.snapshot().status == "idle"
    assert ui.web.chapters_job.snapshot().status == "idle"


def test_the_full_path_is_never_in_any_page_or_status(tmp_path: Path) -> None:
    ui, fake = make_ts_ui(tmp_path)
    path = media(tmp_path)
    start(ui, path)
    assert fake.calls
    seen = [page(ui), page(ui, V2), ui.client.get("/setup").text, ui.client.get("/changes").text,
            ui.client.get("/agent").text, ui.client.get("/api/status").text,
            ui.client.get(f"/videos/{V1}/transcript.srt").text]
    for text in seen:
        assert str(path) not in text and str(path.parent) not in text
        assert "private-folder-xyz" not in text
    assert "My Clip.mp4" in seen[0]
    with store_of(ui) as store:
        record = store.latest_transcript(V1)
    assert record is not None and "private-folder-xyz" not in json.dumps(record.__dict__)


def test_nothing_in_the_transcript_flow_reaches_youtube(tmp_path: Path) -> None:
    from fakes import FakeYouTubeWriter

    writer = FakeYouTubeWriter(0)
    ui, _ = make_ts_ui(tmp_path, chapters_fn=chapters_with([reply(GOOD)]),
                       writer_factory=lambda creds: writer)
    start(ui, media(tmp_path))
    nf(ui, f"/videos/{V1}/chapters")
    ui.web.chapters_job.wait()
    ui.client.get(f"/videos/{V1}/transcript.srt")
    assert writer.calls == []
    with store_of(ui) as store:
        assert store.list_pushes() == []


# --- setup and the polling script ---------------------------------------------------------------
def test_setup_says_whether_ffmpeg_was_found(tmp_path: Path, monkeypatch) -> None:
    ui, _ = make_ts_ui(tmp_path)
    html = ui.client.get("/setup").text
    assert "ffmpeg was found on this computer." in html and "ffmpeg found" in html
    assert "need it to read the sound of a video file" in html
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    html = ui.client.get("/setup").text
    assert "ffmpeg was not found on this computer. Install it and restart Yezatas." in html
    assert "<script>" not in html and "style=" not in html


def test_the_polling_script_watches_the_new_jobs() -> None:
    js = (Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert '"transcribe"' in js and '"chapters"' in js
