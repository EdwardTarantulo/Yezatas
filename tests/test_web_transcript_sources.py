"""The three sources of a transcript in the web UI besides the speech upload (which is in
test_web_transcript.py): the captions already on YouTube, the sound tagger, and the Local mode
messages. The YouTube captions reader, the sound tagger and the transcriber are fakes; nothing
touches the network, ffmpeg or PANNs."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from webhelpers import fake_creds, http_error, make_config, make_ui, make_video, seed_catalog
from yezatas import agent, captions, primaryai, soundtags, transcribe
from yezatas.quota import QuotaGuard
from yezatas.soundtags import SoundTagError, SoundTagResult
from yezatas.store import Store, TranscriptRecord
from yezatas.transcribe import TranscribeError, TranscribeResult

V1 = "vidsrc001"
FAKE_KEY = "FAKE-KEY-0123456789abcdef"
SRT = ("1\n00:00:00,000 --> 00:00:04,000\nHello and welcome.\n\n"
       "2\n00:00:04,000 --> 00:00:09,500\nToday we repot a cactus.\n\n"
       "3\n00:00:09,500 --> 00:00:15,000\nFirst, a bigger pot.\n\n"
       "4\n00:00:15,000 --> 00:00:21,250\nThen fresh soil.\n")
LISTING = {"items": [{"id": "cap-en", "snippet": {"language": "en", "trackKind": "standard",
                                                  "status": "serving"}}]}


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)
    monkeypatch.delenv(primaryai.PRIMARY_AI_ENV, raising=False)
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(soundtags, "available", lambda: True)


class FakeCaptionReader:
    """The injected caption reader: records the calls, answers from a script."""

    def __init__(self, listing: Any = LISTING, srt: Any = SRT, list_error: Exception | None = None,
                 srt_error: Exception | None = None) -> None:
        self.listing, self.srt = listing, srt
        self.list_error, self.srt_error = list_error, srt_error
        self.calls: list[tuple[str, str]] = []

    def caption_tracks(self, video_id: str) -> Any:
        self.calls.append(("list", video_id))
        if self.list_error is not None:
            raise self.list_error
        return self.listing

    def caption_srt(self, caption_id: str) -> Any:
        self.calls.append(("download", caption_id))
        if self.srt_error is not None:
            raise self.srt_error
        return self.srt


class FakeTagger:
    """The injected sound tagger: sees the uploaded copy, stores nothing (the job does)."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, video_id: str, media_path: str, report: Any) -> SoundTagResult:
        self.calls.append({"video_id": video_id, "path": media_path,
                           "existed": Path(media_path).is_file()})
        report("Listening for sounds")
        if self.error is not None:
            raise self.error
        segments = [{"start_ms": 0, "end_ms": 20_000, "text": "[Music]"},
                    {"start_ms": 20_000, "end_ms": 30_000, "text": "[Applause]"}]
        record = TranscriptRecord(0, video_id, "", soundtags.MODEL_NAME, Path(media_path).name,
                                  30_000, segments, [], "")
        return SoundTagResult(record, "Found 2 sound captions in 0:30 of audio.", 0.0)


class FakeSpeech:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, store: Store, video_id: str, path: str, language: str, report: Any,
                 keywords: tuple[str, ...] = ()) -> TranscribeResult:
        self.calls.append({"video_id": video_id, "path": path, "language": language,
                           "existed": Path(path).is_file()})
        parts = [{"start_ms": 0, "end_ms": 50_000, "text": "Some words."}]
        record = TranscriptRecord(0, video_id, language, "voice-test", Path(path).name, 50_000,
                                  parts, [], "")
        stored = store.get_transcript(store.add_transcript(record))
        assert stored is not None
        return TranscribeResult(stored, "Transcribed 0:50 of audio in 1 piece.", 0.0025)


def make_src_ui(tmp_path: Path, *, reader: Any = "default", tagger: Any = "default",
                speech: Any = "default", write: bool = True, language: str | None = "en",
                config: Any = None, **kwargs: Any):
    reader = FakeCaptionReader() if reader == "default" else reader
    tagger = FakeTagger() if tagger == "default" else tagger
    speech = FakeSpeech() if speech == "default" else speech
    loader = (lambda cfg: fake_creds()) if write else (lambda cfg: None)
    ui = make_ui(tmp_path, config=config, transcribe=speech, soundtags=tagger,
                 caption_reader=lambda creds: reader, write_credentials_loader=loader, **kwargs)
    seed_catalog(ui.config, [make_video(1, video_id=V1, title="Cactus repotting", tags=["cactus"],
                                        privacy_status="public", default_language=language,
                                        duration="PT5M30S")])
    return ui, reader, tagger, speech


def esc(text: str) -> str:
    """How the page writes a sentence (Jinja escapes quotes and angle brackets)."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("'", "&#39;").replace('"', "&#34;"))


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def page(ui: Any) -> str:
    return ui.client.get(f"/videos/{V1}").text


def card(html: str) -> str:
    return html.split('id="chapters"')[1].split("</section>")[0]


def block(html: str, ident: str) -> str:
    """One source option of the card: from its id up to the next one (or the end)."""
    section = card(html)
    return section.split(f'id="{ident}"')[1].split('class="source-option"')[0]


def captions_run(ui: Any) -> Any:
    resp = nf(ui, f"/videos/{V1}/transcribe/captions")
    ui.web.transcribe_job.wait()
    return resp


def upload(ui: Any, *, kind: str = "sounds", name: str = "clip.mp4", wait: bool = True,
           **extra: str) -> Any:
    resp = nf(ui, f"/videos/{V1}/transcribe", data={"kind": kind, **extra},
              files={"file": (name, b"not really a video", "video/mp4")})
    if wait:
        ui.web.transcribe_job.wait()
    return resp


def uploads_left(ui: Any) -> list[Path]:
    root = ui.config.home / "tmp" / "uploads"
    return sorted(root.iterdir()) if root.is_dir() else []


def stored(ui: Any) -> TranscriptRecord | None:
    with Store(ui.config.catalog_db) as store:
        return store.latest_transcript(V1)


# --- captions already on YouTube ----------------------------------------------------------------
def test_the_card_offers_captions_first_without_calling_youtube(tmp_path: Path) -> None:
    ui, reader, _, _ = make_src_ui(tmp_path)
    section = card(page(ui))
    assert section.index('id="source-captions"') < section.index('id="source-file"')
    assert "Captions already on YouTube" in section
    assert "Read the captions on YouTube (250 quota units)." in section
    assert f'action="/videos/{V1}/transcribe/captions"' in section and ">Use these captions<" in section
    assert reader.calls == []  # a page load never spends quota


def test_captions_run_as_a_job_and_become_the_transcript(tmp_path: Path) -> None:
    ui, reader, _, _ = make_src_ui(tmp_path)
    resp = captions_run(ui)
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#chapters"
    assert reader.calls == [("list", V1), ("download", "cap-en")]
    record = stored(ui)
    assert record is not None and record.model == captions.MODEL_NAME and record.language == "en"
    assert [s["text"] for s in record.segments][:2] == ["Hello and welcome.", "Today we repot a cactus."]
    assert record.duration_ms == 21_250
    section = card(page(ui))
    assert 'id="transcribe-done"' in section
    assert "Read 4 caption lines (English · uploaded by you)." in section
    assert "no cost" in section and "250 units of your daily YouTube quota" in section
    assert 'id="transcript-text"' in section and "Read the transcript (4 lines)" in section
    assert "0:09 First, a bigger pot." in section
    assert "cost about" not in section
    srt = ui.client.get(f"/videos/{V1}/transcript.srt")
    assert srt.status_code == 200 and "00:00:09,500 --> 00:00:15,000" in srt.text


def test_reading_captions_is_charged_to_the_quota_log(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    captions_run(ui)
    with Store(ui.config.catalog_db) as store:
        assert QuotaGuard(store, ui.config.daily_quota_budget).used() == 250


def test_the_track_in_the_videos_language_is_chosen(tmp_path: Path) -> None:
    listing = {"items": [
        {"id": "cap-en", "snippet": {"language": "en", "trackKind": "standard"}},
        {"id": "cap-de", "snippet": {"language": "de", "trackKind": "standard"}}]}
    ui, reader, _, _ = make_src_ui(tmp_path, reader=FakeCaptionReader(listing=listing), language="de")
    captions_run(ui)
    assert ("download", "cap-de") in reader.calls
    record = stored(ui)
    assert record is not None and record.language == "de"


ERRORS = [
    ("none", FakeCaptionReader(listing={"items": []}), captions.NO_CAPTIONS_MESSAGE),
    ("scope", FakeCaptionReader(list_error=http_error(403, "insufficientPermissions")),
     captions.SCOPE_MESSAGE),
    ("quota", FakeCaptionReader(list_error=http_error(403, "quotaExceeded")), captions.QUOTA_MESSAGE),
    ("not_found", FakeCaptionReader(srt_error=http_error(404, "notFound")),
     "YouTube could not find this video or its captions."),
    ("network", FakeCaptionReader(list_error=OSError("down")),
     "Could not reach YouTube. Check your internet connection and try again."),
    ("api", FakeCaptionReader(srt_error=http_error(500, "backendError")),
     "YouTube would not download the captions of this video (HTTP 500)."),
    ("srt", FakeCaptionReader(srt="1\n00:00:xx --> 00:00:02,000\nHi\n"),
     "The caption file is malformed: cue 1 has a time that is not HH:MM:SS,mmm."),
]


@pytest.mark.parametrize("kind, reader, message", ERRORS, ids=[e[0] for e in ERRORS])
def test_every_captions_error_becomes_its_sentence_in_the_card(tmp_path: Path, kind: str,
                                                              reader: Any, message: str) -> None:
    ui, _, _, _ = make_src_ui(tmp_path, reader=reader)
    captions_run(ui)
    snap = ui.web.transcribe_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == f"captions_{kind}"
    section = card(page(ui))
    assert 'id="transcribe-failed"' in section and esc(message) in section
    assert stored(ui) is None


def test_a_scope_error_offers_the_same_reconnect_as_comments(tmp_path: Path) -> None:
    reader = FakeCaptionReader(list_error=http_error(401, "authError"))
    ui, _, _, _ = make_src_ui(tmp_path, reader=reader)
    captions_run(ui)
    part = block(page(ui), "source-captions")
    assert 'id="captions-no-write"' in part and 'action="/changes/allow-edits"' in part
    assert "Allow Yezatas to edit your videos" in part and "Use these captions" not in part


def test_without_the_edit_permission_captions_ask_for_it_and_read_nothing(tmp_path: Path) -> None:
    ui, reader, _, _ = make_src_ui(tmp_path, write=False)
    part = block(page(ui), "source-captions")
    assert 'action="/changes/allow-edits"' in part and "Use these captions" not in part
    nf(ui, f"/videos/{V1}/transcribe/captions")
    assert "Reading captions needs the YouTube permission that pushing uses" in page(ui)
    assert reader.calls == [] and ui.web.transcribe_job.snapshot().status == "idle"


def test_captions_stop_when_the_daily_quota_budget_has_no_room(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), daily_quota_budget=200)
    ui, reader, _, _ = make_src_ui(tmp_path, config=config)
    captions_run(ui)
    assert reader.calls == []
    section = card(page(ui))
    assert "needs 250 units of the daily YouTube quota budget" in section


def test_captions_for_an_unknown_video_are_a_404(tmp_path: Path) -> None:
    ui, reader, _, _ = make_src_ui(tmp_path)
    assert nf(ui, "/videos/nosuchvideo/transcribe/captions").status_code == 404
    assert reader.calls == []


def test_a_second_start_while_the_job_runs_is_refused(tmp_path: Path) -> None:
    gate = threading.Event()

    class Slow(FakeCaptionReader):
        def caption_tracks(self, video_id: str) -> Any:
            assert gate.wait(10)
            return super().caption_tracks(video_id)

    ui, reader, _, _ = make_src_ui(tmp_path, reader=Slow())
    try:
        nf(ui, f"/videos/{V1}/transcribe/captions")
        nf(ui, f"/videos/{V1}/transcribe/captions")
        assert "A transcription is already running" in page(ui)
    finally:
        gate.set()
        ui.web.transcribe_job.wait()


def test_captions_and_sound_tags_never_show_a_price(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    html = page(ui)
    assert "$" not in block(html, "source-captions")
    kind = block(html, "source-file").split('<legend>')[1].split("</fieldset>")[0]
    assert "$" not in kind and "Price" not in kind
    assert 'id="ts-price"' in card(html)  # the price belongs to speech only
    assert html.split('id="ts-price"')[1].split("</p>")[0].count("$0.18") == 1


# --- music and sounds ----------------------------------------------------------------------------
def test_the_sound_choice_is_in_the_file_form(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    part = block(page(ui), "source-file")
    assert "What is in the sound" in part
    assert 'name="kind" value="speech" checked' in part and "Speech" in part
    assert 'name="kind" value="sounds"' in part and "Music and sounds, no speech" in part
    assert "sound-tagger" not in part and 'id="sounds-not-installed"' not in part
    assert "free, and nothing is sent" in part


def test_sound_tags_run_on_the_uploaded_copy_and_are_stored(tmp_path: Path) -> None:
    ui, _, tagger, speech = make_src_ui(tmp_path)
    resp = upload(ui)  # no language, no names, no price: none of it is asked
    assert resp.status_code == 303
    assert len(tagger.calls) == 1 and speech.calls == []
    call = tagger.calls[0]
    assert call["video_id"] == V1 and call["existed"] and Path(call["path"]).name == "clip.mp4"
    assert uploads_left(ui) == []
    record = stored(ui)
    assert record is not None and record.model == soundtags.MODEL_NAME and record.language == ""
    assert record.source_name == "clip.mp4"
    section = card(page(ui))
    assert "Found 2 sound captions in 0:30 of audio." in section
    assert "no cost" in section and "nothing was sent" in section
    assert "Sounds, no speech" in section and "cost about" not in section
    assert "0:20 [Applause]" in section


def test_sound_tags_do_not_need_a_language_or_a_meta_key(tmp_path: Path) -> None:
    ui, _, tagger, _ = make_src_ui(tmp_path, language=None)
    upload(ui, language="")
    assert len(tagger.calls) == 1 and ui.web.transcribe_job.snapshot().status == "done"


def test_a_failing_tagger_shows_its_sentence_and_the_copy_is_deleted(tmp_path: Path) -> None:
    error = SoundTagError("tagger", "The sound tagger failed (RuntimeError).")
    ui, _, tagger, _ = make_src_ui(tmp_path, tagger=FakeTagger(error=error))
    upload(ui)
    assert tagger.calls[0]["existed"] is True
    snap = ui.web.transcribe_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "tagger"
    assert "The sound tagger failed (RuntimeError)." in card(page(ui))
    assert uploads_left(ui) == [] and stored(ui) is None


def test_an_unexpected_tagger_error_is_generic_and_the_copy_is_deleted(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path, tagger=FakeTagger(error=KeyError("secret 123")))
    upload(ui)
    assert ui.web.transcribe_job.snapshot().message == "The transcription stopped unexpectedly (KeyError)."
    assert uploads_left(ui) == []


def test_a_tagger_error_does_not_show_the_folder_of_the_copy(tmp_path: Path) -> None:
    def failing(video_id: str, media_path: str, report: Any) -> Any:
        raise SoundTagError("ffmpeg", f"ffmpeg could not read the audio of this file: {media_path}")

    ui, _, _, _ = make_src_ui(tmp_path, tagger=failing)
    upload(ui)
    message = ui.web.transcribe_job.snapshot().message
    assert "clip.mp4" in message and "uploads" not in message and "upload-" not in message


def test_the_default_tagger_is_the_real_one_with_the_copy_and_the_report(tmp_path: Path,
                                                                        monkeypatch) -> None:
    seen: list[tuple[str, str, bool]] = []

    def fake_tag_video(video_id: str, media_path: Any, **kw: Any) -> SoundTagResult:
        seen.append((video_id, Path(media_path).name, callable(kw.get("report"))))
        record = TranscriptRecord(0, video_id, "", soundtags.MODEL_NAME, Path(media_path).name,
                                  10_000, [{"start_ms": 0, "end_ms": 10_000, "text": "[Music]"}],
                                  [], "")
        return SoundTagResult(record, "Found 1 sound caption in 0:10 of audio.", 0.0)

    monkeypatch.setattr(soundtags, "tag_video", fake_tag_video)
    ui = make_ui(tmp_path, transcribe=FakeSpeech(), caption_reader=lambda creds: FakeCaptionReader())
    seed_catalog(ui.config, [make_video(1, video_id=V1, privacy_status="public")])
    upload(ui)
    assert seen == [(V1, "clip.mp4", True)]
    assert stored(ui) is not None and uploads_left(ui) == []


def test_without_the_sound_tagger_the_card_says_so_and_the_post_is_refused(tmp_path: Path,
                                                                          monkeypatch) -> None:
    monkeypatch.setattr(soundtags, "available", lambda: False)
    ui, _, tagger, _ = make_src_ui(tmp_path)
    part = block(page(ui), "source-file")
    assert 'id="sounds-not-installed"' in part
    assert "The sound tagger is not installed. See" in part
    assert 'href="/configuration#step-transcripts"' in part and "Configuration &gt; Transcripts and sounds" in part
    upload(ui)
    assert esc(soundtags.MISSING_MESSAGE) in page(ui)
    assert tagger.calls == [] and uploads_left(ui) == []
    assert ui.web.transcribe_job.snapshot().status == "idle"
    # speech still works
    upload(ui, kind="speech", language="en")
    assert ui.web.transcribe_job.snapshot().status == "done"


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    ui, _, tagger, speech = make_src_ui(tmp_path)
    upload(ui, kind="humming")
    assert "Choose whether the sound is speech or music and sounds." in page(ui)
    assert tagger.calls == [] and speech.calls == [] and uploads_left(ui) == []


def test_a_wrong_file_type_is_refused_for_sounds_too(tmp_path: Path) -> None:
    ui, _, tagger, _ = make_src_ui(tmp_path)
    upload(ui, name="notes.txt")
    assert "notes.txt is not a video or audio file" in page(ui)
    assert tagger.calls == [] and uploads_left(ui) == []


# --- Primary AI is Local ------------------------------------------------------------------------
def local(ui: Any, key: bool = False) -> None:
    primaryai.save_primary_ai(ui.config, "local")
    if key:
        agent.save_api_key(ui.config, FAKE_KEY)


def test_in_local_mode_without_a_key_the_speech_option_shows_the_amber_note(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    local(ui)
    part = block(page(ui), "source-file")
    assert 'class="fix speech-only" id="transcribe-local-no-key"' in part
    assert ("Transcribing speech from a file needs Meta's API. Primary AI is Local and no Meta key "
            "is saved.") in part
    assert "or use the YouTube captions." in part and 'href="/configuration#step-ai"' in part
    assert "Add a key in Configuration" in part
    assert 'id="transcribe-no-key"' not in part and 'id="ts-price"' not in part


def test_in_local_mode_without_a_key_a_speech_upload_is_refused_but_other_sources_work(
        tmp_path: Path) -> None:
    ui, reader, tagger, speech = make_src_ui(tmp_path)
    local(ui)
    upload(ui, kind="speech", language="en")
    html = page(ui)
    assert "Transcribing speech from a file needs Meta&#39;s API." in html or \
        "Transcribing speech from a file needs Meta's API." in html
    assert speech.calls == [] and uploads_left(ui) == []
    assert ui.web.transcribe_job.snapshot().status == "idle"
    upload(ui, kind="sounds")  # free and local: no key needed
    assert len(tagger.calls) == 1
    captions_run(ui)  # the captions use the YouTube sign-in
    assert ("list", V1) in reader.calls and stored(ui).model == captions.MODEL_NAME  # type: ignore[union-attr]


def test_in_local_mode_with_a_key_speech_works_and_shows_its_price(tmp_path: Path) -> None:
    ui, _, _, speech = make_src_ui(tmp_path)
    local(ui, key=True)
    part = block(page(ui), "source-file")
    assert 'id="transcribe-local-no-key"' not in part and 'id="transcribe-no-key"' not in part
    assert "Price: about $0.18 per hour of audio." in part
    assert FAKE_KEY not in page(ui)
    upload(ui, kind="speech", language="en")
    assert len(speech.calls) == 1 and speech.calls[0]["existed"] is True
    assert ui.web.transcribe_job.snapshot().status == "done"
    assert uploads_left(ui) == []
    assert "The cost was about" in card(page(ui))


def test_in_meta_mode_without_a_key_the_old_hint_stays_and_nothing_is_refused_early(
        tmp_path: Path) -> None:
    ui, _, _, speech = make_src_ui(tmp_path)
    part = block(page(ui), "source-file")
    assert 'id="transcribe-no-key"' in part and "Add one in Configuration" in part
    assert 'id="transcribe-local-no-key"' not in part
    upload(ui, kind="speech", language="en")
    assert len(speech.calls) == 1


# --- the card as a whole -------------------------------------------------------------------------
def test_the_card_has_no_script_inline_style_or_event_handler(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    local(ui)
    section = card(page(ui))
    assert "<script" not in section and "style=" not in section and "onclick" not in section
    assert "onchange" not in section


def test_the_intro_names_the_three_sources_and_what_is_sent(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    intro = card(page(ui)).split('<p class="muted">')[1].split("</p>")[0]
    assert "Captions already on YouTube" in intro and "no AI is involved" in intro
    assert "temporary copy" in intro and "only the sound is sent, never the picture" in intro
    assert "nothing leaves the computer" in intro
    assert "Nothing goes to YouTube until you approve and push it." in intro


def test_the_card_never_says_setup_where_it_means_configuration(tmp_path: Path) -> None:
    ui, _, _, _ = make_src_ui(tmp_path)
    assert " in Setup" not in card(page(ui))
    assert "Setup." not in transcribe.AUTH_MESSAGE


def test_nothing_in_the_new_sources_reaches_youtube(tmp_path: Path) -> None:
    from fakes import FakeYouTubeWriter

    writer = FakeYouTubeWriter(0)
    ui, _, _, _ = make_src_ui(tmp_path, writer_factory=lambda creds: writer)
    captions_run(ui)
    upload(ui)
    assert writer.calls == []
    with Store(ui.config.catalog_db) as store:
        assert store.list_pushes() == []


def test_chapters_are_not_made_from_sound_labels() -> None:
    from yezatas.soundtags import MODEL_NAME
    from yezatas.web.views import chapters_blocker

    class V:
        duration = "PT8M41S"
        description = ""

    class R:
        model = MODEL_NAME
        duration_ms = 521000

    assert "sound labels" in chapters_blocker(V(), None, R())
