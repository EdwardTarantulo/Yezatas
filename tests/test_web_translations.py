"""Translations page - glossary UI and job wiring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import markupsafe
import pytest

from webhelpers import make_ui, make_video, seed_catalog
from fakes import ScriptedVerifier
from yezatas import glossary as glossary_mod
from yezatas.store import Store


def shown(text: str) -> str: return str(markupsafe.escape(text))


def glossary_path(ui: Any) -> Path:
    return ui.config.home / glossary_mod.GLOSSARY_FILE  # noqa: E501


def page(ui: Any, path: str = "/translations") -> str:
    resp = ui.client.get(path)
    assert resp.status_code == 200
    return resp.text


def post_and_follow(ui: Any, path: str, data: dict[str, Any]) -> tuple[Any, str]:
    """POST via ui.post (adds CSRF) and follow the redirect to read flash."""
    resp = ui.post(path, data=data)
    # ui.post uses TestClient without follow_redirects -> 303
    if resp.status_code in (302, 303, 307, 308):
        loc = resp.headers.get("location", "/translations")
        # strip fragment
        loc = loc.split("#")[0]
        html = ui.client.get(loc).text
        return resp, html
    # already followed or error
    return resp, resp.text


# ---------------------------------------------------------------------------
# page rendering + nav
# ---------------------------------------------------------------------------

def test_page_renders_no_terms_hint_and_nav(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    html = page(ui, "/translations")
    assert 'id="glossary-no-terms"' in html
    assert "No terms yet." in html
    assert "Terms here are sent with every translation (titles, descriptions and subtitles), and code checks each translation for them. A term that is broken flags the translation." in html

    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    # /translations right after /replies
    assert "/replies" in items and "/translations" in items
    assert items.index("/translations") == items.index("/replies") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/translations"]


# ---------------------------------------------------------------------------
# add keep term + fixed translations
# ---------------------------------------------------------------------------

def test_add_keep_term_shows_table_and_writes_file(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})
    assert shown('Added the term "Mesa" to the glossary.') in html
    assert "Mesa" in html
    assert "Keep as it is" in html
    # file has it
    terms = glossary_mod.load_glossary(glossary_path(ui))
    assert len(terms) == 1 and terms[0].term == "Mesa" and terms[0].keep is True

    # add Cactus with two fixed translations
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "de = Kaktus\nes = Cactus Grande"})
    assert shown('Added the term "Cactus" to the glossary.') in html
    # table shows de: Kaktus, es: Cactus Grande
    assert "Cactus" in html
    assert "de: Kaktus, es: Cactus Grande" in html

    terms = glossary_mod.load_glossary(glossary_path(ui))
    assert len(terms) == 2


def test_add_cactus_translations_displayed(tmp_path: Path) -> None:
    # isolated check of Cactus translations rendering
    ui = make_ui(tmp_path)
    post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "de = Kaktus\nes = Cactus Grande"})
    html = page(ui)
    assert "de: Kaktus, es: Cactus Grande" in html


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------

def test_translation_line_without_equals_flashes_and_does_not_write(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    # ensure empty
    assert glossary_path(ui).exists() is False or glossary_mod.load_glossary(glossary_path(ui)) == []
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "de Kaktus"})
    assert shown('Write each fixed translation as "de = Wort", one per line.') in html
    # file not written / still empty
    if glossary_path(ui).exists():
        terms = glossary_mod.load_glossary(glossary_path(ui))
        assert terms == []
    else:
        assert not glossary_path(ui).exists() or glossary_path(ui).read_text().strip() == ""


def test_unknown_language_code_flashes_and_writes_nothing(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "xx = Foo"})
    assert shown('Unknown language "xx"') in html
    # not written
    if glossary_path(ui).exists():
        # if file exists it should be empty / no terms
        try:
            terms = glossary_mod.load_glossary(glossary_path(ui))
            assert terms == []
        except glossary_mod.GlossaryError:
            pytest.fail("file should not be written with invalid term")
    else:
        assert True


def test_keep_ticked_with_translations_flashes_cannot_be_both(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": "de = Kaktus"})
    assert "cannot be both" in html.lower()
    # file not written
    if glossary_path(ui).exists():
        try:
            terms = glossary_mod.load_glossary(glossary_path(ui))
            assert terms == []
        except glossary_mod.GlossaryError:
            pytest.fail("file should remain empty")


def test_duplicate_term_case_insensitive(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "mesa", "keep": "1", "translations": ""})
    assert "That term is already in the glossary." in html


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def test_delete_by_index_removes_right_term(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})
    post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "de = Kaktus\nes = Cactus Grande"})
    # delete index 0 (Mesa)
    resp, html = post_and_follow(ui, "/translations/glossary/delete", {"index": "0"})
    assert shown('Removed the term "Mesa".') in html
    html2 = page(ui)
    assert "Mesa" not in html2
    assert "Cactus" in html2
    terms = glossary_mod.load_glossary(glossary_path(ui))
    assert len(terms) == 1 and terms[0].term == "Cactus"


def test_delete_invalid_index_flashes(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})
    for bad in ("9", "abc"):
        resp, html = post_and_follow(ui, "/translations/glossary/delete", {"index": bad})
        assert "That term is no longer in the glossary." in html


# ---------------------------------------------------------------------------
# broken glossary.json
# ---------------------------------------------------------------------------

def test_broken_glossary_shows_error_and_no_form_and_post_does_not_change_bytes(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    # first add a valid term so file exists, then corrupt
    post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})
    p = glossary_path(ui)
    p.write_text("{not json", encoding="utf-8")
    html = page(ui)
    # shows error
    assert 'id="glossary-error"' in html
    assert "Glossary is not valid JSON" in html
    assert 'id="glossary-add"' not in html

    before = p.read_bytes()
    resp, html2 = post_and_follow(ui, "/translations/glossary/add", {"term": "Cactus", "translations": "de = Kaktus"})
    after = p.read_bytes()
    assert before == after
    # still error
    html3 = page(ui)
    assert 'id="glossary-error"' in html3


# ---------------------------------------------------------------------------
# XSS / script
# ---------------------------------------------------------------------------

def test_term_with_script_is_refused(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp, html = post_and_follow(ui, "/translations/glossary/add", {"term": "<script>x</script>", "keep": "1", "translations": ""})
    # response never contains raw script
    assert "<script>x" not in html
    assert "<script>x" not in resp.text
    # also page should not contain it
    html2 = page(ui)
    assert "<script>x" not in html2
    # file not written
    if glossary_path(ui).exists():
        try:
            terms = glossary_mod.load_glossary(glossary_path(ui))
            assert all("<script>" not in t.term for t in terms)
        except glossary_mod.GlossaryError:
            pass


# ---------------------------------------------------------------------------
# CSRF
# ---------------------------------------------------------------------------

def test_post_without_csrf_is_refused(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    # direct client post without csrf token
    resp = ui.client.post("/translations/glossary/add", data={"term": "Mesa", "keep": "1"})
    # affiliate tests expect 403 (or 400) refuse
    assert resp.status_code in (400, 403)
    # file not created
    if glossary_path(ui).exists():
        content = glossary_path(ui).read_bytes()
        # should not contain Mesa
        assert b"Mesa" not in content
    else:
        assert True


# ---------------------------------------------------------------------------
# job wiring
# ---------------------------------------------------------------------------

def test_job_wiring_translator_receives_glossary_and_broken_suffix(tmp_path: Path) -> None:
    from yezatas import translate as translate_mod

    class RecordingTranslator:
        def __init__(self) -> None:
            self.model = "test-model"
            self.calls: list[tuple[str, str, str, str, str]] = []
            self.last_glossary: str | None = None

        def translate(self, source_language: str, target_language: str, title: str, description: str, glossary: str = "") -> Any:  # type: ignore[override]
            self.last_glossary = glossary
            self.calls.append((source_language, target_language, title, description, glossary))
            return translate_mod.Translated(f"[{target_language}] {title}", f"[{target_language}] {description}", self.model, 10, 20)

    translator = RecordingTranslator()
    ui = make_ui(tmp_path, translate_tools=lambda cfg: (translator, ScriptedVerifier(up=False)))

    # save glossary with Mesa keep
    post_and_follow(ui, "/translations/glossary/add", {"term": "Mesa", "keep": "1", "translations": ""})

    # seed video whose title has Mesa
    vid = "vid00000001"
    seed_catalog(ui.config, [
        make_video(1, video_id=vid, title="Mesa in the desert", description="A link https://shop.invalid/item and Mesa", privacy_status="public"),
    ])

    # locate state object
    web = getattr(ui, "web", None)
    if web is None:
        web = getattr(ui, "state", None)
    if web is None:
        web = getattr(ui.app.state, "web", None)
    if web is None:
        # fallback build WebState directly
        from yezatas.web.app import WebState  # type: ignore
        # best effort: create new WebState with same config (not ideal but satisfies spec)
        web = WebState(ui.config, port=8765, token="test-run-token", sign_in=lambda c: None, reader_factory=lambda creds: ui.reader, translate_tools=lambda cfg: (translator, ScriptedVerifier(up=False)))  # type: ignore

    result = web._run_translate(vid, "en", ("es",), "sid", lambda line: None)
    # translator got rules
    assert translator.last_glossary is not None
    assert "Mesa" in translator.last_glossary
    assert "Keep these terms exactly as written" in translator.last_glossary

    # now break file and run again
    glossary_path(ui).write_text("{not json", encoding="utf-8")
    translator2 = RecordingTranslator()
    # need to inject new translator into same web or create new ui
    # easiest: update web's translate_tools to return new translator
    # monkey patch _translate_tools if present
    orig = web._translate_tools
    web._translate_tools = lambda cfg: (translator2, ScriptedVerifier(up=False))  # type: ignore
    result2 = web._run_translate(vid, "en", ("es",), "sid", lambda line: None)
    web._translate_tools = orig  # type: ignore

    assert result2.message.endswith("(the glossary file could not be read, so it was not used)")


# ---------------------------------------------------------------------------
# outdated translations
# ---------------------------------------------------------------------------

def _hash(title: str, description: str) -> str:
    from yezatas.translate import source_hash as _sh
    return _sh(title, description)


def _add_translation(store: Store, video_id: str, language: str, source_hash: str, status: str = "verified") -> None:
    from yezatas.store import TranslationRecord
    store.add_translation(TranslationRecord(
        id=0, video_id=video_id, language=language, source_language="en",
        source_hash=source_hash, title="t", description="d",
        translator="m", verifier="", status=status, score=None,
        issues=[], problems=[], back_translation={}, input_tokens=0, output_tokens=0,
        created_at="",
    ))


def test_outdated_not_listed_when_hash_matches(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    vid = "vid00000001"
    title = "Cactus tips"
    desc = "How to repot https://shop.invalid/item"
    seed_catalog(ui.config, [make_video(1, video_id=vid, title=title, description=desc, privacy_status="public")])
    h = _hash(title, desc)
    with Store(ui.config.catalog_db) as store:
        _add_translation(store, vid, "es", h)
    html = page(ui)
    assert 'id="outdated-none"' in html
    assert "Every translation matches its video&#39;s current text." in html


def test_outdated_listed_when_video_text_changed(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    vid = "vid00000001"
    title = "Cactus tips"
    desc = "How to repot https://shop.invalid/item"
    seed_catalog(ui.config, [make_video(1, video_id=vid, title=title, description=desc, privacy_status="public")])
    other_hash = _hash("old", "old desc")
    with Store(ui.config.catalog_db) as store:
        _add_translation(store, vid, "es", other_hash)
    html = page(ui)
    assert 'id="outdated-none"' not in html
    # language name appears and link to video
    assert f"/videos/{vid}#translate" in html
    # lang_name for es is Spanish
    from yezatas.web.views import lang_name
    assert lang_name("es") in html


def test_outdated_uses_draft_text(tmp_path: Path) -> None:
    from yezatas import edits
    ui = make_ui(tmp_path)
    vid = "vid00000001"
    title = "Cactus tips"
    desc = "How to repot https://shop.invalid/item"
    seed_catalog(ui.config, [make_video(1, video_id=vid, title=title, description=desc, privacy_status="public")])
    draft_title = "Mesa draft"
    draft_desc = "Draft https://shop.invalid/item2"
    draft_hash = _hash(draft_title, draft_desc)
    catalog_hash = _hash(title, desc)
    # draft
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, vid, draft_title, draft_desc, [])
    # translation of draft text should NOT be outdated
    with Store(ui.config.catalog_db) as store:
        _add_translation(store, vid, "es", draft_hash)
    html = page(ui)
    assert 'id="outdated-none"' in html
    # now add translation of catalog text - should be outdated when draft exists
    # need a different language to avoid latest override
    with Store(ui.config.catalog_db) as store:
        _add_translation(store, vid, "de", catalog_hash)
    html2 = page(ui)
    assert f"/videos/{vid}#translate" in html2
    from yezatas.web.views import lang_name
    assert lang_name("de") in html2


def test_outdated_failed_never_listed(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    vid = "vid00000001"
    title = "Cactus"
    desc = "Mesa https://shop.invalid/item"
    seed_catalog(ui.config, [make_video(1, video_id=vid, title=title, description=desc, privacy_status="public")])
    other_hash = _hash("other", "other")
    with Store(ui.config.catalog_db) as store:
        _add_translation(store, vid, "es", other_hash, status="failed")
    html = page(ui)
    assert 'id="outdated-none"' in html
    assert f"/videos/{vid}#translate" not in html


def test_outdated_subtitles(tmp_path: Path) -> None:
    from webhelpers import seed_transcript
    from yezatas.store import SubtitleTranslationRecord
    ui = make_ui(tmp_path)
    vid = "vid00000001"
    seed_catalog(ui.config, [make_video(1, video_id=vid, title="Cactus", description="Mesa https://shop.invalid/item", privacy_status="public")])
    tid1 = seed_transcript(ui.config, vid, segments=2)
    tid2 = seed_transcript(ui.config, vid, segments=2)
    segs = [{"start_ms": 0, "end_ms": 1000, "text": "hi"}]
    with Store(ui.config.catalog_db) as store:
        store.add_subtitle_translation(SubtitleTranslationRecord(
            id=0, video_id=vid, transcript_id=tid1, language="es", source_language="en",
            segments=segs, translator="m", verifier="", status="verified", score=None,
            issues=[], problems=[], back_translation={}, input_tokens=0, output_tokens=0, created_at=""))
    html = page(ui)
    # outdated subtitles should be listed (older transcript, newest has none in es)
    from yezatas.web.views import lang_name
    assert lang_name("es") in html
    assert f"/videos/{vid}#translate" in html
    # now add translation for latest transcript in same language -> not outdated
    with Store(ui.config.catalog_db) as store:
        store.add_subtitle_translation(SubtitleTranslationRecord(
            id=0, video_id=vid, transcript_id=tid2, language="es", source_language="en",
            segments=segs, translator="m", verifier="", status="verified", score=None,
            issues=[], problems=[], back_translation={}, input_tokens=0, output_tokens=0, created_at=""))
    html2 = page(ui)
    # should be empty again (no outdated subtitle languages)
    # there are no title translations, so overall none
    assert 'id="outdated-none"' in html2


def test_outdated_limit(tmp_path: Path) -> None:
    from yezatas.web.views import build_outdated_view
    ui = make_ui(tmp_path)
    vids = [f"vid0000000{i}" for i in range(1, 4)]
    for i, vid in enumerate(vids, start=1):
        seed_catalog(ui.config, [make_video(i, video_id=vid, title=f"Cactus {i}", description="Mesa https://shop.invalid/item", privacy_status="public")])
    other_hash = _hash("old", "old")
    with Store(ui.config.catalog_db) as store:
        for vid in vids:
            _add_translation(store, vid, "es", other_hash)
        view = build_outdated_view(store, limit=2)
        assert len(view.rows) == 2
        assert view.more == 1
