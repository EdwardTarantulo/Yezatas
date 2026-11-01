"""Translations in the web UI: the video page card, the translate job, adding and removing
translations in the draft, the Changes diff, the push and history notes, and the Setup step.
The translator and the checker are fakes; nothing touches the network or Ollama."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest

from fakes import ScriptedTranslator, ScriptedVerifier
from webhelpers import make_push_ui, make_ui, make_video, seed_catalog
from yezatas import agent, edits, translate
from yezatas.store import Store, TranslationRecord
from yezatas.translate import TranslateError, Verdict
from yezatas.web import views

FAKE_KEY = "FAKE-KEY-0123456789abcdef"
V1, V2 = "vidtr0001", "vidtr0002"
DESCRIPTION = "Repotting a cactus.\n0:00 Intro\nMore at https://example.com/cactus #cactus"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)
    monkeypatch.delenv(translate.TRANSLATOR_KEY_ENV, raising=False)


class Tools:
    """The injected `translate_tools`: hands out the scripted translator and checker."""

    def __init__(self, translator: Any = "default", verifier: Any = "default") -> None:
        self.translator = ScriptedTranslator() if translator == "default" else translator
        self.verifier = ScriptedVerifier() if verifier == "default" else verifier

    def __call__(self, config: Any) -> Any:
        return self.translator, self.verifier


def make_tr_ui(tmp_path: Path, tools: Tools | None = None, *, language: str | None = "en",
               localizations: dict | None = None):
    tools = tools or Tools()
    ui = make_ui(tmp_path, translate_tools=tools)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus repotting", description=DESCRIPTION,
                   tags=["cactus"], privacy_status="public", default_language=language,
                   localizations=localizations or {}),
        make_video(2, video_id=V2, title="Second video", description="Plain text.", tags=[],
                   privacy_status="public"),
    ])
    return ui, tools


def nf(ui: Any, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def finish(ui: Any) -> None:
    ui.web.translate_job.wait()


def card(html: str) -> str:
    return html.split('id="translations"')[1].split("</section>")[0]


def item(html: str, lang: str) -> str:
    """The result list entry of one language, from the video page or its card."""
    rest = html.split(f'id="tr-{lang}"')[1]
    return re.split(r'<li class="tr-item"|<div class="actions">', rest)[0]


def run_translate(ui: Any, video_id: str = V1, source: str = "en", targets: list[str] | None = None):
    resp = nf(ui, f"/videos/{video_id}/translate",
              data={"source_language": source, "targets": targets or ["es", "de"]})
    finish(ui)
    return resp


def records(ui: Any, video_id: str = V1) -> dict[str, TranslationRecord]:
    with Store(ui.config.catalog_db) as store:
        return store.latest_translations(video_id)


def draft_of(ui: Any, video_id: str = V1) -> Any:
    with Store(ui.config.catalog_db) as store:
        return store.get_draft(video_id)


# --- the card -----------------------------------------------------------------------------------
def test_the_card_names_the_language_models_and_the_privacy_rules(tmp_path: Path) -> None:
    ui, tools = make_tr_ui(tmp_path)
    html = ui.client.get(f"/videos/{V1}").text
    section = card(html)
    assert "<h2 id=\"h-translate\">Translations</h2>" in section
    assert '<strong id="tr-language">English</strong>' in section
    assert "spark-open-test" in section and "glimmer-test through Ollama" in section
    assert "Nothing goes to YouTube until you approve and push it." in section
    assert "a few minutes" not in section  # that sentence belongs to the running state
    assert f'action="/videos/{V1}/translate"' in section
    assert "<script>" not in html and "style=" not in html
    assert html.index('id="suggest"') < html.index('id="translations"') < html.index('id="edit"')


def test_the_privacy_sentence_names_baseten_or_metas_api(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    translate.save_translator_key(ui.config, FAKE_KEY)
    assert "spark-open-test on Baseten" in card(ui.client.get(f"/videos/{V1}").text)
    translate.delete_translator_key(ui.config)
    agent.save_api_key(ui.config, "FAKE-META-KEY")
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert "through Meta's Model API to be translated. Meta's terms apply (<a href=\"/setup#ai-data\">what Meta may do with it</a>)." in section
    assert "FAKE-META-KEY" not in section


def test_without_a_translator_the_button_is_disabled_and_setup_is_linked(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(translator=None))
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert 'id="translate-no-key"' in section and 'href="/setup#step-translate"' in section
    assert re.search(r'class="button primary" disabled>Translate<', section)


def test_with_a_translator_the_button_is_enabled(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert 'id="translate-no-key"' not in section and 'id="translate-no-checker"' not in section
    assert re.search(r'class="button primary">Translate<', section)


def test_without_the_checker_the_card_says_translations_will_be_unverified(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(verifier=ScriptedVerifier(up=False)))
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert 'id="translate-no-checker"' in section and 'marked "Unverified"' in section
    assert 'href="/setup#step-translate"' in section
    assert re.search(r'class="button primary">Translate<', section)  # still possible


def test_a_video_without_a_language_asks_for_one_and_never_guesses(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, language=None)
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert '<strong id="tr-language">Not set</strong>' in section
    assert '<select id="tr-source" name="source_language" required>' in section
    assert '<option value="" selected>Choose the language of this video</option>' in section
    assert " selected>English</option>" not in section and 'id="tr-language-hint"' in section


def test_the_video_language_is_preselected_and_the_default_targets_are_ticked(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert '<option value="en" selected>English</option>' in section
    assert 'value=""' not in section and " required" not in section
    for code in translate.DEFAULT_TARGETS:
        assert f'name="targets" value="{code}" checked' in section
    assert 'name="targets" value="it" checked' not in section
    assert 'name="targets" value="it">' in section
    assert 'name="targets" value="en"' in section  # every language is offered


def test_the_source_language_is_not_a_ticked_target(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, language="es")
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert '<option value="es" selected>Spanish</option>' in section
    assert 'name="targets" value="es" checked' not in section
    assert 'name="targets" value="de" checked' in section


def test_a_regional_video_language_preselects_its_base_language(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, language="en-GB")
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert '<option value="en" selected>English</option>' in section
    assert "English (en-GB)" in section  # the label keeps the exact code


def test_languages_already_on_youtube_are_marked_and_the_own_language_is_not(tmp_path: Path) -> None:
    loc = {"de-DE": {"title": "T", "description": "D"}, "fr": {"title": "T", "description": "D"},
           "en": {"title": "Cactus repotting", "description": DESCRIPTION}}
    ui, _ = make_tr_ui(tmp_path, localizations=loc)
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert "(on YouTube as de-DE)" in section and "(on YouTube)" in section
    assert section.count("on YouTube") >= 2
    en_label = re.search(r'name="targets" value="en">\s*<span>English(.*?)</span>', section, re.S)
    assert en_label and "on YouTube" not in en_label.group(1)
    assert 'name="targets" value="es" checked> <span>Spanish</span>' in section
    assert "Translations in the draft" not in section  # no draft: no list, and no own language


def test_the_card_uses_the_drafts_language_over_the_videos(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, language=None)
    with Store(ui.config.catalog_db) as store:
        edits.set_language(store, V1, "es")
    section = card(ui.client.get(f"/videos/{V1}").text)
    assert '<strong id="tr-language">Spanish</strong>' in section
    assert '<option value="es" selected>Spanish</option>' in section


# --- the translate job --------------------------------------------------------------------------
def test_translating_runs_the_job_and_stores_the_records(tmp_path: Path) -> None:
    ui, tools = make_tr_ui(tmp_path)
    resp = run_translate(ui, targets=["de", "es"])
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#translations"
    assert [c[1] for c in tools.translator.calls] == ["es", "de"] or \
        {c[1] for c in tools.translator.calls} == {"es", "de"}
    assert all(c[0] == "en" for c in tools.translator.calls)
    found = records(ui)
    assert set(found) == {"es", "de"} and all(r.status == "verified" for r in found.values())
    assert found["es"].title == "[es] Cactus repotting" and found["es"].source_language == "en"
    html = ui.client.get(f"/videos/{V1}").text
    assert "Translation finished: 2 verified." in html  # the flash, shown once
    assert "Translation finished" not in ui.client.get(f"/videos/{V1}").text
    assert draft_of(ui) is None  # translating never makes a draft by itself


def test_the_job_uses_the_drafts_text(tmp_path: Path) -> None:
    ui, tools = make_tr_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "Better cactus title", DESCRIPTION, ["cactus"])
    run_translate(ui, targets=["es"])
    assert tools.translator.calls[0][2] == "Better cactus title"


def test_the_card_shows_progress_while_the_job_runs(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(source: str, target: str, title: str, description: str) -> Any:
        entered.set()
        assert gate.wait(10)
        return translate.Translated("T", "D", "spark-open-test")

    ui, tools = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": slow})))
    nf(ui, f"/videos/{V1}/translate", data={"source_language": "en", "targets": ["es"]})
    assert entered.wait(10)
    html = ui.client.get(f"/videos/{V1}").text
    assert 'id="translate-msg"' in html and "Translating 1 language" in html
    assert "a few minutes" in html and 'data-poll="1"' in html
    assert 'action="/videos/vidtr0001/translate"' not in card(html)  # no second start from this page
    other = ui.client.get(f"/videos/{V2}").text
    assert 'id="translate-busy"' in other and 'data-poll="1"' in other
    assert re.search(r'class="button primary" disabled>Translate<', card(other))
    assert ui.client.get("/api/status").json()["translate"]["status"] == "running"
    gate.set()
    finish(ui)
    assert ui.client.get("/api/status").json()["translate"]["status"] == "done"


def test_only_one_translation_runs_at_a_time(tmp_path: Path) -> None:
    gate = threading.Event()
    entered = threading.Event()

    def slow(*args: Any) -> Any:
        entered.set()
        assert gate.wait(10)
        return translate.Translated("T", "D", "spark-open-test")

    ui, tools = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": slow})))
    nf(ui, f"/videos/{V1}/translate", data={"source_language": "en", "targets": ["es"]})
    assert entered.wait(10)
    nf(ui, f"/videos/{V2}/translate", data={"source_language": "en", "targets": ["de"]})
    assert "A translation is already running." in ui.client.get(f"/videos/{V2}").text
    gate.set()
    finish(ui)
    assert set(records(ui, V2)) == set()
    assert len(tools.translator.calls) == 1


def test_a_translation_may_run_while_a_suggestion_runs(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    assert ui.web.suggest_job is not ui.web.translate_job
    assert ui.web.suggest_job.snapshot().status == "idle"


def test_invalid_form_values_are_refused_with_a_flash_and_nothing_starts(tmp_path: Path) -> None:
    ui, tools = make_tr_ui(tmp_path)
    cases = [
        ({"targets": ["es"]}, "Choose the language of this video first."),
        ({"source_language": "xx", "targets": ["es"]}, "cannot translate from that language"),
        ({"source_language": "en"}, "Pick at least one language"),
        ({"source_language": "en", "targets": ["es", "klingon"]}, "not available"),
        ({"source_language": "en", "targets": ["es", "en"]}, "cannot be translated into its own language"),
    ]
    for data, message in cases:
        resp = nf(ui, f"/videos/{V1}/translate", data=data)
        assert resp.status_code == 303
        html = ui.client.get(f"/videos/{V1}").text
        assert 'class="flash error"' in html and message in html, message
    assert tools.translator.calls == [] and ui.web.translate_job.snapshot().status == "idle"


def test_translate_without_a_translator_flashes_and_starts_nothing(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(translator=None))
    nf(ui, f"/videos/{V1}/translate", data={"source_language": "en", "targets": ["es"]})
    html = ui.client.get(f"/videos/{V1}").text
    assert "There is no translation key yet." in html and "Nothing was sent." in html
    assert 'href="/setup#step-translate"' in html
    assert ui.web.translate_job.snapshot().status == "idle"


def test_translate_for_an_unknown_video_is_a_404(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    assert nf(ui, "/videos/nope/translate",
              data={"source_language": "en", "targets": ["es"]}).status_code == 404
    assert nf(ui, "/videos/nope/translations/add", data={"languages": ["es"]}).status_code == 404


def test_a_job_without_a_translator_fails_with_a_clear_message(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(translator=None))
    assert ui.web.start_translate(V1, "en", ["es"], "sid")
    finish(ui)
    snap = ui.web.translate_job.snapshot()
    refusal = "There is no translation key. Add one in Setup. Nothing was sent."
    assert snap.status == "failed" and snap.message == refusal
    html = ui.client.get(f"/videos/{V1}").text
    assert 'class="flash error"' in html and refusal in html
    assert 'action="/translate/done"' in html
    assert refusal not in ui.client.get(f"/videos/{V2}").text  # only on its own video
    resp = nf(ui, "/translate/done", data={"video_id": V1})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#translations"
    assert refusal not in ui.client.get(f"/videos/{V1}").text


def test_a_translator_error_is_shown_without_the_key(tmp_path: Path) -> None:
    refused = TranslateError("auth", translate.AUTH_MESSAGE)
    ui, _ = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": refused})))
    run_translate(ui, targets=["es"])
    found = records(ui)
    assert found["es"].status == "failed" and found["es"].problems == [translate.AUTH_MESSAGE]
    html = ui.client.get(f"/videos/{V1}").text
    assert translate.AUTH_MESSAGE in html and "Translation finished: 1 failed." in html
    assert 'class="flash error"' in html


def test_a_crash_in_the_job_is_reported_without_details(tmp_path: Path) -> None:
    def boom(config: Any) -> Any:
        raise RuntimeError("secret detail")

    ui = make_ui(tmp_path, translate_tools=boom)
    seed_catalog(ui.config, [make_video(1, video_id=V1, default_language="en")])
    assert ui.web.start_translate(V1, "en", ["es"], "sid")
    finish(ui)
    assert "secret detail" not in ui.web.translate_job.snapshot().message


def test_a_video_without_text_is_reported_by_the_job(tmp_path: Path) -> None:
    ui = make_ui(tmp_path, translate_tools=Tools())
    seed_catalog(ui.config, [make_video(1, video_id=V1, title="", description="", default_language="en")])
    assert ui.web.start_translate(V1, "en", ["es"], "sid")
    finish(ui)
    assert ui.web.translate_job.snapshot().message == "There is no text to translate."


# --- the results --------------------------------------------------------------------------------
def test_results_show_status_score_text_back_translation_and_issues(tmp_path: Path) -> None:
    bad = Verdict(3, ("The price was dropped.",), "Back title", "Back text", "glimmer-test")
    ui, _ = make_tr_ui(tmp_path, Tools(verifier=ScriptedVerifier({"de": bad})))
    run_translate(ui, targets=["es", "de"])
    section = card(ui.client.get(f"/videos/{V1}").text)
    es = item(section, "es")
    assert "Spanish" in es and 'class="pill done">Verified' in es and "Score 5/5" in es
    assert "[es] Cactus repotting" in es and "<details" in es and "[es] Repotting a cactus." in es
    assert "Back-translation" in es
    assert 'name="languages" value="es" checked> Add to the draft' in es
    de = item(section, "de")
    assert 'class="pill warn">Flagged' in de and "Score 3/5" in de
    assert "The price was dropped." in de and "Back title" in de and "Back text" in de
    assert 'name="languages" value="de"> I checked this translation' in de  # not ticked
    # LANGUAGES order, not the order they were picked in
    assert section.index('id="tr-es"') < section.index('id="tr-de"')
    assert f'action="/videos/{V1}/translations/add"' in section


def test_unverified_results_show_the_checkers_note(tmp_path: Path) -> None:
    down = ScriptedVerifier(up=True, verdicts={"es": TranslateError("answer", "The checker copied the text.")})
    ui, _ = make_tr_ui(tmp_path, Tools(verifier=down))
    run_translate(ui, targets=["es"])
    es = item(ui.client.get(f"/videos/{V1}").text, "es")
    assert 'class="pill unverified">Unverified' in es and "The checker copied the text." in es
    assert 'name="languages" value="es"> I checked this translation' in es


def test_results_without_the_checker_are_unverified_and_unticked(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(verifier=ScriptedVerifier(up=False)))
    run_translate(ui, targets=["es"])
    es = item(ui.client.get(f"/videos/{V1}").text, "es")
    assert "Unverified" in es and "Score" not in es
    assert 'name="languages" value="es"> I checked this translation' in es


def test_a_failed_result_lists_its_problem_and_cannot_be_added(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": TranslateError("network", "No connection.")})))
    run_translate(ui, targets=["es", "de"])
    section = card(ui.client.get(f"/videos/{V1}").text)
    es = item(section, "es")
    assert 'class="pill problem">Failed' in es and "No connection." in es
    assert 'name="languages"' not in es
    assert 'name="languages" value="de"' in section


def test_a_flagged_code_check_problem_is_listed(tmp_path: Path) -> None:
    lost = translate.Translated("[es] t", "no link here", "spark-open-test")
    ui, _ = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": lost})))
    run_translate(ui, targets=["es"])
    es = item(ui.client.get(f"/videos/{V1}").text, "es")
    assert "Flagged" in es and "is missing from the translation" in es
    assert 'name="languages" value="es"> I checked this translation' in es


def test_a_result_made_from_older_text_is_marked_and_not_offered(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "A new title", DESCRIPTION, ["cactus"])
    section = card(ui.client.get(f"/videos/{V1}").text)
    es = item(section, "es")
    assert "Older text: translate again" in es and 'name="languages"' not in es
    assert re.search(r'class="button primary" disabled>Add the ticked', section)


def test_a_result_already_in_the_draft_is_marked_and_not_ticked(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    es = item(ui.client.get(f"/videos/{V1}").text, "es")
    assert "Already in the draft" in es and 'name="languages" value="es"> Add to the draft' in es


# --- adding to the draft ------------------------------------------------------------------------
def test_adding_verified_translations_makes_a_draft_with_the_language(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, language=None)
    run_translate(ui, targets=["es", "de"])
    resp = nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "de"]})
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#translations"
    row = draft_of(ui)
    assert row.default_language == "en" and set(row.localizations) == {"es", "de"}
    assert row.localizations["es"] == {"title": "[es] Cactus repotting",
                                       "description": records(ui)["es"].description}
    assert (row.title, row.description) == (row.base_title, row.base_description)
    html = ui.client.get(f"/videos/{V1}").text
    assert "Added 2 translations to the draft. Review them on the Changes page." in html
    assert row.approved_at is None


def test_a_single_translation_flash_is_singular(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    assert "Added 1 translation to the draft." in ui.client.get(f"/videos/{V1}").text


def test_only_ticked_translations_are_added_flagged_ones_when_ticked(tmp_path: Path) -> None:
    bad = Verdict(2, ("A fact changed.",), "b", "b", "glimmer-test")
    ui, _ = make_tr_ui(tmp_path, Tools(verifier=ScriptedVerifier({"de": bad})))
    run_translate(ui, targets=["es", "de", "fr"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    assert set(draft_of(ui).localizations) == {"es"}  # the flagged ones were not ticked
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["de"]})  # now ticked on purpose
    assert set(draft_of(ui).localizations) == {"es", "de"}


def test_adding_nothing_is_refused(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={})
    assert "Tick at least one translation to add." in ui.client.get(f"/videos/{V1}").text
    assert draft_of(ui) is None


def test_failed_and_unknown_translations_cannot_be_added_even_when_posted(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(ScriptedTranslator({"es": TranslateError("api", "Broke.")})))
    run_translate(ui, targets=["es", "de"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "de"]})
    assert "The Spanish translation failed, so it cannot be added." in ui.client.get(f"/videos/{V1}").text
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["fr"]})
    assert "There is no French translation to add." in ui.client.get(f"/videos/{V1}").text
    assert draft_of(ui) is None


def test_older_text_cannot_be_added_even_when_posted(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "A new title", DESCRIPTION, ["cactus"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    html = ui.client.get(f"/videos/{V1}").text
    assert "made from older text" in html
    assert draft_of(ui).localizations is None


def test_translations_from_different_source_languages_are_refused(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, source="en", targets=["es"])
    run_translate(ui, source="de", targets=["fr"])  # same text, another declared language
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "fr"]})
    assert "different source languages" in ui.client.get(f"/videos/{V1}").text
    assert draft_of(ui) is None
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["fr"]})
    assert draft_of(ui).default_language == "de"


def test_a_new_translation_replaces_the_regional_key_studio_uses(tmp_path: Path) -> None:
    loc = {"de-DE": {"title": "Alt", "description": "Alt"}, "fr": {"title": "Vieux", "description": ""}}
    ui, _ = make_tr_ui(tmp_path, language="en-GB", localizations=loc)
    run_translate(ui, source="en", targets=["de", "es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["de", "es"]})
    row = draft_of(ui)
    assert set(row.localizations) == {"de-DE", "es", "fr"}  # German replaced de-DE, fr kept
    assert row.localizations["de-DE"]["title"] == "[de] Cactus repotting"
    assert row.default_language is None  # an "en" text on an en-GB video keeps the video's code
    assert "Video language" not in ui.client.get("/changes").text
    assert "Saved as de-DE" in card(ui.client.get(f"/videos/{V1}").text)


def test_draft_text_edits_after_adding_keep_the_translations(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    nf(ui, f"/videos/{V1}/draft", data={"title": "Edited", "description": DESCRIPTION, "tags": "cactus"})
    assert set(draft_of(ui).localizations) == {"es"}


# --- the draft's translations and removing -------------------------------------------------------
def test_the_draft_translations_are_listed_with_a_remove_button(tmp_path: Path) -> None:
    loc = {"fr": {"title": "Vieux", "description": ""}, "en": {"title": "Cactus repotting", "description": DESCRIPTION}}
    ui, _ = make_tr_ui(tmp_path, localizations=loc)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    section = card(ui.client.get(f"/videos/{V1}").text)
    listing = section.split("Translations in the draft")[1]
    assert "Spanish" in listing and "New in this draft" in listing
    assert "French" in listing and "On YouTube" in listing
    assert "English" not in listing  # the video's own language is not a translation
    assert f'action="/videos/{V1}/translations/es/remove"' in listing
    assert f'action="/videos/{V1}/translations/fr/remove"' in listing
    assert "deletes it there once you approve and push" in listing


def test_removing_a_translation_takes_it_out_of_the_draft(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es", "de"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "de"]})
    resp = nf(ui, f"/videos/{V1}/translations/es/remove")
    assert resp.status_code == 303 and resp.headers["location"] == f"/videos/{V1}#translations"
    assert set(draft_of(ui).localizations) == {"de"}
    assert "Removed the Spanish translation from the draft." in ui.client.get(f"/videos/{V1}").text
    nf(ui, f"/videos/{V1}/translations/es/remove")
    assert "There is no es translation to remove." in ui.client.get(f"/videos/{V1}").text


def test_removing_the_last_new_translation_drops_the_draft(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    run_translate(ui, targets=["es"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es"]})
    nf(ui, f"/videos/{V1}/translations/es/remove")
    assert draft_of(ui) is None  # the language alone is not worth a draft once nothing else differs
    assert "Translations in the draft" not in ui.client.get(f"/videos/{V1}").text


def test_removing_a_regional_translation_uses_its_exact_key(tmp_path: Path) -> None:
    loc = {"de-DE": {"title": "Alt", "description": ""}}
    ui, _ = make_tr_ui(tmp_path, localizations=loc)
    run_translate(ui, targets=["de"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["de"]})
    assert f'action="/videos/{V1}/translations/de-DE/remove"' in ui.client.get(f"/videos/{V1}").text
    nf(ui, f"/videos/{V1}/translations/de-DE/remove")
    assert "Removed the German (de-DE) translation" in ui.client.get(f"/videos/{V1}").text
    assert draft_of(ui).localizations == {}


# --- Changes, push and history -----------------------------------------------------------------
def changes_after_adding(tmp_path: Path, **kw: Any):
    ui, _ = make_tr_ui(tmp_path, **kw)
    run_translate(ui, targets=["es", "de"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "de"]})
    return ui


def test_changes_shows_the_language_and_the_added_translations(tmp_path: Path) -> None:
    ui = changes_after_adding(tmp_path, language=None)
    html = ui.client.get("/changes").text
    assert "Changes: video language; translations (2 added)" in html
    assert "Video language: not set &rarr; English" in html
    block = html.split("<h3 class=\"diff-h\">Translations</h3>")[1]
    assert "Spanish" in block and "German" in block
    assert "[es] Cactus repotting" in block and "<details" in block and "[de] Repotting a cactus." in block
    assert "stay as they are" not in block and "stays as it is" not in block


def test_changes_shows_changed_removed_and_kept_translations(tmp_path: Path) -> None:
    loc = {"de": {"title": "Alt", "description": "Alt"}, "fr": {"title": "F", "description": "F"},
           "it": {"title": "I", "description": "I"}, "pt": {"title": "P", "description": "P"},
           "en": {"title": "Cactus repotting", "description": DESCRIPTION}}
    ui, _ = make_tr_ui(tmp_path, localizations=loc)
    run_translate(ui, targets=["es", "de"])
    nf(ui, f"/videos/{V1}/translations/add", data={"languages": ["es", "de"]})
    nf(ui, f"/videos/{V1}/translations/fr/remove")
    html = ui.client.get("/changes").text
    assert "translations (1 added, 1 changed, 1 removed)" in html
    block = html.split("<h3 class=\"diff-h\">Translations</h3>")[1]
    assert "Removed translations:" in block and "French" in block
    assert "2 other translations stay as they are." in block  # it and pt; the own language en is not counted


def test_changes_without_translation_changes_has_no_translation_block(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, localizations={"fr": {"title": "F", "description": "F"}})
    with Store(ui.config.catalog_db) as store:
        edits.save_draft(store, V1, "Another title", DESCRIPTION, ["cactus"])
    html = ui.client.get("/changes").text
    assert "Changes: title" in html and ">Translations</h3>" not in html
    assert "stay as they are" not in html and "Video language" not in html


def test_summarize_diff_for_language_and_translations(tmp_path: Path) -> None:
    from yezatas.diffs import DraftDiff, LocalizationsDiff, diff_draft
    from yezatas.validate import Draft

    base = diff_draft(Draft("t", "d", ()), Draft("t", "d", ()))
    assert views.summarize_diff(base) == "No changes."
    only_language = DraftDiff(base.title, False, base.description, base.tags, True, ("en", "es"))
    assert views.summarize_diff(only_language) == "Changes: video language"
    loc = LocalizationsDiff(True, ("de", "es"), ("fr",), ("it",), ())
    both = DraftDiff(base.title, False, base.description, base.tags, True, None, loc)
    assert views.summarize_diff(both) == "Changes: translations (2 added, 1 changed, 1 removed)"


def push_ready(tmp_path: Path):
    ui, writer = make_push_ui(tmp_path, count=1, translate_tools=Tools())
    return ui, writer


def test_the_push_page_mentions_translations_and_the_language(tmp_path: Path) -> None:
    ui, _ = push_ready(tmp_path)
    with Store(ui.config.catalog_db) as store:
        video_id = store.list_videos()[0].video_id
        edits.add_translations(store, video_id, "es", {"fr": ("Salut", "Texte")})
        edits.approve(store, video_id)
    html = ui.client.get("/changes/push").text
    assert "Translations replace the video's whole translation list on YouTube" in html
    assert "the ones shown as kept are sent again unchanged." in html
    assert "The video language is set on YouTube too." in html


def test_the_push_page_stays_quiet_for_text_only_drafts(tmp_path: Path) -> None:
    ui, _ = push_ready(tmp_path)
    with Store(ui.config.catalog_db) as store:
        video = store.list_videos()[0]
        edits.save_draft(store, video.video_id, video.title + " new", video.description, list(video.tags))
        edits.approve(store, video.video_id)
    html = ui.client.get("/changes/push").text
    assert "push-confirm" in html and "Translations replace" not in html and "video language" not in html


def test_history_lists_the_translations_in_a_backup(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    before = {"title": "Old title", "description": "Old", "tags": [], "defaultLanguage": "en",
              "localizations": {"es": {"title": "a", "description": "b"},
                                "de": {"title": "a", "description": "b"},
                                "en": {"title": "a", "description": "b"}}}
    with Store(ui.config.catalog_db) as store:
        store.begin_push(V1, "manual", before)
        store.begin_push(V2, "manual", {"title": "Plain", "description": "", "tags": []})
    html = ui.client.get("/history").text
    assert "Backup includes translations: de, es" in html
    assert html.count("Backup includes translations") == 1


# --- Setup --------------------------------------------------------------------------------------
def test_setup_shows_the_translation_step_without_a_key(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path, Tools(translator=None, verifier=ScriptedVerifier(up=False)))
    html = ui.setup_html()
    assert html.index('id="step-ai"') < html.index('id="step-translate"')
    section = html.split('id="step-translate"')[1].split("</section>")[0]
    assert "Translations (optional)" in section and "No key" in section
    assert 'action="/setup/translator-key"' in section and 'type="password"' in section
    assert "/setup/translator-key/delete" not in section
    assert ui.config.verifier_model in section and "Not running" in section and "install Ollama" in section
    assert "only talks to it on this computer" in section


def test_setup_shows_a_ready_checker(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    section = ui.setup_html().split('id="step-translate"')[1].split("</section>")[0]
    assert "Ready" in section and "Not running" not in section


def test_setup_checker_state_follows_the_checker(tmp_path: Path) -> None:
    tools = Tools()
    ui, _ = make_tr_ui(tmp_path, tools)
    assert "Ready" in ui.setup_html().split('id="step-translate"')[1]
    tools.verifier.up = False
    assert "Not running" in ui.setup_html().split('id="step-translate"')[1]


def test_setup_says_when_the_meta_key_is_the_fallback(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    agent.save_api_key(ui.config, "FAKE-META-KEY")
    section = ui.setup_html().split('id="step-translate"')[1].split("</section>")[0]
    assert "Using the Meta key" in section and "your Meta API key" in section
    assert 'type="password"' in section and "/setup/translator-key/delete" not in section
    assert "FAKE-META-KEY" not in section


def test_saving_the_translation_key_writes_the_file_and_never_shows_it(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    resp = nf(ui, "/setup/translator-key", data={"api_key": f"  {FAKE_KEY}  "})
    assert resp.status_code == 303 and resp.headers["location"] == "/setup#step-translate"
    assert ui.config.translator_key_path.read_text(encoding="utf-8") == FAKE_KEY
    assert translate.translator_key_source(ui.config) == "saved"
    setup = ui.setup_html()
    assert "Saved the key on this computer. It is not shown again." in setup and "Key saved" in setup
    assert "/setup/translator-key/delete" in setup and "Replace the saved key" in setup
    for path in ("/setup", "/", f"/videos/{V1}", "/changes", "/history", "/videos", "/api/status", "/agent"):
        body = ui.client.get(path).text
        assert FAKE_KEY not in body and FAKE_KEY[:12] not in body, path
    assert FAKE_KEY not in resp.text
    assert FAKE_KEY.encode() not in ui.config.catalog_db.read_bytes()


@pytest.mark.parametrize("bad", ["", "   ", "two words", "a\nb", "x" * 501])
def test_bad_translation_keys_are_refused(tmp_path: Path, bad: str) -> None:
    ui, _ = make_tr_ui(tmp_path)
    nf(ui, "/setup/translator-key", data={"api_key": bad})
    assert translate.translator_key_source(ui.config) == ""
    html = ui.setup_html()
    assert 'class="flash error"' in html and ("Paste your key first." in html or "one line" in html)


def test_removing_the_saved_translation_key(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    translate.save_translator_key(ui.config, FAKE_KEY)
    resp = nf(ui, "/setup/translator-key/delete")
    assert resp.status_code == 303 and resp.headers["location"] == "/setup#step-translate"
    assert not ui.config.translator_key_path.exists()
    assert "The saved key was deleted." in ui.setup_html()
    nf(ui, "/setup/translator-key/delete")
    assert "There was no saved key." in ui.setup_html()


def test_a_translation_key_from_the_environment_hides_the_form(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    ui, _ = make_tr_ui(tmp_path)
    monkeypatch.setenv(translate.TRANSLATOR_KEY_ENV, FAKE_KEY)
    html = ui.setup_html()
    section = html.split('id="step-translate"')[1].split("</section>")[0]
    assert "Key from the environment" in section and translate.TRANSLATOR_KEY_ENV in section
    assert 'type="password"' not in section and "/setup/translator-key" not in section
    assert FAKE_KEY not in html
    nf(ui, "/setup/translator-key", data={"api_key": "FAKE-OTHER-KEY"})
    assert not ui.config.translator_key_path.exists()
    assert "so nothing was saved" in ui.setup_html()


def test_the_setup_flash_links_to_the_translation_step(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    nf(ui, "/setup/translator-key", data={"api_key": ""})
    assert 'href="#step-translate"' in ui.setup_html()


# --- the default tools --------------------------------------------------------------------------
def test_the_default_tools_use_the_configured_translator_and_checker(tmp_path: Path) -> None:
    from yezatas.web.app import default_translate_tools

    ui = make_ui(tmp_path)
    translator, verifier = default_translate_tools(ui.config)
    assert translator is None  # no key in this environment
    assert verifier.model == ui.config.verifier_model


# --- status, security, markup ---------------------------------------------------------------------
def test_api_status_has_the_translate_job(tmp_path: Path) -> None:
    ui, _ = make_tr_ui(tmp_path)
    assert ui.client.get("/api/status").json()["translate"] == {
        "status": "idle", "message": "", "error_kind": ""}


def test_the_poll_script_knows_the_translate_job() -> None:
    js = (Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static" / "app.js").read_text(
        encoding="utf-8")
    assert '"translate"' in js


POSTS = ["/videos/vidtr0001/translate", "/translate/done", "/videos/vidtr0001/translations/add",
         "/videos/vidtr0001/translations/es/remove", "/setup/translator-key",
         "/setup/translator-key/delete"]


@pytest.mark.parametrize("path", POSTS)
def test_every_new_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, tools = make_tr_ui(tmp_path)
    translate.save_translator_key(ui.config, FAKE_KEY)
    data = {"source_language": "en", "targets": ["es"], "languages": ["es"], "api_key": "FAKE"}
    assert ui.client.post(path, data={}).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong", **data}).status_code == 403
    assert tools.translator.calls == [] and ui.web.translate_job.snapshot().status == "idle"
    assert draft_of(ui) is None and records(ui) == {}
    assert ui.config.translator_key_path.read_text(encoding="utf-8") == FAKE_KEY


def test_nothing_in_the_translation_flow_reaches_youtube(tmp_path: Path) -> None:
    ui, writer = make_push_ui(tmp_path, count=2, translate_tools=Tools())
    with Store(ui.config.catalog_db) as store:
        video_id = store.list_videos()[0].video_id
    nf(ui, f"/videos/{video_id}/translate", data={"source_language": "en", "targets": ["es"]})
    finish(ui)
    nf(ui, f"/videos/{video_id}/translations/add", data={"languages": ["es"]})
    nf(ui, f"/videos/{video_id}/translations/es/remove")
    assert writer.calls == []


def test_new_markup_has_no_inline_script_or_style(tmp_path: Path) -> None:
    loc = {"fr": {"title": "F", "description": "F"}}
    ui = changes_after_adding(tmp_path, localizations=loc)
    for path in ("/setup", "/changes", "/history", f"/videos/{V1}", f"/videos/{V2}"):
        html = ui.client.get(path).text
        assert "style=" not in html and html.count("<script") == html.count("<script src="), path
        assert not re.search(r"\son\w+=", html), path
