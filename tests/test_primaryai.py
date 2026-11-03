"""Primary AI (Meta's Model API or Muse Glimmer on this computer): the saved choice, the one place
that routes the text model, Self-checked labels, Modal, the sound tagger check and the Configuration
page. Nothing touches the network; the clients are built but never called."""

from __future__ import annotations

import importlib.util
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import ScriptedVerifier
from test_web_strategy import seed
from webhelpers import make_config, make_ui, make_video, seed_catalog
from yezatas import agent, packaging, primaryai, transcribe, translate
from yezatas.agent import ModelError
from yezatas.config import DEFAULT_MODAL_MODEL
from yezatas.store import TranslationRecord
from yezatas.web import views

FAKE_KEY = "FAKE-KEY-0123456789abcdef"
V1 = "vidpa0001"
LOCAL = {"YEZATAS_PRIMARY_AI": "local"}


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)
    monkeypatch.delenv(translate.TRANSLATOR_KEY_ENV, raising=False)


def go_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(primaryai.PRIMARY_AI_ENV, "local")


def card_of(html: str, step: str) -> str:
    return html.split(f'id="{step}"')[1].split("</section>")[0]


# --- the saved choice -------------------------------------------------------------------------------
def test_the_default_is_meta(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert primaryai.load_primary_ai(config) == "meta"
    assert primaryai.primary_ai_source(config) == "" and not primaryai.is_local(config)


def test_a_saved_choice_is_read_back(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    primaryai.save_primary_ai(config, "local")
    assert config.primary_ai_path.read_text(encoding="utf-8") == "local"
    assert primaryai.load_primary_ai(config) == "local"
    assert primaryai.primary_ai_source(config) == "saved"
    primaryai.save_primary_ai(config, "meta")
    assert primaryai.load_primary_ai(config) == "meta"


def test_the_environment_wins_over_the_saved_choice(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    primaryai.save_primary_ai(config, "meta")
    assert primaryai.load_primary_ai(config, LOCAL) == "local"
    assert primaryai.primary_ai_source(config, LOCAL) == "environment"
    # a value that is neither is ignored
    assert primaryai.load_primary_ai(config, {"YEZATAS_PRIMARY_AI": "other"}) == "meta"


def test_a_bad_choice_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        primaryai.save_primary_ai(make_config(tmp_path), "other")


def test_a_garbled_saved_file_means_meta(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.credentials_dir.mkdir(parents=True)
    config.primary_ai_path.write_text("???", encoding="utf-8")
    assert primaryai.load_primary_ai(config) == "meta"


def test_the_post_saves_the_choice_and_the_page_shows_it(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.post("/configuration/primary-ai", data={"primary_ai": "local"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/configuration#step-ai"
    assert primaryai.load_primary_ai(ui.config) == "local"
    html = ui.setup_html()
    assert re.search(r'value="local" checked', html) and "Primary AI is now Muse Glimmer" in html


def test_the_post_refuses_a_bad_value_and_an_environment_choice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = make_ui(tmp_path)
    ui.post("/configuration/primary-ai", data={"primary_ai": "bogus"}, follow_redirects=False)
    assert primaryai.primary_ai_source(ui.config) == ""
    monkeypatch.setenv(primaryai.PRIMARY_AI_ENV, "meta")
    ui.post("/configuration/primary-ai", data={"primary_ai": "local"}, follow_redirects=False)
    assert not ui.config.primary_ai_path.exists()
    html = ui.setup_html()
    assert "YEZATAS_PRIMARY_AI" in html and "Save the choice" not in html


# --- one place for the text model --------------------------------------------------------------------
def test_meta_mode_routes_to_meta_with_the_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert primaryai.text_model(config) is None and not primaryai.text_ai_ready(config)
    with pytest.raises(ModelError):
        primaryai.text_client(config)
    agent.save_api_key(config, FAKE_KEY)
    found = primaryai.text_model(config)
    assert found is not None and not found.local
    assert (found.base_url, found.model, found.api_key) == (agent.MODEL_BASE_URL, config.model, FAKE_KEY)
    assert FAKE_KEY not in repr(found)
    assert primaryai.text_model(config, model="muse-wonder-1.0").model == "muse-wonder-1.0"


def test_local_mode_routes_to_ollama_without_a_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    found = primaryai.text_model(config, LOCAL)
    assert found is not None and found.local
    assert found.base_url == f"{config.verifier_url}/v1" and found.model == config.verifier_model
    assert primaryai.text_ai_ready(config, LOCAL)
    # a model picked for Meta is ignored in Local mode
    assert primaryai.text_model(config, LOCAL, model="muse-wonder-1.0").model == config.verifier_model
    client = primaryai.text_client(config, LOCAL)
    assert client.model == config.verifier_model and client.local


def test_local_mode_does_not_send_the_meta_key_to_ollama(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent.save_api_key(config, FAKE_KEY)
    found = primaryai.text_model(config, LOCAL)
    assert found is not None and found.api_key != FAKE_KEY and found.model == config.verifier_model


def test_meta_only_features_are_blocked_only_in_local_mode_without_a_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert primaryai.meta_only_blocked(config, LOCAL)
    assert not primaryai.meta_only_blocked(config, {})
    agent.save_api_key(config, FAKE_KEY)
    assert not primaryai.meta_only_blocked(config, LOCAL)


def test_the_packaging_tools_keep_sam_and_image_on_meta(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent.save_api_key(config, FAKE_KEY)
    tools = packaging.packaging_tools_from_config(config, LOCAL)
    assert tools is not None
    assert tools.vision.model == config.verifier_model
    assert tools.segmenter.model == packaging.SAM_MODEL and tools.maker.model == packaging.IMAGE_MODEL
    assert isinstance(tools.segmenter, packaging.MetaSamClient)


def test_without_a_key_sam_and_image_refuse_with_a_plain_sentence(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    tools = packaging.packaging_tools_from_config(config, LOCAL)
    assert tools is not None and tools.vision.model == config.verifier_model
    with pytest.raises(ModelError, match="Thumbnail variants need Meta's API"):
        tools.maker.edit(b"x", "image/png", "make it brighter")
    with pytest.raises(ModelError, match="Subject and face sizes need Meta's API"):
        tools.segmenter.detect(b"x", "image/png", "a cactus")
    assert packaging.packaging_tools_from_config(config, {}) is None  # Meta mode, no key


# --- translations ------------------------------------------------------------------------------------
def test_the_translator_order_is_baseten_then_modal_then_the_primary_ai(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert translate.translator_from_config(config, {}) is None
    assert translate.translator_key_source(config, {}) == ""
    agent.save_api_key(config, FAKE_KEY)
    assert translate.translator_from_config(config, {}).model == config.model
    assert translate.translator_key_source(config, {}) == "meta"
    chosen = translate.translator_from_config(config, LOCAL)
    assert chosen.model == config.verifier_model and chosen.local
    assert translate.translator_key_source(config, LOCAL) == "local"
    # Modal needs both its key and its endpoint
    translate.save_modal_key(config, "MODAL-KEY-1234")
    assert translate.translator_from_config(config, LOCAL).model == config.verifier_model
    modal_env = {"MODAL_ENDPOINT_URL": "https://example.invalid/v1/"}
    # Local stays local, whatever keys are saved
    assert translate.translator_from_config(config, {**LOCAL, **modal_env}).model == config.verifier_model
    assert translate.translator_from_config(config, modal_env).model == DEFAULT_MODAL_MODEL
    assert translate.translator_key_source(config, modal_env) == "modal"
    # a Baseten key beats Modal
    translate.save_translator_key(config, "BASETEN-KEY-1234")
    assert translate.translator_from_config(config, modal_env).model == config.translator_model


def test_the_modal_key_is_saved_and_deleted_like_the_baseten_key(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert config.modal_key_path == config.credentials_dir / "modal_api_key"
    assert translate.modal_key_source(config, {}) == ""
    translate.save_modal_key(config, f"  {FAKE_KEY} ")
    assert config.modal_key_path.read_text(encoding="utf-8") == FAKE_KEY
    assert translate.modal_key_source(config, {}) == "saved"
    assert translate.modal_key_source(config, {"MODAL_API_KEY": "from-env"}) == "environment"
    assert translate.load_modal_key(config, {"MODAL_API_KEY": "from-env"}) == "from-env"
    with pytest.raises(ValueError):
        translate.save_modal_key(config, "two words")
    assert translate.delete_modal_key(config) and not translate.delete_modal_key(config)


def test_the_modal_key_form_saves_without_showing_the_key(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.post("/configuration/modal-key", data={"api_key": f"  {FAKE_KEY}  "}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/configuration#step-translate"
    assert ui.config.modal_key_path.read_text(encoding="utf-8") == FAKE_KEY
    html = ui.setup_html()
    assert FAKE_KEY not in html and "A Modal key is saved on this computer" in html
    assert "MODAL_ENDPOINT_URL" in html and DEFAULT_MODAL_MODEL in html
    ui.post("/configuration/modal-key/delete", follow_redirects=False)
    assert not ui.config.modal_key_path.exists()


def test_the_modal_environment_key_hides_the_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_API_KEY", FAKE_KEY)
    ui = make_ui(tmp_path)
    ui.post("/configuration/modal-key", data={"api_key": "other-key"}, follow_redirects=False)
    assert not ui.config.modal_key_path.exists()
    html = ui.setup_html()
    assert 'id="modal-key"' not in html and FAKE_KEY not in html


def record(status: str, translator: str, verifier: str) -> TranslationRecord:
    return TranslationRecord(
        id=1, video_id=V1, language="es", source_language="en", source_hash="h", title="t",
        description="d", translator=translator, verifier=verifier, status=status, score=5,
        issues=[], problems=[], back_translation={}, input_tokens=0, output_tokens=0,
        created_at="2026-01-01T00:00:00+00:00")


def test_a_translation_checked_by_its_own_model_is_self_checked() -> None:
    assert views.translation_label(record("verified", "muse-glimmer:30b", "muse-glimmer:30b"))[0] \
        == "Self-checked"
    assert views.translation_label(record("verified", "meta-models/spark", "muse-glimmer:30b"))[0] \
        == "Verified"
    assert views.translation_label(record("flagged", "muse-glimmer:30b", "muse-glimmer:30b"))[0] \
        == "Flagged"
    assert views.translation_label(record("unverified", "m", ""))[0] == "Unverified"


# --- prices -----------------------------------------------------------------------------------------
def strategy_page(tmp_path: Path) -> str:
    ui = make_ui(tmp_path, strategy=lambda store, home, as_of, report: None)
    seed(ui.config, datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    agent.save_api_key(ui.config, FAKE_KEY)
    resp = ui.client.get("/strategy")
    assert resp.status_code == 200
    return resp.text


def test_prices_show_in_meta_mode_and_are_hidden_in_local_mode(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    meta = strategy_page(tmp_path / "a")
    assert "One run costs about $" in meta and "Meta's Model API (muse-spark-1.3)" in meta
    go_local(monkeypatch)
    local_page = strategy_page(tmp_path / "b")
    assert "One run costs about" not in local_page
    assert "muse-glimmer:30b on this computer. Nothing leaves it." in local_page
    assert "Meta's Model API (" not in local_page


# --- a feature that cannot run ----------------------------------------------------------------------
def packaging_ui(tmp_path: Path) -> Any:
    ui = make_ui(tmp_path, packaging_tools=lambda cfg: packaging.packaging_tools_from_config(cfg),
                 packaging_fetch=lambda url: b"")
    seed_catalog(ui.config, [make_video(1, video_id=V1, title="Cactus", privacy_status="public")])
    return ui


def test_sam_sizes_say_what_to_do_in_local_mode_without_a_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    go_local(monkeypatch)
    ui = packaging_ui(tmp_path)
    section = card_of(ui.client.get(f"/videos/{V1}").text, "thumbnail")
    assert "Subject and face sizes need Meta&#39;s API. Primary AI is Local and no Meta key is saved." \
        in section or "Subject and face sizes need Meta's API. Primary AI is Local and no Meta key is saved." in section
    assert '<a href="/configuration#step-ai">Add a key in Configuration</a>' in section
    assert "Research thumbnail" not in section
    # the route refuses too, and starts nothing
    ui.post(f"/videos/{V1}/packaging", follow_redirects=False)
    assert 'class="flash error"' in ui.client.get(f"/videos/{V1}").text
    assert not ui.web.packaging_job.snapshot().running
    assert ui.web.packaging_job.snapshot().status == "idle"


def test_with_a_meta_key_the_button_is_back_in_local_mode(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    go_local(monkeypatch)
    ui = packaging_ui(tmp_path)
    agent.save_api_key(ui.config, FAKE_KEY)
    section = card_of(ui.client.get(f"/videos/{V1}").text, "thumbnail")
    assert "Research thumbnail" in section and "need Meta" not in section


def test_meta_mode_keeps_the_button(tmp_path: Path) -> None:
    ui = packaging_ui(tmp_path)
    agent.save_api_key(ui.config, FAKE_KEY)
    section = card_of(ui.client.get(f"/videos/{V1}").text, "thumbnail")
    assert "Research thumbnail" in section and "need Meta" not in section


# --- the Configuration page --------------------------------------------------------------------------
def test_the_page_is_called_configuration_and_keeps_its_urls(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    resp = ui.client.get("/configuration")
    html = resp.text
    assert resp.status_code == 200
    assert "<title>Configuration - Yezatas</title>" in html
    assert '<h1 class="page-title">Configuration</h1>' in html
    assert "<span>Configuration</span>" in html and "<span>Setup</span>" not in html
    for anchor in ("step-ai", "step-translate", "step-transcripts", "step-pull"):
        assert f'id="{anchor}"' in html
    assert "<h2>Primary AI</h2>" in html and "<h2>Third-party alternatives</h2>" in html
    assert "<h2>Transcripts and sounds</h2>" in html
    assert "Other companies that host Meta's open model, Muse Spark Open, for translations." in html


def test_the_nav_item_sits_at_the_bottom_of_the_review_group(tmp_path: Path) -> None:
    html = make_ui(tmp_path).client.get("/").text
    review = html.split('<p class="nav-label">Review</p>')[1].split("</div>")[0]
    assert review.rfind('href="/configuration"') > review.rfind('href="/agent"') > 0
    assert 'aria-label="Configuration"' in html


def test_the_primary_ai_card_in_meta_and_local_mode(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = card_of(make_ui(tmp_path / "a").setup_html(), "step-ai")
    assert 'name="primary_ai" value="meta" checked' in card
    assert "Needs a Meta API key; each feature shows its price." in card
    assert "Free and private: every text feature, translations included, stays on this computer" in card
    assert "ollama pull muse-glimmer:30b" in card and "https://ollama.com/download" in card
    assert "Three features need Meta" not in card  # only when Local is chosen
    go_local(monkeypatch)
    ui = make_ui(tmp_path / "b")
    card = card_of(ui.setup_html(), "step-ai")
    assert 'name="primary_ai" value="local" checked' in card
    assert "Three features need Meta's API. Add a Meta key above to use them." in card
    assert "about $0.18 / hour" in card and "SAM 3.1" in card and "Muse Image" in card
    assert "Self-checked" in card
    agent.save_api_key(ui.config, FAKE_KEY)
    card = card_of(ui.setup_html(), "step-ai")
    assert "Your saved Meta key is used for them, and they show their price." in card
    assert FAKE_KEY not in card


def test_the_status_line_follows_the_checker(tmp_path: Path) -> None:
    down = card_of(make_ui(tmp_path / "a").setup_html(), "step-ai")
    assert 'id="local-ai-help"' in down and "Not running" in down
    up = make_ui(tmp_path / "b", translate_tools=lambda cfg: (
        None, ScriptedVerifier(up=True, model="muse-glimmer:30b")))
    card = card_of(up.setup_html(), "step-ai")
    assert "Ollama running with muse-glimmer:30b" in card and 'id="local-ai-help"' not in card


# --- ffmpeg and the sound tagger ----------------------------------------------------------------------
def test_the_page_says_whether_the_sound_tagger_is_installed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "panns_inference" else real(name, *a, **k))
    card = card_of(make_ui(tmp_path / "a").setup_html(), "step-transcripts")
    assert "Sound tagger (PANNs)" in card and "Not installed" in card
    assert "labels music and sounds in videos without speech" in card
    assert "pip install panns-inference" in card
    assert card.count("<details") >= 3 and "<script" not in card and "style=" not in card
    assert "The sound model (about 300 MB) downloads the first time it is used." in card

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: object() if name == "panns_inference" else real(name, *a, **k))
    card = card_of(make_ui(tmp_path / "b").setup_html(), "step-transcripts")
    assert ">Installed<" in card and "pip install panns-inference" not in card


def test_ffmpeg_help_lists_the_three_systems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcribe, "find_ffmpeg", lambda: None)
    card = card_of(make_ui(tmp_path).setup_html(), "step-transcripts")
    for command in ("winget install Gyan.FFmpeg", "brew install ffmpeg", "sudo apt install ffmpeg"):
        assert command in card
    assert "Run these in a terminal, then restart Yezatas." in card
