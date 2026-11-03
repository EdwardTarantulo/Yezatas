"""The Copilot page: the chat with its answers, ref links, proposals and footer, the starter buttons, the
ask job, the proposal buttons and the new chat. The copilot function is a fake that writes turns with the
store methods (like the real one); no model, no network, and nothing here ever calls YouTube."""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from markupsafe import Markup

from webhelpers import Ui, make_push_ui, make_ui, make_video, seed_catalog
from yezatas import agent, copilot
from yezatas.agent import ModelError, estimate_cost
from yezatas.store import Store
from yezatas.web import app as web_app
from yezatas.web.app import NO_PACKAGING_KEY
from yezatas.web.views import STARTER_QUESTIONS, render_answer

NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
MODEL = "muse-spark-1.3"
V1, V2, V3 = "vid000001", "vid000002", "vid000003"


@pytest.fixture(autouse=True)
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(agent.KEY_ENV, raising=False)


def videos() -> list[Any]:
    return [
        make_video(1, video_id=V1, title="Cactus basics", description="Line one.\nLine two.",
                   tags=["cactus", "garden"], privacy_status="public"),
        make_video(2, video_id=V2, title="Succulent <care> & more", privacy_status="public"),
        make_video(3, video_id=V3, title="Watering tips", description="Water slowly.",
                   tags=["water"], privacy_status="public"),
    ]


class FakeCopilot:
    """Writes one answered turn (and its proposals) the way `copilot.ask` does."""

    def __init__(self, answer: str = "Fix V1 first.", refs: dict[str, str] | None = None,
                 proposals: tuple[tuple[str, str, str, str], ...] = (), error: Exception | None = None,
                 gate: threading.Event | None = None) -> None:
        self.answer = answer
        self.refs = refs if refs is not None else {"V1": V1}
        self.proposals = proposals  # (video id, title, description, reason)
        self.error = error
        self.gate = gate
        self.entered = threading.Event()
        self.calls: list[tuple[str, int, datetime]] = []

    def __call__(self, store: Store, question: str, chat_id: int, as_of: datetime,
                 report: Any) -> int:
        self.calls.append((question, chat_id, as_of))
        report("Reading the catalog")
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10), "the test never opened the gate"
        if self.error is not None:
            raise self.error
        turn = store.begin_copilot_turn(chat_id, question, MODEL, NOW)
        store.finish_copilot_turn(
            turn, "answered", self.answer, self.refs, [], [], (1000, 200, 300), NOW)
        for video_id, title, description, reason in self.proposals:
            store.add_copilot_proposal(turn, video_id, title, description, ["cactus"], reason, NOW)
        return turn


def make_cp_ui(tmp_path: Path, fake: FakeCopilot | None = None, key: bool = True,
               seeded: bool = True, **kwargs: Any) -> tuple[Ui, FakeCopilot]:
    fake = fake or FakeCopilot()
    ui = make_ui(tmp_path, copilot=fake, packaging_key=lambda cfg: key, **kwargs)
    if seeded:
        seed_catalog(ui.config, videos())
    return ui, fake


def page(ui: Ui, path: str = "/copilot") -> str:
    resp = ui.client.get(path)
    assert resp.status_code == 200
    return resp.text


def ask(ui: Ui, question: str) -> Any:
    resp = ui.post("/copilot/ask", data={"question": question}, follow_redirects=False)
    ui.web.copilot_job.wait()
    return resp


def store_turn(ui: Ui, chat_id: int = 1, question: str = "Q?", answer: str = "A.",
               refs: dict[str, str] | None = None, status: str = "answered",
               notes: list[str] | None = None, log: list[Any] | None = None) -> int:
    with Store(ui.config.catalog_db) as store:
        turn = store.begin_copilot_turn(chat_id, question, MODEL, NOW)
        store.finish_copilot_turn(turn, status, answer, refs or {}, notes or [], log or [],
                                  (1000, 200, 300), NOW)
    return turn


def store_proposal(ui: Ui, turn: int, video_id: str = V1, title: str = "Cactus basics for beginners",
                   description: str = "Line one.\nLine two, better.", reason: str = "Clearer.") -> int:
    with Store(ui.config.catalog_db) as store:
        return store.add_copilot_proposal(turn, video_id, title, description, ["cactus"], reason, NOW)


def proposal_status(ui: Ui, proposal_id: int) -> str:
    with Store(ui.config.catalog_db) as store:
        found = store.get_copilot_proposal(proposal_id)
    assert found is not None
    return found.status


# --- the answer renderer ---------------------------------------------------------------------------
TITLES = {V1: "Cactus basics", V2: "Succulent <care> & more"}


def test_render_answer_escapes_everything_first() -> None:
    out = render_answer("<script>alert(1)</script> \"quoted\" 'single' & more", {}, {})
    assert isinstance(out, Markup)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "&#34;quoted&#34;" in out and "&amp; more" in out


def test_render_answer_bold_cannot_carry_html() -> None:
    out = render_answer("**<img src=x onerror=alert(1)>** and **fine**", {}, {})
    assert "<img" not in out
    assert out == "<p><strong>&lt;img src=x onerror=alert(1)&gt;</strong> and <strong>fine</strong></p>"


def test_render_answer_paragraphs_bullets_and_numbered_lists() -> None:
    text = "First para\nsame para.\n\nSecond para.\n- one\n* two\n\n1. step a\n2. step b\n  more of b"
    assert render_answer(text, {}, {}) == (
        "<p>First para<br>same para.</p><p>Second para.</p>"
        "<ul><li>one</li><li>two</li></ul>"
        "<ol><li>step a</li><li>step b<br>more of b</li></ol>")


def test_render_answer_a_year_or_a_dash_is_not_a_list() -> None:
    assert render_answer("2023. A good year\n-5 degrees", {}, {}) == (
        "<p>2023. A good year<br>-5 degrees</p>")


def test_render_answer_known_refs_link_to_the_video_with_its_escaped_title() -> None:
    out = render_answer("Fix **V2** and V1.", {"V1": V1, "V2": V2}, TITLES)
    assert f'<a href="/videos/{V1}">Cactus basics</a> <small class="cp-ref">V1</small>' in out
    assert (f'<strong><a href="/videos/{V2}">Succulent &lt;care&gt; &amp; more</a> '
            '<small class="cp-ref">V2</small></strong>') in out
    assert "<care>" not in out


def test_render_answer_unknown_refs_and_gone_videos_stay_plain_text() -> None:
    out = render_answer("V1, V7 and V9.", {"V1": V1, "V9": "vid000099"}, TITLES)
    assert "V7" in out and "V9" in out
    assert out.count("<a ") == 1  # only V1 is known and still in the catalog
    assert ">V9<" not in out and ">V7<" not in out


def test_render_answer_a_hostile_video_id_cannot_break_out_of_the_link() -> None:
    bad = 'x"><script>alert(1)</script>'
    out = render_answer("V1", {"V1": bad}, {bad: "Title"})
    assert "<script>" not in out and 'href="/videos/x%22%3E%3Cscript%3E' in out


# --- the page ------------------------------------------------------------------------------------------
def test_an_empty_chat_shows_the_intro_the_starters_and_the_safety_line(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    html = page(ui)
    assert "Start with a question" in html
    for question in STARTER_QUESTIONS:
        assert f'name="question" value="{question}"' in html
    assert len(STARTER_QUESTIONS) == 3
    assert html.count('action="/copilot/ask"') == 4  # three starters and the question box
    assert 'id="cp-question"' in html and f'maxlength="{copilot.MAX_QUESTION_CHARS}"' in html
    assert "Answers use only your catalog." in html
    assert "Numbers come from Yezatas, not the model" in html
    assert "nothing reaches YouTube until you approve and push it on the Changes page" in html
    assert 'id="copilot-new"' not in html  # nothing to start over yet
    assert "Meta's terms apply (<a href=\"/setup#ai-data\">what Meta may do with it</a>)." in html and "Nothing is sent to YouTube." in html


def test_the_nav_has_a_copilot_item_after_bulk_edit(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/copilot") == items.index("/bulk") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/copilot"]


def test_a_filled_chat_shows_question_answer_refs_footer_and_the_tool_log(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    log = [{"type": "tool", "name": "find_videos", "arguments": {"has_chapters": False},
            "result_chars": 321}, {"type": "problems", "round": 1, "problems": ["Unsupported 37"]},
           {"type": "guard", "round": 1, "numbers": ["7,391"]},
           {"type": "guard", "round": 2, "numbers": ["12"]}]
    turn = store_turn(ui, question="What to fix?", answer="Start with V1.\n\n- V2 is next",
                      refs={"V1": V1, "V2": V2}, notes=["A sentence was removed."], log=log)
    html = page(ui)
    assert "Start with a question" not in html and 'id="copilot-new"' in html
    card = html.split(f'id="turn-{turn}"')[1].split("</section>")[0]
    assert "What to fix?" in card
    assert f'<a href="/videos/{V1}">Cactus basics</a>' in card
    assert "Succulent &lt;care&gt; &amp; more" in card
    assert "<li>" in card
    assert "A sentence was removed." in card
    cost = estimate_cost(MODEL, 1000, 200, 300)
    assert cost is not None
    assert f"{MODEL} &middot; 1,000 tokens in (200 cached), 300 out &middot; about ${cost:.3f}" in card
    assert "1 tool call" in card
    assert "<details" in card and "<code>find_videos</code>" in card and "321 characters" in card
    assert '{&#34;has_chapters&#34;:false}' in card
    assert "Unsupported 37" in card
    assert "Number check: 7,391 came from no tool; the model was asked to fix it." in card
    assert "Number check: 12 came from no tool; the sentence was removed." in card
    assert "Ask a follow-up" in html


def test_hostile_text_in_questions_titles_and_reasons_is_escaped(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui, question="<script>alert('q')</script>", answer="<b>hi</b> V2",
                      refs={"V2": V2}, notes=["<i>note</i>"])
    store_proposal(ui, turn, V2, title="New <b>title</b>", reason="<u>because</u>")
    html = page(ui)
    for raw in ("<script>alert", "<b>hi</b>", "<i>note</i>", "<b>title</b>", "<u>because</u>"):
        assert raw not in html
    assert "&lt;script&gt;alert" in html and "&lt;u&gt;because&lt;/u&gt;" in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and " style=" not in html


def test_a_failed_and_an_interrupted_turn_say_so(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    store_turn(ui, question="One", status="failed", answer="Muse Spark is busy. Try again.")
    with Store(ui.config.catalog_db) as store:
        store.begin_copilot_turn(1, "Two", MODEL, NOW)  # never finished
    html = page(ui)
    assert "Muse Spark is busy. Try again." in html
    assert "This question did not finish. Ask it again." in html


def test_the_page_never_calls_the_copilot(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    page(ui)
    assert fake.calls == []


def test_no_pulled_catalog_goes_back_to_setup(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path, seeded=False)
    resp = ui.client.get("/copilot", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"
    resp = ui.post("/copilot/ask", data={"question": "Hi"}, follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/setup"
    assert fake.calls == []


# --- asking -------------------------------------------------------------------------------------------
def test_asking_runs_the_job_and_the_answer_shows(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    resp = ask(ui, "  Which videos need work?  ")
    assert resp.status_code == 303 and resp.headers["location"] == "/copilot"
    assert [c[0] for c in fake.calls] == ["Which videos need work?"]
    assert fake.calls[0][1] == 1  # the first chat
    assert ui.client.get("/api/status").json()["copilot"]["status"] == "done"
    html = page(ui)
    assert "Which videos need work?" in html and "Fix " in html
    assert f'<a href="/videos/{V1}">Cactus basics</a>' in html
    ask(ui, "And then?")
    assert fake.calls[1][1] == 1  # the same chat continues
    assert html.count('class="card cp-turn"') == 1 and page(ui).count('class="card cp-turn"') == 2


def test_the_views_moment_is_the_pull_time_not_now(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    ask(ui, "When?")
    with Store(ui.config.catalog_db) as store:
        pull = store.latest_pull()
    assert pull is not None
    assert fake.calls[0][2] == web_app.pull_moment(pull)


def test_a_starter_button_asks_its_question(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    ask_resp = ui.post("/copilot/ask", data={"question": STARTER_QUESTIONS[0]},
                       follow_redirects=False)
    ui.web.copilot_job.wait()
    assert ask_resp.status_code == 303
    assert [c[0] for c in fake.calls] == [STARTER_QUESTIONS[0]]


@pytest.mark.parametrize("path", ["/copilot/ask", "/copilot/new", "/copilot/proposals/1/draft",
                                  "/copilot/proposals/1/dismiss"])
def test_every_post_needs_the_csrf_token(tmp_path: Path, path: str) -> None:
    ui, fake = make_cp_ui(tmp_path)
    turn = store_turn(ui)
    store_proposal(ui, turn)
    assert ui.client.post(path, data={"question": "Hi"}, follow_redirects=False).status_code == 403
    assert ui.client.post(path, data={"csrf": "wrong", "question": "Hi"},
                          follow_redirects=False).status_code == 403
    assert fake.calls == [] and proposal_status(ui, 1) == "open"
    with Store(ui.config.catalog_db) as store:
        assert store.get_draft(V1) is None


@pytest.mark.parametrize("text", ["", "   \n  "])
def test_an_empty_question_is_refused_in_plain_words(tmp_path: Path, text: str) -> None:
    ui, fake = make_cp_ui(tmp_path)
    ask(ui, text)
    assert fake.calls == []
    assert "Type a question first. Nothing was sent." in page(ui)


def test_a_too_long_question_is_refused_but_the_limit_itself_is_fine(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    ask(ui, "x" * (copilot.MAX_QUESTION_CHARS + 1))
    assert fake.calls == []
    assert (f"The question has {copilot.MAX_QUESTION_CHARS + 1:,} characters; at most "
            f"{copilot.MAX_QUESTION_CHARS:,} are allowed. Nothing was sent.") in page(ui)
    ask(ui, "y" * copilot.MAX_QUESTION_CHARS)
    assert len(fake.calls) == 1


def test_without_a_key_the_buttons_are_disabled_and_ask_sends_nothing(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path, key=False)
    html = page(ui)
    assert 'id="copilot-no-key"' in html and 'href="/setup#step-ai"' in html
    assert re.search(r'id="copilot-ask" disabled', html)
    assert html.count("cp-starter") - 1 == len(STARTER_QUESTIONS)
    assert len(re.findall(r'class="button secondary cp-starter" disabled', html)) == 3
    ask(ui, "Hello?")
    assert fake.calls == []
    assert NO_PACKAGING_KEY in page(ui)


def test_a_second_question_while_one_runs_is_refused_and_the_page_polls(tmp_path: Path) -> None:
    gate = threading.Event()
    ui, fake = make_cp_ui(tmp_path, FakeCopilot(gate=gate))
    ui.post("/copilot/ask", data={"question": "Slow one"}, follow_redirects=False)
    assert fake.entered.wait(10)
    try:
        html = page(ui)
        assert 'id="copilot-msg"' in html and 'data-poll="1"' in html
        assert "Reading the catalog" in html
        assert 'id="copilot-form"' not in html  # no second question box while it runs
        assert ui.client.get("/api/status").json()["copilot"]["status"] == "running"
        ui.post("/copilot/ask", data={"question": "Another"}, follow_redirects=False)
        assert "An answer is already being written." in page(ui)
        assert len(fake.calls) == 1
        ui.post("/copilot/new", follow_redirects=False)
        assert "Wait for it to finish." in page(ui)
    finally:
        gate.set()
        ui.web.copilot_job.wait()
    assert ui.client.get("/api/status").json()["copilot"]["status"] == "done"
    assert 'data-poll="0"' in page(ui)


@pytest.mark.parametrize("error, shown", [
    (copilot.CopilotError("There is no pulled catalog yet. Pull it in Configuration first."),
     "There is no pulled catalog yet. Pull it in Configuration first."),
    (ModelError("auth", "The key was refused."), "The key was refused."),
    (RuntimeError("secret internals"), "The answer stopped unexpectedly (RuntimeError)."),
])
def test_a_failed_job_shows_its_plain_message_once(tmp_path: Path, error: Exception,
                                                   shown: str) -> None:
    ui, _ = make_cp_ui(tmp_path, FakeCopilot(error=error))
    ask(ui, "Will fail")
    assert ui.web.copilot_job.snapshot().status == "failed"
    html = page(ui)
    assert 'id="copilot-failed"' in html and shown in html
    assert "secret internals" not in html
    assert 'id="copilot-failed"' not in page(ui)


def test_the_page_script_polls_the_copilot_job() -> None:
    js = (Path(web_app.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
    assert '"copilot"' in js


def test_the_default_copilot_without_a_key_fails_the_job_with_the_no_key_message(
        tmp_path: Path) -> None:
    ui = make_ui(tmp_path, packaging_key=lambda cfg: True)
    seed_catalog(ui.config, videos())
    ask(ui, "No key anywhere")
    snap = ui.web.copilot_job.snapshot()
    assert snap.status == "failed" and snap.error_kind == "auth" and snap.message == NO_PACKAGING_KEY


# --- proposals ----------------------------------------------------------------------------------------
def test_a_proposal_card_shows_before_after_excerpt_and_reason(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui, refs={"V1": V1})
    pid = store_proposal(ui, turn)
    html = page(ui)
    card = html.split(f'id="proposal-{pid}"')[1].split("</section>")[0]
    assert ('<span class="cp-old">Cactus basics</span> &rarr; '
            '<span class="cp-new">Cactus basics for beginners</span>') in card
    assert "Description:" in card and "Line two." in card and "Line two, better." in card
    assert "Tags: 1 removed" in card
    assert "Why:</strong> Clearer." in card
    assert f'action="/copilot/proposals/{pid}/draft"' in card and "Make draft" in card
    assert f'action="/copilot/proposals/{pid}/dismiss"' in card and "Dismiss" in card
    assert '<small class="cp-ref">V1</small>' in card


def test_make_draft_saves_a_copilot_draft_and_the_card_says_drafted(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui, refs={"V1": V1})
    pid = store_proposal(ui, turn)
    resp = ui.post(f"/copilot/proposals/{pid}/draft", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/copilot#turn-{turn}"
    with Store(ui.config.catalog_db) as store:
        draft = store.get_draft(V1)
    assert draft is not None and draft.source == "copilot"
    assert draft.title == "Cactus basics for beginners" and draft.note == "Copilot: Clearer."
    assert draft.approved_at is None
    assert proposal_status(ui, pid) == "drafted"
    html = page(ui)
    assert 'Draft saved for &#34;Cactus basics&#34;. Review it on the Changes page; nothing changes on ' in html
    assert "approve and push." in html
    card = html.split(f'id="proposal-{pid}"')[1].split("</section>")[0]
    assert "Drafted" in card and 'href="/changes"' in card
    assert "Make draft" not in card and "Dismiss" not in card


def test_a_second_make_draft_is_refused_with_the_copilots_sentence(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui)
    first = store_proposal(ui, turn)
    second = store_proposal(ui, turn, title="Another title")
    ui.post(f"/copilot/proposals/{first}/draft", follow_redirects=False)
    ui.post(f"/copilot/proposals/{first}/draft", follow_redirects=False)
    assert "This proposal was already used or dismissed." in page(ui)
    ui.post(f"/copilot/proposals/{second}/draft", follow_redirects=False)
    assert "This video already has a draft. Push or discard it first." in page(ui)
    assert proposal_status(ui, second) == "open"
    html = page(ui)
    card = html.split(f'id="proposal-{second}"')[1].split("</section>")[0]
    assert "already has a draft" in card and "Make draft" not in card and "Dismiss" in card
    with Store(ui.config.catalog_db) as store:
        draft = store.get_draft(V1)
    assert draft is not None and draft.title == "Cactus basics for beginners"


def test_dismiss_hides_the_buttons_and_keeps_no_draft(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui)
    pid = store_proposal(ui, turn)
    resp = ui.post(f"/copilot/proposals/{pid}/dismiss", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == f"/copilot#turn-{turn}"
    assert proposal_status(ui, pid) == "dismissed"
    card = page(ui).split(f'id="proposal-{pid}"')[1].split("</section>")[0]
    assert "Dismissed" in card and "Make draft" not in card
    with Store(ui.config.catalog_db) as store:
        assert store.get_draft(V1) is None
    ui.post(f"/copilot/proposals/{pid}/draft", follow_redirects=False)  # a dismissed one can't be drafted
    assert "This proposal was already used or dismissed." in page(ui)


def test_a_proposal_for_a_video_that_is_gone_can_only_be_dismissed(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    turn = store_turn(ui)
    pid = store_proposal(ui, turn, video_id="vid999999")
    card = page(ui).split(f'id="proposal-{pid}"')[1].split("</section>")[0]
    assert "no longer in your catalog" in card and "Make draft" not in card and "Dismiss" in card


@pytest.mark.parametrize("action", ["draft", "dismiss"])
@pytest.mark.parametrize("raw", ["999", "abc", "-1", "1" * 40])
def test_an_unknown_proposal_id_is_a_plain_error(tmp_path: Path, action: str, raw: str) -> None:
    ui, _ = make_cp_ui(tmp_path)
    resp = ui.post(f"/copilot/proposals/{raw}/{action}", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/copilot"
    assert "This proposal does not exist." in page(ui)


# --- new chat -------------------------------------------------------------------------------------------
def test_new_chat_starts_empty_and_the_next_question_goes_to_it(tmp_path: Path) -> None:
    ui, fake = make_cp_ui(tmp_path)
    store_turn(ui, chat_id=1, question="Old question")
    assert "Old question" in page(ui)
    resp = ui.post("/copilot/new", follow_redirects=False)
    assert resp.status_code == 303 and resp.headers["location"] == "/copilot"
    html = page(ui)
    assert "Old question" not in html and "Start with a question" in html
    ask(ui, "Fresh start")
    assert fake.calls[0][1] == 2
    html = page(ui)
    assert "Fresh start" in html and "Old question" not in html


def test_a_restart_opens_the_newest_chat(tmp_path: Path) -> None:
    ui, _ = make_cp_ui(tmp_path)
    store_turn(ui, chat_id=1, question="First chat")
    store_turn(ui, chat_id=2, question="Second chat")
    html = page(ui)
    assert "Second chat" in html and "First chat" not in html


# --- the live channel -----------------------------------------------------------------------------------
def test_nothing_on_this_page_touches_the_youtube_writer(tmp_path: Path) -> None:
    fake = FakeCopilot(proposals=(("vid000001", "A better title", "New text.", "Why not"),))
    ui, writer = make_push_ui(tmp_path, copilot=fake, packaging_key=lambda cfg: True)
    ask(ui, "Anything to improve?")
    with Store(ui.config.catalog_db) as store:
        turn = store.latest_copilot_chat_id()
        assert turn is not None
        proposals = store.list_copilot_proposals([t.id for t in store.list_copilot_turns(turn)])
    assert len(proposals) == 1
    page(ui)
    ui.post("/copilot/new", follow_redirects=False)
    ui.post(f"/copilot/proposals/{proposals[0].id}/draft", follow_redirects=False)
    ui.post(f"/copilot/proposals/{proposals[0].id}/dismiss", follow_redirects=False)
    with Store(ui.config.catalog_db) as store:
        draft = store.get_draft("vid000001")
    assert draft is not None and draft.source == "copilot" and draft.approved_at is None
    assert writer.calls == [] and writer.updates == []
