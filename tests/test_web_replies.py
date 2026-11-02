"""The Replies page: the template list with its edit and delete forms, the add form, the
{name} line, adding, editing and deleting with their exact flash sentences, duplicate and
empty refusals, unknown ids, CSRF, escaping and the nav item. Fakes only: no network, no
model, and nothing here ever calls YouTube."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from webhelpers import Ui, make_ui, make_video, seed_catalog
from yezatas import replytemplates
from yezatas.store import Store

NOW = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
V1, V2 = "vid00000001", "vid00000002"


def make_replies_ui(tmp_path: Path) -> Ui:
    ui = make_ui(tmp_path)
    seed_catalog(ui.config, [
        make_video(1, video_id=V1, title="Cactus basics", description="How to repot a cactus.",
                   tags=["cactus"], privacy_status="public"),
        make_video(2, video_id=V2, title="Watering tips", description="Water slowly.",
                   tags=["water"], privacy_status="public")])
    return ui


def page(ui: Ui) -> str:
    resp = ui.client.get("/replies")
    assert resp.status_code == 200
    return resp.text


def nf(ui: Ui, path: str, **kwargs: Any) -> Any:
    kwargs.setdefault("follow_redirects", False)
    return ui.post(path, **kwargs)


def seed_template(ui: Ui, name: str, body: str) -> int:
    with Store(ui.config.catalog_db) as store:
        return replytemplates.save_template(store, name, body, now=NOW)


def stored(ui: Ui) -> list[Any]:
    with Store(ui.config.catalog_db) as store:
        return store.list_reply_templates()


def card(html: str, template_id: int) -> str:
    return html.split(f'id="template-{template_id}"')[1].split("</section>")[0]


# --- the page ------------------------------------------------------------------------------------

def test_the_empty_page_shows_the_empty_state_the_about_line_and_the_add_form(
        tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    html = page(ui)
    assert '<h1 class="page-title">Reply templates</h1>' in html
    about = re.search(r'id="replies-about"[^>]*>([^<]*)<', html)
    assert about and "{name}" in about.group(1)
    assert '<p class="muted" id="replies-empty">No templates yet. Add one below.</p>' in html
    assert 'id="template-' not in html
    add = html.split('id="reply-add"')[1].split("</section>")[0]
    assert "<h2" in add and ">Add a template</h2>" in add
    assert 'method="post" action="/replies/add"' in add and 'name="csrf"' in add
    assert '<label for="reply-name">Name</label>' in add
    assert 'id="reply-name" name="name"' in add
    assert '<label for="reply-body">Reply text</label>' in add
    assert 'id="reply-body" name="body" rows="4"' in add
    assert "{name}" in add  # the hint under the reply text
    assert ">Add template</button>" in add
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and "style=" not in html


def test_the_page_lists_templates_with_their_edit_and_delete_forms(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    thanks = seed_template(ui, "Thanks", "Thanks, {name}! Glad it helped.")
    schedule = seed_template(ui, "Schedule", "Hi {name}, new videos arrive every week.")
    html = page(ui)
    assert 'id="replies-empty"' not in html
    about = re.search(r'id="replies-about"[^>]*>([^<]*)<', html)
    assert about and "{name}" in about.group(1)
    for template_id, name, body in (
            (thanks, "Thanks", "Thanks, {name}! Glad it helped."),
            (schedule, "Schedule", "Hi {name}, new videos arrive every week.")):
        section = card(html, template_id)
        assert f">{name}</h2>" in section or f"<strong>{name}</strong>" in section
        assert f"<p>{body}</p>" in section
        assert f'method="post" action="/replies/{template_id}/edit"' in section
        assert 'name="csrf"' in section
        assert f'name="name" value="{name}"' in section
        assert f'name="body" rows="4">{body}</textarea>' in section
        assert ">Save</button>" in section
        assert f'method="post" action="/replies/{template_id}/delete"' in section
        assert ">Delete</button>" in section
    assert 'id="reply-add"' in html and 'action="/replies/add"' in html


# --- adding --------------------------------------------------------------------------------------

def test_add_saves_the_template_redirects_and_flashes_saved(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    resp = nf(ui, "/replies/add", data={"name": "Thanks", "body": "Thanks, {name}!"})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    saved = stored(ui)
    assert [(t.name, t.body) for t in saved] == [("Thanks", "Thanks, {name}!")]
    html = ui.client.get("/replies").text
    assert "Template saved." in html
    assert f'id="template-{saved[0].id}"' in html and "Thanks, {name}!" in html
    assert 'id="replies-empty"' not in html


def test_add_a_duplicate_name_in_another_case_is_refused(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    seed_template(ui, "Thanks", "Thanks, {name}!")
    resp = nf(ui, "/replies/add", data={"name": "thanks", "body": "Something else."})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "already a template called" in ui.client.get("/replies").text
    assert [(t.name, t.body) for t in stored(ui)] == [("Thanks", "Thanks, {name}!")]


@pytest.mark.parametrize("name", ["", "   "])
def test_add_without_a_name_is_refused(tmp_path: Path, name: str) -> None:
    ui = make_replies_ui(tmp_path)
    resp = nf(ui, "/replies/add", data={"name": name, "body": "Thanks, {name}!"})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "Enter a template name." in ui.client.get("/replies").text
    assert stored(ui) == []


@pytest.mark.parametrize("body", ["", "   "])
def test_add_without_a_reply_text_is_refused(tmp_path: Path, body: str) -> None:
    ui = make_replies_ui(tmp_path)
    resp = nf(ui, "/replies/add", data={"name": "Thanks", "body": body})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "Enter a reply text." in ui.client.get("/replies").text
    assert stored(ui) == []


# --- editing and deleting --------------------------------------------------------------------------

def test_edit_updates_the_name_and_body_and_flashes_saved(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    template_id = seed_template(ui, "Thanks", "Thanks, {name}!")
    resp = nf(ui, f"/replies/{template_id}/edit",
              data={"name": "Thank you", "body": "Thank you, {name}! It means a lot."})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    saved = stored(ui)
    assert [(t.id, t.name, t.body) for t in saved] == [
        (template_id, "Thank you", "Thank you, {name}! It means a lot.")]
    html = ui.client.get("/replies").text
    assert "Template saved." in html
    section = card(html, template_id)
    assert "Thank you, {name}! It means a lot." in section and "Thanks, {name}!" not in section


def test_edit_onto_another_templates_name_is_refused(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    seed_template(ui, "Thanks", "Thanks, {name}!")
    schedule = seed_template(ui, "Schedule", "Hi {name}, new videos arrive every week.")
    resp = nf(ui, f"/replies/{schedule}/edit",
              data={"name": "THANKS", "body": "Hi {name}, new videos arrive every week."})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "already a template called" in ui.client.get("/replies").text
    assert {t.id: (t.name, t.body) for t in stored(ui)}[schedule] == (
        "Schedule", "Hi {name}, new videos arrive every week.")


def test_delete_removes_the_template_and_flashes_deleted(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    template_id = seed_template(ui, "Thanks", "Thanks, {name}!")
    resp = nf(ui, f"/replies/{template_id}/delete")
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert stored(ui) == []
    html = ui.client.get("/replies").text
    assert "Template deleted." in html
    assert f'id="template-{template_id}"' not in html
    assert 'id="replies-empty"' in html


def test_edit_an_unknown_id_flashes_that_it_does_not_exist(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    seed_template(ui, "Thanks", "Thanks, {name}!")
    resp = nf(ui, "/replies/99999/edit", data={"name": "New", "body": "Hi {name}"})
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "This template does not exist." in ui.client.get("/replies").text
    assert [(t.name, t.body) for t in stored(ui)] == [("Thanks", "Thanks, {name}!")]


def test_delete_an_unknown_id_flashes_that_it_does_not_exist(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    template_id = seed_template(ui, "Thanks", "Thanks, {name}!")
    resp = nf(ui, "/replies/99999/delete")
    assert resp.status_code == 303 and resp.headers["location"] == "/replies"
    assert "This template does not exist." in ui.client.get("/replies").text
    assert [t.id for t in stored(ui)] == [template_id]


# --- CSRF ------------------------------------------------------------------------------------------

def test_every_post_needs_the_csrf_token(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    template_id = seed_template(ui, "Thanks", "Thanks, {name}!")
    before = [(t.id, t.name, t.body) for t in stored(ui)]
    attempts = [
        ("/replies/add", {"name": "New", "body": "Hi {name}"}),
        (f"/replies/{template_id}/edit", {"name": "Changed", "body": "Changed, {name}"}),
        (f"/replies/{template_id}/delete", {}),
    ]
    for path, data in attempts:
        assert ui.client.post(path, data=data, follow_redirects=False).status_code == 403
        assert ui.client.post(path, data={**data, "csrf": "wrong"},
                              follow_redirects=False).status_code == 403
    assert [(t.id, t.name, t.body) for t in stored(ui)] == before
    assert "Changed" not in page(ui)


# --- escaping and the shell ------------------------------------------------------------------------

def test_hostile_names_and_bodies_are_escaped(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    with Store(ui.config.catalog_db) as store:
        store.save_reply_template("<script>alert(1)</script>",
                                  "Hi {name} <script>alert(1)</script>", now=NOW)
    html = page(ui)
    assert "<script>alert(1)</script>" not in html and "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and "style=" not in html


def test_the_nav_has_a_replies_item_after_copilot(tmp_path: Path) -> None:
    ui = make_replies_ui(tmp_path)
    html = page(ui)
    items = re.findall(r'<a class="nav-item[^"]*" href="([^"]+)"', html)
    assert items.index("/replies") == items.index("/comments") + 1
    assert re.findall(r'class="nav-item active" href="([^"]+)"', html) == ["/replies"]
