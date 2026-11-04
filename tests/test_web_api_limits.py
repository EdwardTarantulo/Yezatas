"""The "YouTube API limits" pop-up on the Bulk edit page: the control, the dialog and the rows."""

from __future__ import annotations

import re
from pathlib import Path

from webhelpers import make_ui
from yezatas.web.views import API_LIMITS

STATIC = Path(__file__).resolve().parents[1] / "src" / "yezatas" / "web" / "static"


def page(tmp_path: Path, path: str = "/bulk") -> str:
    response = make_ui(tmp_path).client.get(path)
    assert response.status_code == 200
    return response.text


def dialog_of(html: str) -> str:
    match = re.search(r'<dialog class="lim-dialog".*?</dialog>', html, re.S)
    assert match
    return match.group(0)


def test_the_button_opens_a_native_dialog_named_for_screen_readers(tmp_path: Path) -> None:
    html = page(tmp_path)
    assert re.search(r'<button type="button"[^>]*data-dialog-open="api-limits"[^>]*aria-haspopup="dialog"[^>]*>'
                     r"YouTube API limits</button>", html)
    dialog = dialog_of(html)
    assert 'id="api-limits"' in dialog and 'aria-labelledby="lim-title"' in dialog and 'id="lim-title"' in dialog
    assert "data-dialog-close" in dialog


def test_the_three_groups_show_every_row_in_a_table_with_headers(tmp_path: Path) -> None:
    dialog = dialog_of(page(tmp_path))
    assert [g.status for g in API_LIMITS.groups] == ["can", "part", "none"]
    assert [g.label for g in API_LIMITS.groups] == ["Can do", "Can do in part", "Not available"]
    assert dialog.count('scope="col"') == 3 and dialog.count('scope="rowgroup"') == 3
    for group in API_LIMITS.groups:
        assert f'id="lim-{group.status}"' in dialog
        for row in group.rows:
            assert f'<th scope="row">{row.name}</th>' in dialog


def test_every_row_says_what_it_means_and_where_to_do_it() -> None:
    for group in API_LIMITS.groups:
        assert group.rows
        for row in group.rows:
            assert row.name and row.meaning.endswith(".") and row.where
            assert row.href == "" or row.href.startswith("/")


def test_the_rows_the_creator_asked_about_stay_honest() -> None:
    """Ads and thumbnails are why this exists: the wording must match Google's docs, not a guess."""
    rows = {row.name: (group.status, row) for group in API_LIMITS.groups for row in group.rows}
    assert rows["Ads on or off for a video"][0] == "none"
    assert "Monetization" in rows["Ads on or off for a video"][1].where
    status, thumb = rows["Custom or automatic thumbnail"]
    assert status == "part" and "does not read this yet" in thumb.meaning
    assert rows["Pinning a comment"][0] == "none" and "More, then Pin" in rows["Pinning a comment"][1].where


def test_the_quota_block_and_its_sources_are_on_the_page(tmp_path: Path) -> None:
    dialog = dialog_of(page(tmp_path))
    assert "10,000 units a day" in dialog and "midnight Pacific Time" in dialog
    for _, url in API_LIMITS.sources:
        assert f'href="{url}"' in dialog and url.startswith("https://")
    assert dialog.count('rel="noopener noreferrer"') == len(API_LIMITS.sources)
    assert "November 4, 2026" in dialog


def test_no_inline_styles_or_scripts_and_no_form_for_the_shared_submit_handler(tmp_path: Path) -> None:
    dialog = dialog_of(page(tmp_path))
    assert "style=" not in dialog and "onclick" not in dialog and "<script" not in dialog
    assert "<form" not in dialog  # app.js rewrites every form on submit; Close is a plain button


def test_the_control_is_on_bulk_edit_only(tmp_path: Path) -> None:
    html = page(tmp_path, "/copilot")
    assert "data-dialog-open" not in html and "lim-dialog" not in html


def test_the_script_opens_it_before_the_early_return_on_pages_that_do_not_poll() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert script.index("data-dialog-open") < script.index('if (document.body.dataset.poll !== "1") return;')
    assert "showModal" in script and "dialog.close()" in script


def test_the_scrim_has_a_colour_in_both_themes() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert css.count("--scrim:") == 2 and "background: var(--scrim)" in css
