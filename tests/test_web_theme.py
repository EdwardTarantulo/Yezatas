import re
from pathlib import Path

from webhelpers import make_ui

STATIC = Path(__file__).parent.parent / "src" / "yezatas" / "web" / "static"
TOKEN = re.compile(r"(--[a-z0-9-]+):")


def _block(css: str, selector: str) -> str:
    start = css.index(selector + " {")
    return css[start:css.index("\n}\n", start)]


def test_theme_script_loads_in_the_head_of_every_page(tmp_path: Path) -> None:
    ui = make_ui(tmp_path)
    for path in ("/configuration", "/videos"):
        html = ui.client.get(path).text
        head = html[:html.index("</head>")]
        assert '<script src="/static/theme.js"></script>' in head, path
        assert '<meta name="color-scheme" content="light dark">' in head, path
    bare = make_ui(tmp_path / "bare", signed_in_browser=False).client
    assert bare.get("/static/theme.js").status_code == 200


def test_toggle_is_a_hidden_button_until_the_script_runs(tmp_path: Path) -> None:
    html = make_ui(tmp_path).client.get("/configuration").text
    button = re.search(r"<button[^>]*data-theme-toggle[^>]*>", html).group(0)
    assert 'type="button"' in button and "hidden" in button
    assert 'aria-pressed="false"' in button and 'aria-label="Dark theme"' in button
    assert "<script>" not in html and "style=" not in html


def test_dark_theme_overrides_every_colour_but_the_google_pictures() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    light = set(TOKEN.findall(_block(css, ":root")))
    dark = set(TOKEN.findall(_block(css, ':root[data-theme="dark"]')))
    assert dark <= light
    assert light - dark == {t for t in light if t.startswith("--g-")}


def _contrast(a: str, b: str) -> float:
    def lum(h: str) -> float:
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# Text colour on background colour, as app.css pairs them.
TEXT_PAIRS = [
    ("text", "bg"), ("text", "surface"), ("text", "hover"), ("muted", "surface"), ("muted", "hover"),
    ("link", "surface"), ("primary-text", "primary-bg"), ("good-text", "good-bg"), ("warn", "warn-bg"),
    ("bad", "bad-bg"), ("info-text", "info-bg"), ("text", "info-bg"), ("text", "warn-bg"), ("text", "code-bg"),
    ("diff-add-text", "diff-add-bg"), ("diff-del-text", "diff-del-bg"), ("text", "diff-add-mark"),
    ("text", "diff-del-mark"), ("muted", "diff-gap-bg"),
    ("link", "brand-soft"), ("warn", "surface"),  # the redesigned active nav item and fix-points chips
    ("g-text", "g-bg"), ("g-muted", "g-bg"),  # the mock YouTube pages of the packaging preview
]


def test_both_themes_meet_text_contrast_aa() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    hexes = re.compile(r"(--[a-z0-9-]+):\s*(#[0-9a-f]{6})")
    light = dict(hexes.findall(_block(css, ":root")))
    dark = {**light, **dict(hexes.findall(_block(css, ':root[data-theme="dark"]')))}
    for name, tokens in (("light", light), ("dark", dark)):
        for fg, bg in TEXT_PAIRS:
            ratio = _contrast(tokens["--" + fg], tokens["--" + bg])
            assert ratio >= 4.5, f"{name}: --{fg} on --{bg} is {ratio:.2f}"


def test_no_colour_outside_the_token_blocks() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    rest = css.replace(_block(css, ":root"), "").replace(_block(css, ':root[data-theme="dark"]'), "")
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", rest)


def test_theme_script_remembers_the_choice_and_follows_the_system() -> None:
    js = (STATIC / "theme.js").read_text(encoding="utf-8")
    assert '"yezatas-theme"' in js and "prefers-color-scheme: dark" in js
    assert "localStorage" in js and "try {" in js
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "data-theme-toggle" in app_js and "yezatasTheme.set" in app_js
