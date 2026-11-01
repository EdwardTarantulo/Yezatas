"""Link inventory and dead-link scanner: URL reading, the inventory, the status mapping and the scan.
Videos and links are made up (`.invalid` domains); no test touches the network."""

import socket
import ssl
import threading
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from yezatas.links import (MAX_LINKS, SKIPPED_MESSAGE, LinkEntry, LinkOutcome, check_links,
                           inventory, link_domain, link_kind, outcome_for_error,
                           outcome_for_status, same_place, youtube_video_id)
from yezatas.store import Store, Video

NOW = datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)
TARGET = "vid00000099"


def video(n: int, views: int = 100, description: str = "", privacy: str = "public",
          **kw: Any) -> Video:
    return Video(video_id=f"vid{n:08d}", title=f"Cactus {chr(65 + n % 26)}", description=description,
                 privacy_status=privacy, view_count=views, **kw)


@pytest.fixture()
def store(tmp_path: Path):
    with Store(tmp_path / "t.sqlite") as s:
        yield s


# --- reading a link -------------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={TARGET}",
    f"https://youtube.com/watch?v={TARGET}",
    f"https://m.youtube.com/watch?v={TARGET}",
    f"http://WWW.YouTube.com/watch?v={TARGET}&t=42s&list=PL1",
    f"https://www.youtube.com/watch?feature=share&v={TARGET}",
    f"https://music.youtube.com/watch?v={TARGET}",
    f"https://youtu.be/{TARGET}",
    f"https://youtu.be/{TARGET}?t=30",
    f"youtu.be/{TARGET}",
    f"www.youtube.com/watch?v={TARGET}",
    f"https://www.youtube.com/shorts/{TARGET}",
    f"https://youtube.com/shorts/{TARGET}/",
    f"https://m.youtube.com/shorts/{TARGET}?feature=share",
])
def test_video_links_in_every_form_give_the_id(url: str) -> None:
    assert youtube_video_id(url) == TARGET
    assert link_kind(url) == "video"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=short",  # not 11 characters
    "https://www.youtube.com/watch?v=vid000000999",  # 12 characters
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?list=PL1",
    "https://www.youtube.com/watchx?v=vid00000099",
    "https://youtu.be/",
    "https://youtu.be/bad!",
    "https://www.youtube.com/shorts/",
    "https://www.youtube.com/@someone",
    "https://www.youtube.com/playlist?list=PL1",
    "https://example.invalid/watch?v=vid00000099",
    "https://notyoutube.com/shorts/vid00000099",
    "https://youtube.com.example.invalid/watch?v=vid00000099",
])
def test_other_links_have_no_video_id(url: str) -> None:
    assert youtube_video_id(url) is None
    assert link_kind(url) != "video"


def test_kinds_and_domains() -> None:
    assert link_kind("https://www.youtube.com/@someone") == "youtube"
    assert link_kind("https://youtu.be/") == "youtube"
    assert link_kind("https://music.youtube.com/playlist?list=PL1") == "youtube"
    assert link_kind("https://shop.example.invalid/a") == "external"
    assert link_kind("https://youtube.com.example.invalid/") == "external"
    assert link_domain("https://WWW.Shop.Example.invalid/a?b=1") == "shop.example.invalid"
    assert link_domain("www.example.invalid/x") == "example.invalid"
    assert link_domain("https://m.youtube.com/watch?v=x") == "m.youtube.com"
    assert link_domain("http://[broken") == ""
    assert link_kind("http://[broken") == "external"


# --- the inventory --------------------------------------------------------------------------------
def test_inventory_counts_reach_order_and_domains() -> None:
    a = video(1, 500, "Shop: https://shop.example.invalid/a. Also www.docs.example.invalid/x, thanks")
    b = video(2, 900, "https://shop.example.invalid/a\nhttps://shop.example.invalid/a again", "unlisted")
    c = video(3, 40, "Only https://other.example.invalid/c here", "private")
    d = video(4, 5000, "no links at all")
    entries = inventory([a, b, c, d], {})
    assert [e.url for e in entries] == [
        "https://shop.example.invalid/a", "www.docs.example.invalid/x",
        "https://other.example.invalid/c"]
    shop = entries[0]
    assert shop.video_ids == (b.video_id, a.video_id)  # most viewed first
    assert shop.reach == 1400
    assert shop.public_count == 1  # b is unlisted
    assert shop.domain == "shop.example.invalid" and shop.kind == "external"
    assert shop.check is None and shop.finding == ""
    assert entries[1].domain == "docs.example.invalid" and entries[1].reach == 500
    assert entries[2].public_count == 0 and entries[2].reach == 40


def test_inventory_ties_break_by_url_and_each_description_counts_once() -> None:
    a = video(1, 100, "https://b.example.invalid/ https://a.example.invalid/ https://a.example.invalid/")
    entries = inventory([a], {})
    assert [e.url for e in entries] == ["https://a.example.invalid/", "https://b.example.invalid/"]
    assert [e.reach for e in entries] == [100, 100]
    assert entries[0].video_ids == (a.video_id,)


def test_inventory_same_url_text_is_the_key() -> None:
    a = video(1, 10, "https://a.example.invalid/x")
    b = video(2, 20, "https://a.example.invalid/x/")
    assert len(inventory([a, b], {})) == 2


def test_inventory_attaches_the_latest_check() -> None:
    a = video(1, 10, "https://a.example.invalid/x https://b.example.invalid/y")
    check = _check("https://a.example.invalid/x", "broken")
    entries = {e.url: e for e in inventory([a], {check.url: check})}
    assert entries["https://a.example.invalid/x"].check is check
    assert entries["https://b.example.invalid/y"].check is None


def _check(url: str, status: str) -> Any:
    from yezatas.store import LinkCheck
    return LinkCheck(id=1, url=url, checked_at="2026-06-01T00:00:00Z", status=status,
                     http_status=None, final_url="", message="")


def test_inventory_own_video_states_and_findings() -> None:
    pub = video(10, 1, "")
    unl = video(11, 1, "", "unlisted")
    prv = video(12, 1, "", "private")
    other = video(13, 1, "", "privateScheduled")
    text = " ".join([
        f"https://youtu.be/{pub.video_id}",
        f"https://www.youtube.com/watch?v={unl.video_id}&t=5",
        f"https://www.youtube.com/shorts/{prv.video_id}",
        f"https://youtu.be/{other.video_id}",
        "https://youtu.be/vid99999999",
        "https://www.youtube.com/@someone",
        "https://shop.example.invalid/a",
    ])
    linker = video(1, 100, text)
    entries = {e.url: e for e in inventory([linker, pub, unl, prv, other], {})}
    states = {e.url: (e.kind, e.target_id, e.target_state) for e in entries.values()}
    assert states[f"https://youtu.be/{pub.video_id}"] == ("video", pub.video_id, "public")
    assert states[f"https://www.youtube.com/watch?v={unl.video_id}&t=5"][2] == "unlisted"
    assert states[f"https://www.youtube.com/shorts/{prv.video_id}"][2] == "private"
    assert states[f"https://youtu.be/{other.video_id}"][2] == "private"
    assert states["https://youtu.be/vid99999999"] == ("video", "vid99999999", "not_in_catalog")
    assert states["https://www.youtube.com/@someone"] == ("youtube", "", "")
    assert states["https://shop.example.invalid/a"] == ("external", "", "")

    private = entries[f"https://www.youtube.com/shorts/{prv.video_id}"]
    assert private.target_title == prv.title
    assert private.finding == f'Links to your private video "{prv.title}".'
    assert entries["https://youtu.be/vid99999999"].finding == (
        "Not one of your videos, or deleted: Yezatas can't tell.")
    for url in (f"https://youtu.be/{pub.video_id}", "https://www.youtube.com/@someone",
                f"https://www.youtube.com/watch?v={unl.video_id}&t=5"):
        assert entries[url].finding == ""


def test_inventory_of_nothing() -> None:
    assert inventory([], {}) == []
    assert inventory([video(1, 5, "")], {}) == []


# --- the status mapping ---------------------------------------------------------------------------
def test_status_2xx_is_ok() -> None:
    assert outcome_for_status("https://a.example.invalid/x", 200, "https://a.example.invalid/x") == (
        LinkOutcome("ok", 200, "", ""))
    assert outcome_for_status("https://a.example.invalid/x", 204).status == "ok"


@pytest.mark.parametrize("final", [
    "http://a.example.invalid/x",  # scheme only
    "https://a.example.invalid/x/",  # trailing slash
    "https://A.Example.invalid/x#top",  # host case and fragment
    "https://a.example.invalid:443/x",  # default port
])
def test_scheme_slash_and_case_differences_are_ok(final: str) -> None:
    assert same_place("https://a.example.invalid/x", final)
    assert outcome_for_status("https://a.example.invalid/x", 200, final).status == "ok"


def test_a_slash_only_difference_the_other_way_and_on_a_bare_host() -> None:
    assert outcome_for_status("https://a.example.invalid/x/", 200, "https://a.example.invalid/x").status == "ok"
    assert outcome_for_status("https://a.example.invalid", 200, "https://a.example.invalid/").status == "ok"


@pytest.mark.parametrize("final", [
    "https://b.example.invalid/x",  # other host
    "https://a.example.invalid/y",  # other path
    "https://a.example.invalid/x?utm=1",  # other query
    "https://www.a.example.invalid/x",  # www is a move
    "https://a.example.invalid:8080/x",
])
def test_real_moves_are_redirected(final: str) -> None:
    result = outcome_for_status("https://a.example.invalid/x", 200, final)
    assert result == LinkOutcome("redirected", 200, final, f"Moves to {final}")


def test_not_found_is_broken() -> None:
    assert outcome_for_status("u", 404) == LinkOutcome("broken", 404, "", "Not found (404).")
    assert outcome_for_status("u", 410) == LinkOutcome("broken", 410, "", "Not found (410).")


@pytest.mark.parametrize("code", [500, 502, 503, 599])
def test_server_errors_say_the_problem_is_on_the_site(code: int) -> None:
    result = outcome_for_status("u", code)
    assert result.status == "error" and result.http_status == code
    assert result.message == (f"The site answered {code} (a problem on the site). "
                              "Try again later or open it yourself.")


@pytest.mark.parametrize("code", [301, 304, 400, 401, 403, 429])
def test_other_statuses_are_errors_with_a_plain_sentence(code: int) -> None:
    result = outcome_for_status("u", code)
    assert result.status == "error" and result.http_status == code
    assert result.message == f"The site answered {code}; it may block checks. Open it yourself."


def test_exceptions_map_to_broken_or_error() -> None:
    unreachable = "The site could not be reached."
    assert outcome_for_error(socket.gaierror(-2, "Name or service not known")) == (
        LinkOutcome("broken", None, "", unreachable))
    assert outcome_for_error(ConnectionRefusedError()).message == unreachable
    assert outcome_for_error(urllib.error.URLError(socket.gaierror(-2, "x"))).status == "broken"
    assert outcome_for_error(urllib.error.URLError(ConnectionRefusedError())).status == "broken"

    timeout = outcome_for_error(urllib.error.URLError(socket.timeout("timed out")))
    assert timeout.status == "error" and "did not answer in 10 seconds" in timeout.message
    assert outcome_for_error(TimeoutError()).status == "error"

    cert = outcome_for_error(urllib.error.URLError(ssl.SSLCertVerificationError("bad")))
    assert cert.status == "error" and "security certificate" in cert.message
    assert outcome_for_error(ssl.SSLError("handshake")).status == "error"

    http = urllib.error.HTTPError("https://a.example.invalid/x", 410, "Gone", {}, None)  # type: ignore[arg-type]
    assert outcome_for_error(http) == LinkOutcome("broken", 410, "", "Not found (410).")

    for odd in (urllib.error.URLError("unknown url type"), ValueError("bad"), ConnectionResetError()):
        result = outcome_for_error(odd)
        assert result.status == "error" and result.message.endswith("Open it yourself.")


# --- the scan -------------------------------------------------------------------------------------
def fake_fetch(table: dict[str, LinkOutcome | BaseException], seen: list[str] | None = None):
    lock = threading.Lock()

    def fetch(url: str) -> LinkOutcome:
        with lock:
            if seen is not None:
                seen.append(url)
        answer = table.get(url, LinkOutcome("ok", 200))
        if isinstance(answer, BaseException):
            raise answer
        return answer
    return fetch


def test_every_status_is_saved_with_its_fields(store: Store) -> None:
    table: dict[str, LinkOutcome | BaseException] = {
        "https://ok.example.invalid/": LinkOutcome("ok", 200),
        "https://moved.example.invalid/": LinkOutcome("redirected", 200, "https://new.example.invalid/",
                                                      "Moves to https://new.example.invalid/"),
        "https://gone.example.invalid/": outcome_for_status("u", 404),
        "https://down.example.invalid/": LinkOutcome("broken", None, "", "The site could not be reached."),
        "https://blocks.example.invalid/": outcome_for_status("u", 403),
    }
    summary = check_links(store, list(table), fetch=fake_fetch(table), now=NOW)
    assert summary.counts == {"ok": 1, "redirected": 1, "broken": 2, "error": 1}
    assert (summary.checked, summary.skipped, summary.not_checked) == (5, 0, 0)
    assert summary.count("broken") == 2 and summary.count("skipped") == 0
    saved = store.latest_link_checks()
    assert set(saved) == set(table)
    moved = saved["https://moved.example.invalid/"]
    assert (moved.status, moved.http_status, moved.final_url, moved.message) == (
        "redirected", 200, "https://new.example.invalid/", "Moves to https://new.example.invalid/")
    gone = saved["https://gone.example.invalid/"]
    assert (gone.status, gone.http_status, gone.message) == ("broken", 404, "Not found (404).")
    assert saved["https://down.example.invalid/"].http_status is None
    assert saved["https://blocks.example.invalid/"].message == (
        "The site answered 403; it may block checks. Open it yourself.")
    assert saved["https://ok.example.invalid/"].checked_at.startswith("2026-06-02T09:00:00")


def test_a_fetch_that_raises_becomes_a_result_and_does_not_stop_the_rest(store: Store) -> None:
    table: dict[str, LinkOutcome | BaseException] = {
        "https://a.example.invalid/": socket.gaierror(-2, "x"),
        "https://b.example.invalid/": RuntimeError("boom"),
    }
    summary = check_links(store, ["https://a.example.invalid/", "https://b.example.invalid/",
                                  "https://c.example.invalid/"], fetch=fake_fetch(table))
    saved = store.latest_link_checks()
    assert saved["https://a.example.invalid/"].status == "broken"
    assert saved["https://b.example.invalid/"].status == "error"
    assert saved["https://c.example.invalid/"].status == "ok"
    assert summary.counts == {"broken": 1, "error": 1, "ok": 1}


def test_youtube_links_are_skipped_not_fetched(store: Store) -> None:
    urls = [f"https://youtu.be/{TARGET}", "https://www.youtube.com/@someone",
            "https://ext.example.invalid/"]
    seen: list[str] = []
    summary = check_links(store, urls, fetch=fake_fetch({}, seen))
    assert seen == ["https://ext.example.invalid/"]
    assert (summary.checked, summary.skipped, summary.not_checked) == (1, 2, 0)
    assert summary.counts == {"skipped": 2, "ok": 1}
    saved = store.latest_link_checks()
    assert saved[f"https://youtu.be/{TARGET}"].status == "skipped"
    assert saved[f"https://youtu.be/{TARGET}"].message == SKIPPED_MESSAGE
    assert saved["https://www.youtube.com/@someone"].http_status is None


def test_max_links_keeps_the_most_reach_and_counts_the_rest(store: Store) -> None:
    videos = [video(1, 10, "https://low.example.invalid/ https://mid.example.invalid/"),
              video(2, 50, "https://mid.example.invalid/ https://top.example.invalid/"),
              video(3, 500, "https://top.example.invalid/ https://zero.example.invalid/")]
    entries = inventory(videos, {})
    assert [e.url for e in entries][:2] == ["https://top.example.invalid/", "https://zero.example.invalid/"]
    seen: list[str] = []
    lines: list[str] = []
    summary = check_links(store, entries, fetch=fake_fetch({}, seen), report=lines.append,
                          max_links=2, workers=1)
    assert sorted(seen) == ["https://top.example.invalid/", "https://zero.example.invalid/"]
    assert (summary.checked, summary.not_checked) == (2, 2)
    assert set(store.latest_link_checks()) == set(seen)
    assert lines[0] == "Checking the 2 links with the most reach; 2 are not checked."


def test_max_links_with_plain_strings_keeps_the_order_given(store: Store) -> None:
    urls = [f"https://s{n}.example.invalid/" for n in range(5)]
    seen: list[str] = []
    check_links(store, urls, fetch=fake_fetch({}, seen), max_links=3, workers=1)
    assert sorted(seen) == urls[:3]


def test_max_links_does_not_cut_the_youtube_skips(store: Store) -> None:
    summary = check_links(store, [f"https://youtu.be/{TARGET}", "https://a.example.invalid/"],
                          fetch=fake_fetch({}), max_links=0)
    assert (summary.checked, summary.skipped, summary.not_checked) == (0, 1, 1)
    assert set(store.latest_link_checks()) == {f"https://youtu.be/{TARGET}"}
    assert MAX_LINKS == 300


def test_duplicate_urls_are_visited_once(store: Store) -> None:
    seen: list[str] = []
    check_links(store, ["https://a.example.invalid/"] * 3, fetch=fake_fetch({}, seen))
    assert seen == ["https://a.example.invalid/"]


def test_progress_lines(store: Store) -> None:
    urls = [f"https://l{n:02d}.example.invalid/" for n in range(23)]
    lines: list[str] = []
    check_links(store, urls, fetch=fake_fetch({}), report=lines.append, workers=3)
    assert lines == ["Checked 10 of 23 links", "Checked 20 of 23 links", "Checked 23 of 23 links"]
    one: list[str] = []
    check_links(store, ["https://solo.example.invalid/"], fetch=fake_fetch({}), report=one.append)
    assert one == ["Checked 1 of 1 link"]
    none: list[str] = []
    check_links(store, [f"https://youtu.be/{TARGET}"], fetch=fake_fetch({}), report=none.append)
    assert none == []


def test_a_new_check_replaces_the_latest_but_keeps_history(store: Store) -> None:
    url = "https://a.example.invalid/"
    check_links(store, [url], fetch=fake_fetch({url: outcome_for_status("u", 404)}))
    check_links(store, [url], fetch=fake_fetch({}))
    assert store.latest_link_checks()[url].status == "ok"


def test_inventory_shows_the_saved_checks(store: Store) -> None:
    url = "https://gone.example.invalid/"
    check_links(store, [url], fetch=fake_fetch({url: outcome_for_status("u", 404)}), now=NOW)
    entries = inventory([video(1, 7, f"Link {url}")], store.latest_link_checks())
    assert entries[0].check is not None and entries[0].check.status == "broken"


def test_with_no_urls_nothing_happens(store: Store) -> None:
    summary = check_links(store, [], fetch=fake_fetch({}))
    assert summary.counts == {} and summary.checked == 0 and store.latest_link_checks() == {}


# --- the real fetch, against a throwaway server on this computer's loopback only -------------------
def test_fetch_url_head_get_fallback_redirects_and_user_agent() -> None:
    import http.server

    from yezatas.links import USER_AGENT, fetch_url
    log: list[tuple[str, str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self) -> None:
            log.append((self.command, self.path, self.headers.get("User-Agent", "")))
            if self.path == "/ok":
                self.send_response(200)
            elif self.path == "/nohead":
                self.send_response(405 if self.command == "HEAD" else 200)
            elif self.path == "/blocks":
                self.send_response(403)
            elif self.path == "/old":
                self.send_response(301)
                self.send_header("Location", "/new")
            elif self.path == "/new":
                self.send_response(200)
            else:
                self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_HEAD = do_GET = _answer

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert fetch_url(f"{base}/ok").status == "ok"
        assert fetch_url(f"{base}/nohead").status == "ok"
        assert [c[0] for c in log if c[1] == "/nohead"] == ["HEAD", "GET"]
        blocked = fetch_url(f"{base}/blocks")
        assert (blocked.status, blocked.http_status) == ("error", 403)
        assert [c[0] for c in log if c[1] == "/blocks"] == ["HEAD", "GET"]
        gone = fetch_url(f"{base}/missing")
        assert (gone.status, gone.http_status) == ("broken", 404)
        assert [c[0] for c in log if c[1] == "/missing"] == ["HEAD"]
        moved = fetch_url(f"{base}/old")
        assert (moved.status, moved.final_url) == ("redirected", f"{base}/new")
        assert all(c[2] == USER_AGENT for c in log) and "Yezatas" in USER_AGENT
    finally:
        server.shutdown()
        server.server_close()
    assert fetch_url("ftp://a.example.invalid/x").status == "error"
