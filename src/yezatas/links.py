"""Link inventory and dead-link scanner: every link in the creator's descriptions, ranked by reach.

Code reads the stored descriptions (no YouTube call) and lists each link once, with the videos that
carry it and their summed views, so the link that matters most comes first. Links to the creator's own
private videos, or to videos not in the catalog, are findings from the catalog alone. Other links can be
visited once each, from the creator's computer, with a plain HEAD request (GET when a site refuses
HEAD); the answer is saved as a `LinkCheck`. Nothing here changes any text: replacing a dead link goes
through Bulk edit, which makes drafts.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .facts import extract_urls
from .store import LinkCheck, Store, Video

TIMEOUT_SECONDS = 10.0
MAX_LINKS = 300
WORKERS = 4
PROGRESS_EVERY = 10
USER_AGENT = f"Yezatas-link-checker/{__version__} (self-hosted, open source)"
SKIPPED_MESSAGE = "YouTube links are checked against your catalog."
UNREACHABLE_MESSAGE = "The site could not be reached."
_HEAD_REFUSED = (403, 405, 501)  # sites that refuse HEAD: ask again with GET
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


# --- reading a link -------------------------------------------------------------------------------
def _split(url: str):  # -> SplitResult | None
    try:
        return urlsplit(url if "://" in url else "//" + url)
    except ValueError:
        return None


def link_domain(url: str) -> str:
    """The host in lowercase without "www."; empty when the link cannot be read."""
    parts = _split(url)
    try:
        host = (parts.hostname or "").lower() if parts else ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _is_youtube(host: str) -> bool:
    return host in ("youtu.be", "youtube.com", "youtube-nocookie.com") or host.endswith(".youtube.com")


def youtube_video_id(url: str) -> str | None:
    """The 11-character id in a YouTube video link (watch?v=, youtu.be/, /shorts/), else None."""
    host = link_domain(url)
    parts = _split(url)
    if parts is None or not _is_youtube(host):
        return None
    segments = [s for s in parts.path.split("/") if s]
    candidate = ""
    if host == "youtu.be":
        candidate = segments[0] if segments else ""
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if segments == ["watch"]:
            values = parse_qs(parts.query).get("v") or [""]
            candidate = values[0]
        elif len(segments) >= 2 and segments[0] == "shorts":
            candidate = segments[1]
    return candidate if _VIDEO_ID.fullmatch(candidate) else None


def link_kind(url: str) -> str:
    """"video" (a YouTube video link), "youtube" (any other YouTube link) or "external"."""
    if youtube_video_id(url):
        return "video"
    return "youtube" if _is_youtube(link_domain(url)) else "external"


# --- the inventory --------------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkEntry:
    """One distinct link (the exact URL text) and where it appears."""

    url: str
    domain: str
    kind: str  # "video" | "youtube" | "external"
    video_ids: tuple[str, ...]  # videos whose description has it, most viewed first
    reach: int  # summed views of those videos
    public_count: int  # how many of those videos are public
    check: LinkCheck | None  # the latest check, if the link was ever checked
    target_id: str = ""  # "video" links: the video it points to
    target_state: str = ""  # "video" links: "public" | "unlisted" | "private" | "not_in_catalog"
    target_title: str = ""  # "video" links to a catalogued video: its title

    @property
    def finding(self) -> str:
        """A plain sentence when the catalog alone shows a problem, else an empty string."""
        if self.target_state == "private":
            return f'Links to your private video "{self.target_title}".'
        if self.target_state == "not_in_catalog":
            return "Not one of your videos, or deleted: Yezatas can't tell."
        return ""


def inventory(videos: Sequence[Video], checks: dict[str, LinkCheck]) -> list[LinkEntry]:
    """Every link in every description, once each, ranked by reach and then by URL."""
    by_id = {v.video_id: v for v in videos}
    carriers: dict[str, dict[str, Video]] = {}
    for video in videos:
        for url in extract_urls(video.description or ""):
            carriers.setdefault(url, {})[video.video_id] = video
    entries: list[LinkEntry] = []
    for url, found in carriers.items():
        ranked = sorted(found.values(), key=lambda v: (-v.view_count, v.video_id))
        kind = link_kind(url)
        target_id = target_state = target_title = ""
        if kind == "video":
            target_id = youtube_video_id(url) or ""
            target = by_id.get(target_id)
            if target is None:
                target_state = "not_in_catalog"
            else:
                target_title = target.title
                state = target.privacy_status
                target_state = state if state in ("public", "unlisted") else "private"
        entries.append(LinkEntry(
            url=url, domain=link_domain(url), kind=kind,
            video_ids=tuple(v.video_id for v in ranked),
            reach=sum(v.view_count for v in ranked),
            public_count=sum(1 for v in ranked if v.privacy_status == "public"),
            check=checks.get(url), target_id=target_id, target_state=target_state,
            target_title=target_title))
    entries.sort(key=lambda e: (-e.reach, e.url))
    return entries


# --- what one visit found -------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkOutcome:
    """The result of visiting one link, before it is saved."""

    status: str  # "ok" | "redirected" | "broken" | "error"
    http_status: int | None = None
    final_url: str = ""
    message: str = ""


Fetch = Callable[[str], LinkOutcome]


def _place(url: str) -> tuple[str, str, str, str]:
    """A link without its scheme, trailing slash, fragment, host case or default port."""
    parts = _split(url)
    if parts is None:
        return (url, "", "", "")
    try:
        host, port = (parts.hostname or "").lower(), parts.port
    except ValueError:
        return (url, "", "", "")
    default = {"http": 80, "https": 443}.get(parts.scheme)
    return (host, str(port) if port and port != default else "",
            parts.path.rstrip("/"), parts.query)


def same_place(url: str, final_url: str) -> bool:
    """True when the two differ only by scheme, a trailing slash, a fragment or host case."""
    return _place(url) == _place(final_url)


def outcome_for_status(url: str, http_status: int, final_url: str = "") -> LinkOutcome:
    """Map the HTTP status of the last answer (after redirects) to what the creator should read."""
    if 200 <= http_status < 300:
        if final_url and not same_place(url, final_url):
            return LinkOutcome("redirected", http_status, final_url, f"Moves to {final_url}")
        return LinkOutcome("ok", http_status, "", "")
    if http_status in (404, 410):
        return LinkOutcome("broken", http_status, "", f"Not found ({http_status}).")
    if 500 <= http_status < 600:
        return LinkOutcome("error", http_status, "", (
            f"The site answered {http_status} (a problem on the site). "
            "Try again later or open it yourself."))
    return LinkOutcome("error", http_status, "", (
        f"The site answered {http_status}; it may block checks. Open it yourself."))


def outcome_for_error(exc: BaseException, url: str = "") -> LinkOutcome:
    """Map an exception from a visit: unreachable sites are broken; the rest are only unclear."""
    if isinstance(exc, urllib.error.HTTPError):
        return outcome_for_status(url, exc.code)
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, BaseException):
        exc = exc.reason
    if isinstance(exc, (socket.gaierror, ConnectionRefusedError)):
        return LinkOutcome("broken", None, "", UNREACHABLE_MESSAGE)
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return LinkOutcome("error", None, "", (
            f"The site did not answer in {TIMEOUT_SECONDS:.0f} seconds. Open it yourself."))
    if isinstance(exc, ssl.SSLError):
        return LinkOutcome("error", None, "", (
            "The site's security certificate could not be checked. Open it yourself."))
    return LinkOutcome("error", None, "", "Yezatas could not check this link. Open it yourself.")


PRIVATE_HOST_MESSAGE = "This link points to a private address, so Yezatas did not visit it."


def _is_private_host(url: str) -> bool:
    """A link to this computer or the local network, by name or as an address. A public name
    that resolves to a private address is not caught here (nothing is resolved before the visit)."""
    try:
        host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan")):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def fetch_url(url: str, timeout: float = TIMEOUT_SECONDS) -> LinkOutcome:
    """Visit one link with the standard library: HEAD, then GET (body unread) if HEAD is refused."""
    target = url if "://" in url else "https://" + url
    if urlsplit(target).scheme not in ("http", "https"):
        return LinkOutcome("error", None, "", "Yezatas could not check this link. Open it yourself.")
    code = 0
    try:
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(
                target, method=method, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return outcome_for_status(target, response.status, response.geturl())
            except urllib.error.HTTPError as exc:
                code = exc.code
                if method == "HEAD" and code in _HEAD_REFUSED:
                    continue
                return outcome_for_status(target, code)
        return outcome_for_status(target, code)
    except Exception as exc:  # any network trouble becomes a sentence, never a crash
        return outcome_for_error(exc, target)


# --- the scan -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkCheckSummary:
    counts: dict[str, int]  # status -> links saved this run (including "skipped")
    checked: int  # external links visited
    skipped: int  # YouTube links, answered from the catalog instead
    not_checked: int  # external links left out by the `max_links` cut

    def count(self, status: str) -> int:
        return self.counts.get(status, 0)


def check_links(
    store: Store, urls: Sequence[LinkEntry | str], *,
    fetch: Fetch | None = None, now: datetime | None = None,
    report: Callable[[str], None] | None = None,
    max_links: int = MAX_LINKS, workers: int = WORKERS,
) -> LinkCheckSummary:
    """Visit the external links once each and save every result.

    `urls` are inventory entries (ranked by their reach) or plain URL strings (kept in the order given).
    YouTube links are not visited: each is saved as "skipped". Above `max_links`, only the external
    links with the most reach are visited and the rest are counted as not checked.
    """
    visit = fetch or fetch_url
    say = report or (lambda _line: None)
    seen: dict[str, int] = {}
    for item in urls:
        url, reach = (item.url, item.reach) if isinstance(item, LinkEntry) else (item, 0)
        seen.setdefault(url, reach)
    ranked = sorted(seen.items(), key=lambda pair: -pair[1])  # stable: ties keep the order given
    external = [url for url, _ in ranked if link_kind(url) == "external"]
    youtube = [url for url, _ in ranked if link_kind(url) != "external"]

    counts: dict[str, int] = {}
    for url in youtube:
        store.save_link_check(url, "skipped", None, "", SKIPPED_MESSAGE, now=now)
        counts["skipped"] = counts.get("skipped", 0) + 1

    chosen = external[:max(0, max_links)]
    not_checked = len(external) - len(chosen)
    if not_checked:
        say(f"Checking the {len(chosen)} links with the most reach; {not_checked} are not checked.")

    def one(url: str) -> LinkOutcome:
        if _is_private_host(url):
            return LinkOutcome("error", None, "", PRIVATE_HOST_MESSAGE)
        try:
            return visit(url)
        except Exception as exc:  # a failing fetch must not stop the others
            return outcome_for_error(exc, url)

    total = len(chosen)
    done = 0
    if chosen:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(one, url): url for url in chosen}
            for future in as_completed(futures):
                outcome = future.result()
                store.save_link_check(futures[future], outcome.status, outcome.http_status,
                                      outcome.final_url, outcome.message, now=now)
                counts[outcome.status] = counts.get(outcome.status, 0) + 1
                done += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    say(f"Checked {done} of {total} link{'s' if total != 1 else ''}")
    return LinkCheckSummary(counts=counts, checked=done, skipped=len(youtube),
                            not_checked=not_checked)
