"""Channel this week (Today page): the channel's daily views, watch time and subscribers and where its
views came from, read from YouTube Analytics (read-only), plus how each of the newest uploads is doing
against the uploads before it at the same age. Code does every number; saved as JSON under
home/analytics; nothing is written to YouTube."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .outliers import _parse_time, _utc
from .seo import is_short
from .store import Video

LAG_DAYS = 3  # Analytics runs 2 to 3 days behind; the newest day read is today minus this
DAYS = 14  # two weeks: the last 7 days and the 7 before
WEEK_DAYS = 7
SOURCE_DAYS = 28
COUNTRY_DAYS = 90
TOP_COUNTRIES = 8
LATEST = 3  # newest uploads compared
EARLIER = 10  # uploads before each one that make "your usual"
MIN_EARLIER = 3  # fewer earlier uploads of the same kind: no comparison
MIN_AGE_DAYS = 3  # fewer days of data: "too new to compare"
MAX_AGE_DAYS = 28  # the comparison window never grows past this
ABOVE = 1.25  # ratio at or over this: above your usual
BELOW = 0.8  # ratio at or under this: below your usual
DAILY_METRICS = "views,estimatedMinutesWatched,subscribersGained,subscribersLost"
SOURCE_LABELS = {
    "YT_SEARCH": "YouTube search",
    "RELATED_VIDEO": "Suggested videos",
    "SUBSCRIBER": "Browse features",
    "EXT_URL": "External sites",
    "NO_LINK_OTHER": "Direct or unknown",
    "PLAYLIST": "Playlists",
    "YT_CHANNEL": "Channel pages",
    "SHORTS": "Shorts feed",
    "NOTIFICATION": "Notifications",
    "END_SCREEN": "End screens",
    "YT_OTHER_PAGE": "Other YouTube pages",
    "HASHTAGS": "Hashtag pages",
    "ADVERTISING": "YouTube ads",
}


COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "CA": "Canada", "AU": "Australia", "NZ": "New Zealand",
    "IE": "Ireland", "DE": "Germany", "AT": "Austria", "CH": "Switzerland", "FR": "France", "BE": "Belgium",
    "NL": "Netherlands", "LU": "Luxembourg", "ES": "Spain", "PT": "Portugal", "IT": "Italy", "GR": "Greece",
    "DK": "Denmark", "NO": "Norway", "SE": "Sweden", "FI": "Finland", "IS": "Iceland", "PL": "Poland",
    "CZ": "Czechia", "SK": "Slovakia", "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "SI": "Slovenia", "RS": "Serbia", "UA": "Ukraine", "LT": "Lithuania", "LV": "Latvia", "EE": "Estonia",
    "TR": "Turkey", "IL": "Israel", "AE": "United Arab Emirates", "SA": "Saudi Arabia", "EG": "Egypt",
    "ZA": "South Africa", "NG": "Nigeria", "KE": "Kenya", "MA": "Morocco", "IN": "India", "PK": "Pakistan",
    "BD": "Bangladesh", "LK": "Sri Lanka", "JP": "Japan", "KR": "South Korea", "CN": "China", "TW": "Taiwan",
    "HK": "Hong Kong", "SG": "Singapore", "MY": "Malaysia", "TH": "Thailand", "VN": "Vietnam",
    "PH": "Philippines", "ID": "Indonesia", "MX": "Mexico", "BR": "Brazil", "AR": "Argentina", "CL": "Chile",
    "CO": "Colombia", "PE": "Peru",
}


class ChannelReader(Protocol):
    def query(self, start_date: str, end_date: str, metrics: str, dimensions: str,
              filters: str, sort: str) -> dict[str, Any]: ...


class GoogleChannelReader:
    """YouTube Analytics v2 behind ChannelReader. Thin on purpose; not unit-tested."""

    def __init__(self, credentials: Any) -> None:
        from googleapiclient.discovery import build

        self._service = build(
            "youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False
        )

    def query(self, start_date: str, end_date: str, metrics: str, dimensions: str,
              filters: str, sort: str) -> dict[str, Any]:
        extra = {"filters": filters} if filters else {}
        return (
            self._service.reports()
            .query(ids="channel==MINE", startDate=start_date, endDate=end_date, metrics=metrics,
                   dimensions=dimensions, sort=sort, **extra)
            .execute()
        )


def table(response: dict[str, Any], names: Sequence[str]) -> list[tuple[Any, ...]]:
    try:
        headers = response.get("columnHeaders")
        rows = response.get("rows")
        if not isinstance(headers, list) or not isinstance(rows, list):
            return []
        col_names: list[str] = []
        for h in headers:
            if not isinstance(h, dict):
                return []
            n = h.get("name")
            if not isinstance(n, str):
                return []
            col_names.append(n)
        indices: list[int] = []
        for name in names:
            if name not in col_names:
                return []
            indices.append(col_names.index(name))
        out: list[tuple[Any, ...]] = []
        if not indices:
            # no columns requested: return empty tuples for valid rows? but spec not needed
            return []
        max_idx = max(indices)
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) <= max_idx:
                continue
            out.append(tuple(row[i] for i in indices))
        return out
    except Exception:
        return []


def last_day(now: datetime) -> date:
    return _utc(now).date() - timedelta(days=LAG_DAYS)


@dataclass(frozen=True)
class Day:
    date: str
    views: int
    minutes: int
    gained: int
    lost: int


def fetch_days(reader: ChannelReader, end: date) -> tuple[Day, ...]:
    start = end - timedelta(days=DAYS - 1)
    resp = reader.query(
        start.isoformat(),
        end.isoformat(),
        DAILY_METRICS,
        "day",
        "",
        "day",
    )
    rows = table(resp, ("day", "views", "estimatedMinutesWatched", "subscribersGained", "subscribersLost"))
    by_day: dict[str, tuple[Any, Any, Any, Any]] = {}
    for day_s, views, minutes, gained, lost in rows:
        if not isinstance(day_s, str):
            continue
        by_day[day_s] = (views, minutes, gained, lost)
    result: list[Day] = []
    for i in range(DAYS):
        d = start + timedelta(days=i)
        iso = d.isoformat()
        vals = by_day.get(iso)
        if vals is None:
            result.append(Day(iso, 0, 0, 0, 0))
        else:
            v_raw, m_raw, g_raw, l_raw = vals
            try:
                v = int(v_raw)
            except Exception:
                v = 0
            try:
                m = int(m_raw)
            except Exception:
                m = 0
            try:
                g = int(g_raw)
            except Exception:
                g = 0
            try:
                l = int(l_raw)
            except Exception:
                l = 0
            result.append(Day(iso, v, m, g, l))
    return tuple(result)


@dataclass(frozen=True)
class Totals:
    views: int
    minutes: int
    subscribers: int


def week_totals(days: Sequence[Day]) -> tuple[Totals, Totals]:
    def totals(part: Sequence[Day]) -> Totals:
        return Totals(sum(d.views for d in part), sum(d.minutes for d in part),
                      sum(d.gained - d.lost for d in part))

    return totals(days[-WEEK_DAYS:]), totals(days[-2 * WEEK_DAYS:-WEEK_DAYS])


def change(this: int, before: int) -> int | None:
    if before <= 0:
        return None
    return round((this - before) * 100 / before)


def fetch_sources(reader: ChannelReader, end: date) -> tuple[tuple[str, int], ...]:
    start = end - timedelta(days=SOURCE_DAYS - 1)
    resp = reader.query(
        start.isoformat(),
        end.isoformat(),
        "views",
        "insightTrafficSourceType",
        "",
        "-views",
    )
    rows = table(resp, ("insightTrafficSourceType", "views"))
    pairs: list[tuple[str, int]] = []
    for code, views_raw in rows:
        if not isinstance(code, str):
            continue
        try:
            v = int(views_raw)
        except Exception:
            continue
        if v <= 0:
            continue
        pairs.append((code, v))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return tuple(pairs)


def source_label(code: str) -> str:
    if code in SOURCE_LABELS:
        return SOURCE_LABELS[code]
    return code.replace("_", " ").capitalize()


def fetch_countries(reader: ChannelReader, end: date) -> tuple[tuple[str, int], ...]:
    """Views by viewer country over COUNTRY_DAYS days ending `end`: ONE query, most views first."""
    start = end - timedelta(days=COUNTRY_DAYS - 1)
    resp = reader.query(start.isoformat(), end.isoformat(), "views", "country", "", "-views")
    pairs: list[tuple[str, int]] = []
    for code, views_raw in table(resp, ("country", "views")):
        if not isinstance(code, str) or not code.strip():
            continue
        try:
            v = int(views_raw)
        except Exception:
            continue
        if v > 0:
            pairs.append((code.strip().upper(), v))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return tuple(pairs)


def country_name(code: str) -> str:
    return COUNTRY_NAMES.get(code.upper(), code.upper())


def source_shares(sources: Sequence[tuple[str, int]], top: int = 3) -> list[tuple[str, int]]:
    total = sum(v for _, v in sources)
    if total == 0:
        return []
    out: list[tuple[str, int]] = []
    for code, views in list(sources)[:top]:
        pct = round(views * 100 / total)
        out.append((source_label(code), pct))
    return out


@dataclass(frozen=True)
class SameAge:
    video_id: str
    status: str
    age_days: int
    views: int
    usual: int | None
    compared: int

    @property
    def ratio(self) -> float | None:
        if self.status != "compared":
            return None
        if self.usual is None or self.usual <= 0:
            return None
        return self.views / self.usual

    @property
    def verdict(self) -> str:
        r = self.ratio
        if r is None:
            return ""
        if r >= ABOVE:
            return "above"
        if r <= BELOW:
            return "below"
        return "usual"


def newest_uploads(videos: Sequence[Video], count: int = LATEST) -> list[Video]:
    eligible: list[tuple[Video, datetime]] = []
    for v in videos:
        if v.privacy_status != "public":
            continue
        dt = _parse_time(v.published_at)
        if dt is None:
            continue
        eligible.append((v, dt))
    eligible.sort(key=lambda x: (-x[1].timestamp(), x[0].video_id))
    return [v for v, _ in eligible[:count]]


def compare_latest(reader: ChannelReader, videos: Sequence[Video], end: date) -> tuple[list[SameAge], int]:
    newest = newest_uploads(videos)
    # cache video_id -> dict[date_str, views]
    cache: dict[str, dict[str, int]] = {}
    calls = 0

    def get_views(video_id: str, pub_date: date) -> dict[str, int]:
        nonlocal calls
        if video_id in cache:
            return cache[video_id]
        start = pub_date
        q_end = pub_date + timedelta(days=MAX_AGE_DAYS - 1)
        if q_end > end:
            q_end = end
        if start > q_end:
            cache[video_id] = {}
            return cache[video_id]
        resp = reader.query(
            start.isoformat(),
            q_end.isoformat(),
            "views",
            "day",
            f"video=={video_id}",
            "day",
        )
        calls += 1
        rows = table(resp, ("day", "views"))
        m: dict[str, int] = {}
        for d_str, v_raw in rows:
            if not isinstance(d_str, str):
                continue
            try:
                v = int(v_raw)
            except Exception:
                continue
            m[d_str] = v
        cache[video_id] = m
        return m

    # Pre-parse all videos for earlier filtering
    parsed: list[tuple[Video, datetime, date]] = []
    for v in videos:
        if v.privacy_status != "public":
            continue
        dt = _parse_time(v.published_at)
        if dt is None:
            continue
        parsed.append((v, dt, dt.date()))

    result: list[SameAge] = []
    for v in newest:
        dt_v = _parse_time(v.published_at)
        assert dt_v is not None
        pub_date = dt_v.date()
        raw_age = (end - pub_date).days + 1
        age = raw_age
        if age > MAX_AGE_DAYS:
            age = MAX_AGE_DAYS
        if age < MIN_AGE_DAYS:
            result.append(SameAge(v.video_id, "too_new", max(age, 0), 0, None, 0))
            continue

        # earlier candidates
        earlier_candidates: list[tuple[Video, datetime]] = []
        is_v_short = is_short(v)
        for ev, edt, _ in parsed:
            if edt >= dt_v:
                continue
            if is_short(ev) != is_v_short:
                continue
            earlier_candidates.append((ev, edt))
        earlier_candidates.sort(key=lambda x: (-x[1].timestamp(), x[0].video_id))
        earlier = [ev for ev, _ in earlier_candidates[:EARLIER]]

        # own views
        own_map = get_views(v.video_id, pub_date)
        own_sum = 0
        for i in range(age):
            d = pub_date + timedelta(days=i)
            own_sum += own_map.get(d.isoformat(), 0)

        if len(earlier) < MIN_EARLIER:
            result.append(SameAge(v.video_id, "few_earlier", age, own_sum, None, len(earlier)))
            continue

        their_sums: list[int] = []
        for e in earlier:
            edt = _parse_time(e.published_at)
            assert edt is not None
            q = edt.date()
            emap = get_views(e.video_id, q)
            s = 0
            for i in range(age):
                d = q + timedelta(days=i)
                s += emap.get(d.isoformat(), 0)
            their_sums.append(s)

        usual = round(statistics.median(their_sums))
        result.append(SameAge(v.video_id, "compared", age, own_sum, usual, len(earlier)))

    return result, calls


@dataclass(frozen=True)
class ChannelStats:
    fetched_at: str
    end_date: str
    days: tuple[Day, ...]
    sources: tuple[tuple[str, int], ...]
    latest: tuple[SameAge, ...]
    calls: int
    countries: tuple[tuple[str, int], ...] = ()  # (country code, views), 90 days


def fetch_stats(reader: ChannelReader, videos: Sequence[Video], *, now: datetime | None = None,
               report: Callable[[str], None] | None = None) -> ChannelStats:
    if now is None:
        now = datetime.now(timezone.utc)
    end = last_day(now)
    if report is not None:
        report("Reading the channel's daily numbers")
    days = fetch_days(reader, end)
    if report is not None:
        report("Reading where views came from")
    sources = fetch_sources(reader, end)
    if report is not None:
        report("Reading where viewers watch from")
    countries = fetch_countries(reader, end)
    if report is not None:
        report("Comparing the newest uploads")
    latest_list, comp_calls = compare_latest(reader, videos, end)
    calls = 3 + comp_calls
    fetched_at = _utc(now).isoformat()
    return ChannelStats(
        fetched_at=fetched_at,
        end_date=end.isoformat(),
        days=days,
        sources=sources,
        latest=tuple(latest_list),
        calls=calls,
        countries=countries,
    )


def save_stats(home: Path, stats: ChannelStats) -> Path:
    dest = Path(home) / "analytics" / "channel.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "fetched_at": stats.fetched_at,
        "end_date": stats.end_date,
        "days": [
            {"date": d.date, "views": d.views, "minutes": d.minutes, "gained": d.gained, "lost": d.lost}
            for d in stats.days
        ],
        "sources": [[c, v] for c, v in stats.sources],
        "latest": [
            {"video_id": s.video_id, "status": s.status, "age_days": s.age_days, "views": s.views, "usual": s.usual, "compared": s.compared}
            for s in stats.latest
        ],
        "calls": stats.calls,
        "countries": [[c, v] for c, v in stats.countries],
    }
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix="channel.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, dest)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return dest


def _is_int_strict(v: Any) -> bool:
    return type(v) is int  # bool rejected


def load_stats(home: Path) -> ChannelStats | None:
    p = Path(home) / "analytics" / "channel.json"
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    try:
        if not isinstance(data, dict):
            return None
        fetched_at = data.get("fetched_at")
        end_date = data.get("end_date")
        days_raw = data.get("days")
        sources_raw = data.get("sources")
        latest_raw = data.get("latest")
        calls = data.get("calls")
        if not isinstance(fetched_at, str):
            return None
        if not isinstance(end_date, str):
            return None
        if not isinstance(days_raw, list):
            return None
        if not isinstance(sources_raw, list):
            return None
        if not isinstance(latest_raw, list):
            return None
        if not _is_int_strict(calls):
            return None
        days: list[Day] = []
        for item in days_raw:
            if not isinstance(item, dict):
                return None
            d = item.get("date")
            v = item.get("views")
            m = item.get("minutes")
            g = item.get("gained")
            l = item.get("lost")
            if not isinstance(d, str):
                return None
            if not _is_int_strict(v):
                return None
            if not _is_int_strict(m):
                return None
            if not _is_int_strict(g):
                return None
            if not _is_int_strict(l):
                return None
            days.append(Day(date=d, views=v, minutes=m, gained=g, lost=l))
        sources: list[tuple[str, int]] = []
        for item in sources_raw:
            if not isinstance(item, list) or len(item) != 2:
                return None
            c, v = item
            if not isinstance(c, str):
                return None
            if not _is_int_strict(v):
                return None
            sources.append((c, v))
        latest: list[SameAge] = []
        for item in latest_raw:
            if not isinstance(item, dict):
                return None
            vid = item.get("video_id")
            status = item.get("status")
            age_days = item.get("age_days")
            views = item.get("views")
            usual = item.get("usual")
            compared = item.get("compared")
            if not isinstance(vid, str):
                return None
            if not isinstance(status, str):
                return None
            if not _is_int_strict(age_days):
                return None
            if not _is_int_strict(views):
                return None
            if usual is not None and not _is_int_strict(usual):
                return None
            if not _is_int_strict(compared):
                return None
            latest.append(SameAge(video_id=vid, status=status, age_days=age_days, views=views, usual=usual, compared=compared))
        countries_raw = data.get("countries", [])  # files saved before countries existed have none
        if not isinstance(countries_raw, list):
            return None
        countries: list[tuple[str, int]] = []
        for item in countries_raw:
            if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                    or not _is_int_strict(item[1])):
                return None
            countries.append((item[0], item[1]))
        return ChannelStats(
            fetched_at=fetched_at,
            end_date=end_date,
            days=tuple(days),
            sources=tuple(sources),
            latest=tuple(latest),
            calls=calls,
            countries=tuple(countries),
        )
    except Exception:
        return None
