from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yezatas.quota import (COSTS, QuotaExceeded, QuotaGuard, UnknownOperation, cost_of,
                           quota_day)
from yezatas.store import Store


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "c.sqlite") as s:
        yield s


@pytest.fixture
def clock() -> Clock:
    return Clock(utc(2026, 1, 15, 12, 0))


def test_cost_table() -> None:
    expected = {"videos.list": 1, "channels.list": 1, "playlistItems.list": 1,
                "videos.update": 50, "thumbnails.set": 50, "captions.list": 50,
                "captions.download": 200, "captions.insert": 400, "captions.update": 450}
    assert COSTS == expected
    assert cost_of("captions.insert") == 400


def test_unknown_operation(store: Store, clock: Clock) -> None:
    with pytest.raises(UnknownOperation):
        cost_of("videos.delete")
    g = QuotaGuard(store, 1000, clock)
    with pytest.raises(UnknownOperation):
        g.charge("videos.delete")
    with pytest.raises(UnknownOperation):
        g.preview([("videos.delete", 1)])
    assert issubclass(UnknownOperation, KeyError)
    assert g.used() == 0


def test_quota_day_winter_boundary() -> None:
    assert quota_day(utc(2026, 1, 15, 7, 59)) == "2026-01-14"
    assert quota_day(utc(2026, 1, 15, 8, 0)) == "2026-01-15"


def test_quota_day_summer_boundary() -> None:
    assert quota_day(utc(2026, 7, 15, 6, 59)) == "2026-07-14"
    assert quota_day(utc(2026, 7, 15, 7, 0)) == "2026-07-15"


def test_quota_day_dst_transitions() -> None:
    # Spring forward 2026-03-08: midnight Pacific of the 8th is 08:00 UTC, of the 9th 07:00 UTC.
    assert quota_day(utc(2026, 3, 8, 7, 59)) == "2026-03-07"
    assert quota_day(utc(2026, 3, 8, 8, 0)) == "2026-03-08"
    assert quota_day(utc(2026, 3, 9, 6, 59)) == "2026-03-08"
    assert quota_day(utc(2026, 3, 9, 7, 0)) == "2026-03-09"
    # Fall back 2026-11-01: midnight of the 1st is 07:00 UTC, of the 2nd is 08:00 UTC.
    assert quota_day(utc(2026, 11, 1, 6, 59)) == "2026-10-31"
    assert quota_day(utc(2026, 11, 1, 7, 0)) == "2026-11-01"
    assert quota_day(utc(2026, 11, 2, 7, 59)) == "2026-11-01"
    assert quota_day(utc(2026, 11, 2, 8, 0)) == "2026-11-02"


def test_quota_day_accepts_other_timezones() -> None:
    plus2 = timezone(timedelta(hours=2))
    assert quota_day(datetime(2026, 1, 15, 9, 59, tzinfo=plus2)) == "2026-01-14"
    assert quota_day(datetime(2026, 1, 15, 10, 0, tzinfo=plus2)) == "2026-01-15"


def test_naive_datetime_rejected(store: Store) -> None:
    with pytest.raises(ValueError):
        quota_day(datetime(2026, 1, 15, 12, 0))
    g = QuotaGuard(store, 100, lambda: datetime(2026, 1, 15, 12, 0))
    with pytest.raises(ValueError):
        g.used()
    with pytest.raises(ValueError):
        g.charge("videos.list")


def test_charge_and_remaining(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 1000, clock)
    assert g.remaining() == 1000
    assert g.charge("videos.update", 3, note="batch") == 150
    assert g.used() == 150 and g.remaining() == 850
    rows = store.quota_entries("2026-01-15")
    assert len(rows) == 1
    assert (rows[0]["operation"], rows[0]["units"], rows[0]["count"], rows[0]["note"]) == (
        "videos.update", 150, 3, "batch")


def test_boundary_exactly_at_budget_allowed_one_over_refused(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    g.charge("videos.update", 1)
    assert g.charge("videos.update", 1) == 50  # exactly at budget
    assert g.remaining() == 0
    with pytest.raises(QuotaExceeded):
        g.charge("videos.list")  # one over
    assert g.used() == 100


def test_refusal_message_has_needed_and_remaining(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    g.charge("videos.update")
    with pytest.raises(QuotaExceeded, match=r"need 400 .* only 50 remaining"):
        g.charge("captions.insert")
    assert g.used() == 50


@pytest.mark.parametrize("count", [0, -1])
def test_count_must_be_positive(store: Store, clock: Clock, count: int) -> None:
    g = QuotaGuard(store, 100, clock)
    with pytest.raises(ValueError):
        g.charge("videos.list", count)
    with pytest.raises(ValueError):
        g.preview([("videos.list", count)])
    assert g.used() == 0


def test_reset_across_pacific_midnight_winter(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    clock.now = utc(2026, 1, 15, 7, 59)
    g.charge("videos.update", 2)
    assert g.remaining() == 0
    with pytest.raises(QuotaExceeded):
        g.charge("videos.list")
    clock.now = utc(2026, 1, 15, 8, 0)
    assert g.used() == 0 and g.remaining() == 100
    g.charge("videos.update", 2)
    assert store.quota_used("2026-01-14") == 100
    assert store.quota_used("2026-01-15") == 100


def test_reset_across_pacific_midnight_summer(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    clock.now = utc(2026, 7, 15, 6, 59)
    g.charge("videos.update", 2)
    assert g.remaining() == 0
    clock.now = utc(2026, 7, 15, 7, 0)
    assert g.remaining() == 100


def test_preview_does_not_write(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 500, clock)
    g.charge("videos.update", 2)
    p = g.preview([("videos.update", 3), ("videos.list", 10)])
    assert p.lines == (("videos.update", 3, 150), ("videos.list", 10, 10))
    assert (p.total, p.used, p.remaining, p.fits) == (160, 100, 400, True)
    assert store.quota_used("2026-01-15") == 100
    assert len(store.quota_entries("2026-01-15")) == 1


def test_preview_fits_flag(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    assert g.preview([("videos.update", 2)]).fits is True
    assert g.preview([("videos.update", 2), ("videos.list", 1)]).fits is False
    assert g.preview([]).total == 0


def test_charge_plan_all_or_nothing(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    with pytest.raises(QuotaExceeded):
        g.charge_plan([("videos.update", 1), ("videos.update", 1), ("videos.list", 1)])
    assert g.used() == 0 and store.quota_entries("2026-01-15") == []
    with pytest.raises(UnknownOperation):
        g.charge_plan([("videos.list", 1), ("bogus.op", 1)])
    assert g.used() == 0


def test_charge_plan_success(store: Store, clock: Clock) -> None:
    g = QuotaGuard(store, 100, clock)
    assert g.charge_plan([("videos.update", 1), ("videos.list", 5)], note="plan") == 55
    assert g.used() == 55
    assert [r["note"] for r in store.quota_entries("2026-01-15")] == ["plan", "plan"]
