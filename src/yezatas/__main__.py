"""Tiny command line: python -m yezatas signin | pull | quota."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import ensure_dirs, load_config
from .store import Store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yezatas", description="Yezatas YouTube channel assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("signin", help="sign in to YouTube (read-only access)")
    sub.add_parser("pull", help="pull the channel's videos into the local catalog")
    sub.add_parser("quota", help="show today's YouTube API quota use")
    args = parser.parse_args(argv)

    config = load_config()
    ensure_dirs(config)

    if args.command == "signin":
        from .auth import MissingClientSecrets, sign_in

        try:
            sign_in(config)
        except MissingClientSecrets as exc:
            print(exc, file=sys.stderr)
            return 1
        print("Signed in (read-only).")
        return 0

    with Store(config.catalog_db) as store:
        from .quota import QuotaExceeded, QuotaGuard

        guard = QuotaGuard(store, config.daily_quota_budget)
        if args.command == "quota":
            print(f"Quota today: {guard.used()} used, {guard.remaining()} remaining "
                  f"of {guard.budget}")
            return 0

        from .auth import load_credentials
        from .catalog import NoChannelError, pull_catalog
        from .youtube import GoogleYouTubeReader

        creds = load_credentials(config)
        if creds is None:
            print("Not signed in. Run: python -m yezatas signin", file=sys.stderr)
            return 1
        try:
            result = pull_catalog(GoogleYouTubeReader(creds), store, guard)
        except (QuotaExceeded, NoChannelError) as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Pulled {result.video_count} videos ({result.units_used} quota units).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
