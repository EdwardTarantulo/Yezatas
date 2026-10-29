"""Tiny command line: python -m yezatas ui | signin | pull | quota."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .config import Config, ensure_dirs, load_config
from .store import Store


def _run_ui(config: Config, port: int, open_browser: bool) -> int:
    import socket

    import uvicorn

    from .web.app import create_app

    if not 1 <= port <= 65535:
        print(f"Port must be between 1 and 65535, got {port}.", file=sys.stderr)
        return 1
    with socket.socket() as probe:  # fail early with a clear message if the port is taken
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            print(f"Port {port} is already in use. Try: python -m yezatas ui --port {port + 1}",
                  file=sys.stderr)
            return 1
    app = create_app(config, port=port, open_browser=open_browser)
    print("Yezatas is starting. Open this link in your browser (it signs this browser in):")
    print(f"  {app.state.start_url}", flush=True)
    print("Keep this window open while you use Yezatas. Press Ctrl+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yezatas", description="Yezatas YouTube channel assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui", help="open the local web UI (setup screen)")
    ui.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765)")
    ui.add_argument("--no-browser", action="store_true", help="do not open the browser")
    sub.add_parser("signin", help="sign in to YouTube (read-only access)")
    sub.add_parser("pull", help="pull the channel's videos into the local catalog")
    sub.add_parser("quota", help="show today's YouTube API quota use")
    args = parser.parse_args(argv)

    config = load_config()
    ensure_dirs(config)

    if args.command == "ui":
        return _run_ui(config, args.port, not args.no_browser)

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
