"""Setup checklist: work out each step's status from real state. Pure, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..youtube import MAX_RESULTS
from .clientfile import ClientInfo
from .jobs import JobSnapshot

PILLS = {"done": "Done", "todo": "To do", "problem": "Problem", "working": "Working",
         "unverified": "Not checked yet"}


@dataclass(frozen=True)
class CheckState:
    """Result of the connection check (channels.list)."""

    ok: bool
    kind: str = ""
    message: str = ""
    channel_id: str = ""
    channel_title: str = ""
    video_count: int = 0
    has_uploads: bool = False


@dataclass(frozen=True)
class Step:
    number: int
    key: str
    title: str
    status: str  # done | todo | problem | working | unverified (a later step will confirm it)
    note: str = ""  # the problem and its fix, shown when status is "problem"
    enabled: bool = True  # False while an earlier step blocks this one

    @property
    def pill(self) -> str:
        return PILLS[self.status]


def estimate_pull_units(video_count: int) -> int:
    """channels.list + the uploads playlist pages + the videos.list batches (1 unit each).

    `video_count` comes from the public statistics, so private videos can make the real cost a
    little higher. An empty channel still costs the channels call and one empty playlist page.
    """
    batches = math.ceil(video_count / MAX_RESULTS)
    return 1 + max(1, batches) + batches


def compute_steps(
    client: ClientInfo | None,
    client_problem: str,
    signed_in: bool,
    signin: JobSnapshot,
    check: CheckState | None,
    pull: JobSnapshot,
    pulled: bool,
) -> list[Step]:
    check_failed = check is not None and not check.ok
    check_kind = check.kind if check_failed and check else ""
    signin_failed = signin.status == "failed" and signin.error_kind != "cancelled"
    check_ok = check is not None and check.ok
    pull_failed = pull.status == "failed"

    # 1. Google Cloud project: cannot be detected, so it is "done" once the client file is valid.
    s1 = Step(1, "project", "Create a Google Cloud project", "done" if client else "todo")

    # 2. YouTube Data API: only a failed call can tell us it is off.
    if check_kind == "api_disabled" or (pull_failed and pull.error_kind == "api_disabled"):
        message = check.message if check_kind == "api_disabled" and check else pull.message
        s2 = Step(2, "api", "Enable the YouTube Data API v3", "problem", message)
    elif check_ok or pulled:
        s2 = Step(2, "api", "Enable the YouTube Data API v3", "done")
    elif client:  # the user has moved on; step 7 will confirm it
        s2 = Step(2, "api", "Enable the YouTube Data API v3", "unverified")
    else:
        s2 = Step(2, "api", "Enable the YouTube Data API v3", "todo")

    # 3. Consent screen: a finished sign-in proves it.
    title3 = "Set up the consent screen"
    if signed_in:
        s3 = Step(3, "consent", title3, "done")
    elif client:  # the user has moved on; signing in (step 6) will confirm it
        s3 = Step(3, "consent", title3, "unverified")
    else:
        s3 = Step(3, "consent", title3, "todo")

    # 4. Test user: a finished sign-in proves it. access_denied (and a sign-in that never comes
    # back, which is what Google's "Access blocked" page looks like from here) point to it.
    title4 = "Add yourself as a test user"
    if not signed_in and signin_failed and signin.error_kind in ("access_denied", "timeout"):
        s4 = Step(4, "testusers", title4, "problem", signin.message)
    elif signed_in:
        s4 = Step(4, "testusers", title4, "done")
    elif client:
        s4 = Step(4, "testusers", title4, "unverified")
    else:
        s4 = Step(4, "testusers", title4, "todo")

    # 5. Desktop client file.
    if client:
        s5 = Step(5, "client", "Create a Desktop app client and add the JSON file", "done")
    elif client_problem:
        s5 = Step(5, "client", "Create a Desktop app client and add the JSON file", "problem",
                  client_problem)
    else:
        s5 = Step(5, "client", "Create a Desktop app client and add the JSON file", "todo")

    # 6. Sign in.
    title6 = "Sign in with Google (read-only)"
    if client is None:
        s6 = Step(6, "signin", title6, "todo", enabled=False)
    elif signin.running:
        s6 = Step(6, "signin", title6, "working")
    elif check_kind == "auth":
        s6 = Step(6, "signin", title6, "problem", check.message)  # type: ignore[union-attr]
    elif signed_in:
        s6 = Step(6, "signin", title6, "done")
    elif signin_failed:
        s6 = Step(6, "signin", title6, "problem", signin.message)
    else:
        s6 = Step(6, "signin", title6, "todo")

    # 7. Connection check.
    title7 = "Check the connection"
    if not signed_in:
        s7 = Step(7, "check", title7, "todo", enabled=False)
    elif check_ok:
        s7 = Step(7, "check", title7, "done")
    elif check_failed:
        # The full fix here too (it also shows on step 2 or 6), so nobody has to scroll up for it.
        s7 = Step(7, "check", title7, "problem", check.message)  # type: ignore[union-attr]
    else:
        s7 = Step(7, "check", title7, "todo")

    # 8. Catalog pull.
    title8 = "Pull your catalog"
    can_pull = check_ok and check is not None and check.has_uploads
    if pull.running:
        s8 = Step(8, "pull", title8, "working")
    elif pull_failed and pull.error_kind not in ("api_disabled",):
        s8 = Step(8, "pull", title8, "problem", pull.message, enabled=can_pull)
    elif pulled:
        s8 = Step(8, "pull", title8, "done", enabled=can_pull)
    else:
        s8 = Step(8, "pull", title8, "todo", enabled=can_pull)

    return [s1, s2, s3, s4, s5, s6, s7, s8]
