"""Setup checklist: work out each step's status from real state. Pure, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..youtube import MAX_RESULTS
from .clientfile import ClientInfo
from .jobs import JobSnapshot

PILLS = {"done": "Done", "todo": "To do", "problem": "Problem", "working": "Working"}


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
    status: str  # done | todo | problem | working
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
    else:
        s2 = Step(2, "api", "Enable the YouTube Data API v3", "todo")

    # 3. Consent screen and test user: a finished sign-in proves it; access_denied points here.
    if not signed_in and signin_failed and signin.error_kind in ("access_denied", "timeout"):
        s3 = Step(3, "consent", "Set up the consent screen and add yourself as a test user",
                  "problem", signin.message)
    elif signed_in:
        s3 = Step(3, "consent", "Set up the consent screen and add yourself as a test user", "done")
    else:
        s3 = Step(3, "consent", "Set up the consent screen and add yourself as a test user", "todo")

    # 4. Desktop client file.
    if client:
        s4 = Step(4, "client", "Create a Desktop app client and add the JSON file", "done")
    elif client_problem:
        s4 = Step(4, "client", "Create a Desktop app client and add the JSON file", "problem",
                  client_problem)
    else:
        s4 = Step(4, "client", "Create a Desktop app client and add the JSON file", "todo")

    # 5. Sign in.
    title5 = "Sign in with Google (read-only)"
    if client is None:
        s5 = Step(5, "signin", title5, "todo", enabled=False)
    elif signin.running:
        s5 = Step(5, "signin", title5, "working")
    elif check_kind == "auth":
        s5 = Step(5, "signin", title5, "problem", check.message)  # type: ignore[union-attr]
    elif signed_in:
        s5 = Step(5, "signin", title5, "done")
    elif signin_failed:
        s5 = Step(5, "signin", title5, "problem", signin.message)
    else:
        s5 = Step(5, "signin", title5, "todo")

    # 6. Connection check.
    title6 = "Check the connection"
    if not signed_in:
        s6 = Step(6, "check", title6, "todo", enabled=False)
    elif check_ok:
        s6 = Step(6, "check", title6, "done")
    elif check_failed and check_kind not in ("api_disabled", "auth"):
        s6 = Step(6, "check", title6, "problem", check.message)  # type: ignore[union-attr]
    elif check_failed:
        where = "step 2" if check_kind == "api_disabled" else "step 5"
        s6 = Step(6, "check", title6, "problem", f"The check failed. See {where}.")
    else:
        s6 = Step(6, "check", title6, "todo")

    # 7. Catalog pull.
    title7 = "Pull your catalog"
    can_pull = check_ok and check is not None and check.has_uploads
    if pull.running:
        s7 = Step(7, "pull", title7, "working")
    elif pull_failed and pull.error_kind not in ("api_disabled",):
        s7 = Step(7, "pull", title7, "problem", pull.message, enabled=can_pull)
    elif pulled:
        s7 = Step(7, "pull", title7, "done", enabled=can_pull)
    else:
        s7 = Step(7, "pull", title7, "todo", enabled=can_pull)

    return [s1, s2, s3, s4, s5, s6, s7]
