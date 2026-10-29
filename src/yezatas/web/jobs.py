"""Background jobs for the web UI: one worker thread at a time, state guarded by a lock."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

Report = Callable[[str], None]
ErrorMapper = Callable[[BaseException], "tuple[str, str]"]  # exception -> (error_kind, message)


@dataclass(frozen=True)
class JobSnapshot:
    status: str  # "idle" | "running" | "done" | "failed"
    message: str = ""
    error_kind: str = ""
    result: Any = None
    started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self.status == "running"


class Job:
    """Runs `fn(report)` on a daemon thread. Only one run at a time.

    A run that outlives `timeout` seconds (or is cancelled) is abandoned: the job is marked failed
    so the UI can retry, and whatever the old thread does afterwards is ignored.
    """

    def __init__(
        self,
        map_error: ErrorMapper,
        timeout: float | None = None,
        timeout_message: str = "This took too long.",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._map_error = map_error
        self._timeout = timeout
        self._timeout_message = timeout_message
        self._clock = clock
        self._lock = threading.Lock()
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._snap = JobSnapshot("idle")

    def start(self, fn: Callable[[Report], Any]) -> bool:
        """Start a run. Returns False (and does nothing) if one is already running."""
        with self._lock:
            self._expire_locked()
            if self._snap.running:
                return False
            self._generation += 1
            generation = self._generation
            self._snap = JobSnapshot("running", started_at=self._clock())
            thread = threading.Thread(
                target=self._run, args=(generation, fn), name="yezatas-job", daemon=True
            )
            self._thread = thread
        thread.start()
        return True

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            self._expire_locked()
            return self._snap

    def cancel(self, message: str = "Cancelled.") -> bool:
        """Abandon the current run (its thread may keep going in the background)."""
        with self._lock:
            if not self._snap.running:
                return False
            self._generation += 1
            self._snap = JobSnapshot("failed", message, "cancelled")
            return True

    def wait(self, timeout: float = 60.0) -> None:
        """Block until the latest worker thread ends. For tests."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _expire_locked(self) -> None:
        snap = self._snap
        if (
            snap.running
            and self._timeout is not None
            and self._clock() - snap.started_at > self._timeout
        ):
            self._generation += 1
            self._snap = JobSnapshot("failed", self._timeout_message, "timeout")

    def _report(self, generation: int, message: str) -> None:
        with self._lock:
            if generation == self._generation and self._snap.running:
                self._snap = JobSnapshot("running", message, started_at=self._snap.started_at)

    def _run(self, generation: int, fn: Callable[[Report], Any]) -> None:
        try:
            result = fn(lambda message: self._report(generation, message))
            final = JobSnapshot("done", result=result)
        except BaseException as exc:  # a worker must never die silently
            kind, message = self._map_error(exc)
            final = JobSnapshot("failed", message, kind)
        with self._lock:
            if generation == self._generation:
                self._snap = final
