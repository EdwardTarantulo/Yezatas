"""Receiving one media file from a browser form, straight to disk.

The creator picks a video or audio file in the page; the browser sends it to this local server
(127.0.0.1) as a `multipart/form-data` post. The file is written in chunks to a private folder under
the Yezatas home (never held in memory, never in the system's shared temp folder), the size is
counted while it arrives so a file over the limit stops early, and the caller deletes the folder when
its job ends. Only the cleaned file name is ever shown or kept; the browser's own path is not sent.

Fields must come before the file in the form (browsers send them in page order): the caller's
`accept` runs when the file starts, so a refused post (wrong CSRF token, a missing tool) never writes
a byte.
"""

from __future__ import annotations

import re
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

MAX_UPLOAD_BYTES = 8 * 1024 ** 3
FILE_FIELD = "file"
MAX_FIELD_BYTES = 4096  # a text field is a language, a few names or a token
MAX_FIELDS = 12
STALE_SECONDS = 24 * 3600  # folders left by a crash; a job never runs this long
NAME_CHARS = 80
SLACK_BYTES = 1024 * 1024  # the form's own fields and boundaries, on top of the file

_BAD_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                       *(f"LPT{i}" for i in range(1, 10))})


class UploadError(ValueError):
    """The upload cannot be used. The sentence is written for the creator."""


@dataclass
class Received:
    fields: dict[str, str]
    folder: Path | None  # the private folder that holds the file; delete it when done
    path: Path | None  # the saved file (None: the form had no file)
    name: str  # its cleaned name

    def discard(self) -> None:
        if self.folder is not None:
            shutil.rmtree(self.folder, ignore_errors=True)


def human_size(size: int) -> str:
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.1f} GB".replace(".0 GB", " GB")
    return f"{max(1, round(size / 1024 ** 2))} MB"


def clean_file_name(raw: str) -> str:
    """A safe file name from what the browser sent: no folders, no characters a file system
    refuses, no reserved Windows names, at most NAME_CHARS characters before the extension."""
    base = raw.replace("\\", "/").rsplit("/", 1)[-1]
    base = _BAD_CHARS.sub("_", base).strip().rstrip(". ")
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = stem.strip(" .")[:NAME_CHARS].rstrip(" .") or "video"
    suffix = "." + ext.lower()[:11] if ext else ""
    if stem.split(".")[0].upper() in _RESERVED:
        stem = "_" + stem
    return stem + suffix


def sweep_stale(root: Path, now: float | None = None) -> None:
    """Delete upload folders a crash left behind (older than a day)."""
    cutoff = (time.time() if now is None else now) - STALE_SECONDS
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def _declared_length(request: Request) -> int:
    try:
        return max(0, int(request.headers.get("content-length") or 0))
    except ValueError:
        return 0


class _Collector:
    """The parser callbacks: text fields into a dict, the file into `pending` for the writer."""

    def __init__(self, root: Path, suffixes: Sequence[str], max_bytes: int,
                 accept: Callable[[Mapping[str, str]], None]) -> None:
        self.root, self.suffixes, self.max_bytes, self.accept = root, suffixes, max_bytes, accept
        self.fields: dict[str, str] = {}
        self.folder: Path | None = None
        self.path: Path | None = None
        self.name = ""
        self.size = 0
        self.accepted = False
        self.pending: list[bytes | None] = []  # chunks to write; None closes the file
        self._out: Any = None
        self._header_name = b""
        self._header_value = b""
        self._disposition = b""
        self._kind = ""  # "field", "file" or "skip" (an empty file input)
        self._field = ""
        self._buffer = bytearray()

    # --- parser callbacks ----------------------------------------------------------------
    def on_part_begin(self) -> None:
        self._header_name = self._header_value = self._disposition = b""
        self._buffer = bytearray()
        self._kind = "skip"

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._disposition = self._header_value
        self._header_name = self._header_value = b""

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        name = options.get(b"name", b"").decode("utf-8", errors="replace")
        raw_file = options.get(b"filename")
        if raw_file is None:
            if len(self.fields) >= MAX_FIELDS:
                raise UploadError("The form sent too many fields.")
            self._kind, self._field = "field", name
            return
        if not raw_file:  # no file chosen
            self._kind = "skip"
            return
        if name != FILE_FIELD or self.path is not None:
            raise UploadError("Send exactly one file.")
        self._start_file(raw_file.decode("utf-8", errors="replace"))

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._kind == "field":
            self._buffer += data[start:end]
            if len(self._buffer) > MAX_FIELD_BYTES:
                raise UploadError("The form sent a value that is too long.")
        elif self._kind == "file":
            self.size += end - start
            if self.size > self.max_bytes:
                raise UploadError(f"That file is bigger than {human_size(self.max_bytes)}, "
                                  "which is more than Yezatas takes. Nothing was sent.")
            self.pending.append(bytes(data[start:end]))

    def on_part_end(self) -> None:
        if self._kind == "field":
            self.fields[self._field] = self._buffer.decode("utf-8", errors="replace")
        elif self._kind == "file":
            self.pending.append(None)

    # --- the file ------------------------------------------------------------------------
    def _start_file(self, raw_name: str) -> None:
        self.accepted = True
        self.accept(self.fields)  # may refuse; nothing is written yet
        name = clean_file_name(raw_name)
        if Path(name).suffix not in self.suffixes:
            raise UploadError(f"{name} is not a video or audio file Yezatas can read. Use one of: "
                              f"{', '.join(self.suffixes)}. Nothing was sent.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.folder = self.root / f"upload-{secrets.token_hex(8)}"
        self.folder.mkdir()
        self.path = self.folder / name
        self.name = name
        self._out = open(self.path, "wb")
        self._kind = "file"

    def flush(self) -> None:
        """Write the chunks collected so far (runs in a worker thread)."""
        pending, self.pending = self.pending, []
        for chunk in pending:
            if chunk is None:
                self._out.close()
                self._out = None
            elif self._out is not None:
                self._out.write(chunk)

    def close(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None


async def receive_media(request: Request, root: Path, *, suffixes: Sequence[str],
                        accept: Callable[[Mapping[str, str]], None],
                        max_bytes: int | None = None) -> Received:
    """Read a `multipart/form-data` post, saving its one file under `root`.

    `accept(fields)` runs when the file starts (or at the end if the form had no file), with the
    text fields that came before it; it raises to refuse (UploadError for a sentence, or anything
    else, such as the 403 for a wrong CSRF token). Raises UploadError for an unusable upload; the
    saved file is deleted in every failing case. Returns Received (the caller deletes `folder`).
    """
    if max_bytes is None:
        max_bytes = MAX_UPLOAD_BYTES
    kind, params = parse_options_header(request.headers.get("content-type", ""))
    boundary = params.get(b"boundary")
    if kind != b"multipart/form-data" or not boundary:
        raise UploadError("Choose a video or audio file on this computer. Nothing was sent.")
    declared = _declared_length(request)
    if declared > max_bytes + SLACK_BYTES:
        raise UploadError(f"That file is bigger than {human_size(max_bytes)}, which is more than "
                          "Yezatas takes. Nothing was sent.")
    root.mkdir(parents=True, exist_ok=True)
    sweep_stale(root)
    if declared and shutil.disk_usage(root).free < declared + SLACK_BYTES:
        raise UploadError(f"There is not enough free space on this computer for a file of "
                          f"{human_size(declared)}. Nothing was sent.")
    collector = _Collector(root, suffixes, max_bytes, accept)
    parser = MultipartParser(boundary, {
        "on_part_begin": collector.on_part_begin, "on_part_data": collector.on_part_data,
        "on_part_end": collector.on_part_end, "on_header_field": collector.on_header_field,
        "on_header_value": collector.on_header_value, "on_header_end": collector.on_header_end,
        "on_headers_finished": collector.on_headers_finished})
    stream = request.stream()
    try:
        try:
            async for chunk in stream:
                parser.write(chunk)
                await run_in_threadpool(collector.flush)
            parser.finalize()
            await run_in_threadpool(collector.flush)
        except FormParserError as exc:
            raise UploadError("The upload was cut off or not readable. Try again. "
                              "Nothing was sent.") from exc
        if not collector.accepted:
            collector.accept(collector.fields)
        if collector.path is None:
            raise UploadError("Choose a video or audio file on this computer. Nothing was sent.")
    except BaseException as exc:
        collector.close()
        if collector.folder is not None:
            shutil.rmtree(collector.folder, ignore_errors=True)
        if isinstance(exc, Exception):  # not a cancelled request
            await _drain(stream, max_bytes)
        raise
    return Received(collector.fields, collector.folder, collector.path, collector.name)


async def _drain(stream: Any, limit: int) -> None:
    """Read and drop the rest of a refused upload (up to `limit` bytes), so the browser gets the
    answer instead of a broken connection."""
    total = 0
    try:
        async for chunk in stream:
            total += len(chunk)
            if total > limit:
                break
    except Exception:  # noqa: BLE001 - the client left; there is nobody to answer
        return
