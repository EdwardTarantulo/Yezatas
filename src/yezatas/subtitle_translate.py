"""Translated subtitles: Muse Spark translates the text of each transcript segment, code keeps every
time, code checks the result, and Muse Glimmer on the creator's own PC back-translates blind and
compares.

The model never sees or writes a time. A prompt carries segment numbers and text only
(`<s n="7">text</s>`), and every stored segment gets its `start_ms` / `end_ms` copied from the
transcript by code. A language is `verified`, `flagged` (a problem or a doubt), `unverified` (the
checker was not available) or `failed` (no partial subtitles are kept). The result is an SRT file;
nothing here touches YouTube. API keys never appear here: the models are handed in.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, Sequence

from .store import Store, SubtitleTranslationRecord
from .subtitles import to_srt
from .transcribe import ASR_LANGUAGES
from .translate import (
    AUTH_MESSAGE, COMPARE_SCHEMA, EOT, LANGUAGES, _KINDS, Completion, TranslateError,
    Verdict, _facts, _status, parse_json_object, same_language,
)

BATCH_SEGMENTS = 60  # a batch to the translator closes at this many segments ...
BATCH_CHARS = 6000  # ... or this many characters, whichever comes first
CHECK_SEGMENTS = 40  # segments per call to the checker

Messages = list[dict[str, str]]


class TextModel(Protocol):
    """What the translator must offer (`translate.ChatTranslator` does)."""

    model: str

    def complete(self, messages: Messages) -> Completion: ...


class Checker(Protocol):
    """What the checker must offer (`translate.OllamaVerifier` does)."""

    model: str

    def available(self) -> bool: ...

    def chat(self, messages: Messages, schema: dict[str, Any] | None = None) -> Any: ...


def language_name(code: str) -> str:
    """English name of a language code for the prompts. The transcript may use a code that only the
    speech model knows ("zh", "th"); an unknown code is used as it is."""
    for table in (LANGUAGES, ASR_LANGUAGES):
        if code in table:
            return table[code]
    base = code.split("-", 1)[0]
    return ASR_LANGUAGES.get(base, code)


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


# --- prompts ------------------------------------------------------------------------------------------
def _lines(items: Sequence[tuple[int, str]]) -> str:
    return "\n".join(f'<s n="{n}">{text}</s>' for n, text in items)


def _count_rule(items: Sequence[tuple[int, str]], language: str) -> str:
    return (f"Your answer must have exactly {len(items)} <s> tags, numbered {items[0][0]} to "
            f"{items[-1][0]}, every one translated into {language}.")


def segment_messages(source_language: str, target_language: str, items: Sequence[tuple[int, str]],
                     *, insist: bool = False) -> Messages:
    """The translator's prompt: numbered segments, no times. `insist` repeats the format rule in
    the user turn for the one retry after a bad answer."""
    source, target = language_name(source_language), language_name(target_language)
    system = (
        f"You translate the spoken subtitles of a YouTube video from {source} into {target}. "
        'Each <s n="..."> tag is one subtitle segment with fixed times, so translate each one on '
        "its own: keep its meaning inside the same segment, and never merge, split, move, add or "
        f"drop segments. Write natural spoken {target} in the creator's tone. Keep numbers, model, "
        "product and brand names, URLs, @handles, hashtags and emails exactly as they are. Answer "
        'with exactly the same <s n="..."> tags, one per segment, in the same order, and nothing '
        "else."
    )
    user = _lines(items)
    if insist:
        user = f"{_count_rule(items, target)}\n\n{user}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def back_segment_messages(source_language: str, target_language: str,
                          items: Sequence[tuple[int, str]], *, insist: bool = False) -> Messages:
    """The blind step: only the translated segments are shown, never the original."""
    source, target = language_name(source_language), language_name(target_language)
    system = (
        f"You are a careful translator. The user gives you numbered subtitle segments written in "
        f'{target}, as <s n="..."> tags. Translate each segment into {source}, faithfully and '
        "literally. Keep numbers, URLs, @handles and hashtags as they are. Do not add, remove or "
        'improve anything. Answer with exactly the same <s n="..."> tags, in the same order, and '
        "nothing else."
    )
    user = _lines(items)
    if insist:
        user = (f"Translate this {target} text into {source}. Your answer must be in {source}, not "
                f"in {target}. {_count_rule(items, source)}\n\n{user}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compare_segment_messages(source_language: str, source_items: Sequence[tuple[int, str]],
                             back_items: Sequence[tuple[int, str]]) -> Messages:
    system = (
        f"You compare two versions of the subtitles of a YouTube video, both in "
        f'{language_name(source_language)}, as numbered <s n="..."> segments. Version B is a '
        "back-translation of a translation of version A. Decide whether B has the same meaning as "
        "A. Ignore differences in wording and style. Score from 1 to 5: 5 means the same meaning "
        "with nothing added or dropped; 3 or less means a fact was changed, added or dropped. "
        "List each problem in issues as one short sentence naming the segment number and the "
        "changed, added or dropped fact; use an empty list when there is none. Answer with JSON: "
        '{"score": 5, "issues": []}.'
    )
    user = f"Version A\n{_lines(source_items)}\n\nVersion B\n{_lines(back_items)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- reading the answer -------------------------------------------------------------------------------
_SEGMENT = re.compile(r'<s n="(\d+)">(.*?)</s>', re.DOTALL)


def _numbers_text(numbers: Sequence[int]) -> str:
    shown = ", ".join(str(n) for n in numbers[:5]) + (", ..." if len(numbers) > 5 else "")
    return ("segment " if len(numbers) == 1 else "segments ") + shown


def parse_segments(content: str | None, expected: Sequence[int], who: str = "translator"
                   ) -> dict[int, str]:
    """{segment number: text} from a reply of `<s n="7">text</s>` tags, in the order of `expected`.

    If a number appears more than once the LAST one wins (a model that reasons first may quote
    tags). The text must have every expected number, no other, and no empty text.
    """
    if not content:
        raise TranslateError("answer", f"The {who} sent no text.")
    found: dict[int, str] = {}
    for match in _SEGMENT.finditer(content):
        found[int(match.group(1))] = " ".join(match.group(2).replace(EOT, "").split())
    missing = [n for n in expected if n not in found]
    if missing:
        raise TranslateError("answer", f"The {who}'s answer is missing {_numbers_text(missing)}.")
    wanted = set(expected)
    extra = sorted(n for n in found if n not in wanted)
    if extra:
        raise TranslateError(
            "answer", f"The {who}'s answer has {_numbers_text(extra)} that was not asked for.")
    empty = [n for n in expected if not found[n]]
    if empty:
        raise TranslateError("answer", f"The {who}'s answer has no text for {_numbers_text(empty)}.")
    return {n: found[n] for n in expected}


# --- translating --------------------------------------------------------------------------------------
def make_batches(items: Sequence[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Split the segments into batches of at most BATCH_SEGMENTS segments and BATCH_CHARS
    characters. A single longer segment is a batch of its own."""
    batches: list[list[tuple[int, str]]] = []
    size = 0
    for item in items:
        if not batches or len(batches[-1]) >= BATCH_SEGMENTS or size + len(item[1]) > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(item)
        size += len(item[1])
    return batches


@dataclass(frozen=True)
class _Outcome:
    texts: dict[int, str] | None  # None when the batch failed
    error: TranslateError | None
    input_tokens: int = 0
    output_tokens: int = 0


def _one(model: TextModel, source_language: str, target_language: str,
         batch: Sequence[tuple[int, str]], stop: threading.Event) -> _Outcome:
    """Runs in a worker thread: the translator only, no Store access. Errors are returned, not
    raised. A bad answer, a network error or a rate limit is tried once more; a refused key never."""
    numbers = [n for n, _ in batch]
    tokens_in = tokens_out = 0
    insist = False
    error: TranslateError | None = None
    for _attempt in range(2):
        if stop.is_set():  # another batch was refused: do not spend more calls
            return _Outcome(None, TranslateError("auth", AUTH_MESSAGE), tokens_in, tokens_out)
        try:
            completion = model.complete(
                segment_messages(source_language, target_language, batch, insist=insist))
            tokens_in += completion.input_tokens
            tokens_out += completion.output_tokens
            texts = parse_segments(completion.content, numbers)
            return _Outcome(texts, None, tokens_in, tokens_out)
        except TranslateError as exc:
            error = exc
            if exc.kind == "auth":
                stop.set()
                break
            if exc.kind not in ("answer", "network", "rate_limit"):
                break
            insist = insist or exc.kind == "answer"
        except Exception as exc:  # a bug must not lose the other languages
            error = TranslateError(
                "api", f"The translator failed unexpectedly ({type(exc).__name__}).")
            break
    return _Outcome(None, error, tokens_in, tokens_out)


# --- code checks (no model) ---------------------------------------------------------------------------
# groups of three after a space count as one number too (French, Russian, Czech ... write 1 000)
_NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d[\d.,]*\d|\d")


def _digits(number: str) -> str:
    """A number without separators (1,000 = 1.000 = 1 000 = 1000); digits of other scripts
    become 0-9."""
    return "".join(str(unicodedata.decimal(c)) for c in number if c.isdecimal())


def check_segments(numbers: Sequence[int], source_texts: Sequence[str],
                   texts: Sequence[str]) -> list[str]:
    """Problems found by code alone, as short sentences (empty: none). Deterministic.

    The three sequences go together: segment `numbers[i]` had `source_texts[i]` and was translated
    to `texts[i]`. Numbers are compared as multisets; a number only in the translation is fine
    (a spelled-out "two" may become "2").
    """
    problems: list[str] = []
    for n, source, text in zip(numbers, source_texts, texts):
        if "<" in text or ">" in text:
            problems.append(f"Segment {n} contains < or >, which YouTube does not allow.")
        spelled: dict[str, str] = {}  # first spelling of each source number, for the sentence
        for m in _NUMBER.finditer(source):
            spelled.setdefault(_digits(m.group(0)), m.group(0))
        have = Counter(_digits(m.group(0)) for m in _NUMBER.finditer(text))
        want = Counter(_digits(m.group(0)) for m in _NUMBER.finditer(source))
        for key in sorted((want - have).elements()):
            problems.append(f'The number "{spelled[key]}" in segment {n} is missing from the '
                            "translation.")
        if len(text) > 3 * len(source) + 30:
            problems.append(f"Segment {n} is much longer than the original; something may have "
                            "been added.")
    old, new = _facts("\n".join(source_texts)), _facts("\n".join(texts))
    for key, label in _KINDS:
        before_facts, after_facts = old[key], new[key]
        if key == "time":
            missing = sorted((before_facts - after_facts).elements())
            added = sorted((after_facts - before_facts).elements())
        else:
            missing, added = sorted(before_facts - after_facts), sorted(after_facts - before_facts)
        for value in missing:
            problems.append(f'The {label} "{value}" is missing from the translation.')
        for value in added:
            problems.append(f'The translation adds the {label} "{value}".')
    if (texts and all(_same(a, b) for a, b in zip(source_texts, texts))
            and any(c.isalpha() for c in "".join(source_texts))):
        problems.append("The subtitles are identical to the source; they were not translated.")
    return problems


# --- verifying ----------------------------------------------------------------------------------------
def _copied(items: Sequence[tuple[int, str]], source: dict[int, str], back: dict[int, str]) -> bool:
    """True when most of the back-translation is the translation itself: the checker did not
    translate. A segment the translation left unchanged (a name, say) proves nothing."""
    changed = [(n, text) for n, text in items if not _same(text, source[n])]
    copied = sum(1 for n, text in changed if _same(back[n], text))
    return copied * 2 > len(changed)


def _back_translate(checker: Checker, source_language: str, target_language: str,
                    items: Sequence[tuple[int, str]], source: dict[int, str]) -> dict[int, str]:
    numbers = [n for n, _ in items]
    for insist in (False, True):
        content = checker.chat(
            back_segment_messages(source_language, target_language, items, insist=insist))
        if not isinstance(content, str):
            raise TranslateError("answer", "The checker sent no back-translation.")
        back = parse_segments(content, numbers, "checker")
        if not _copied(items, source, back):
            return back
    raise TranslateError(
        "answer", "The checker copied the translation instead of translating it back, so it "
                  "could not check it.")


def _verify_chunk(checker: Checker, source_language: str, target_language: str,
                  items: Sequence[tuple[int, str]], source: dict[int, str]
                  ) -> tuple[int, list[str], dict[int, str]]:
    """(score, issues, back-translation) of one chunk of translated segments."""
    back = _back_translate(checker, source_language, target_language, items, source)
    source_items = [(n, source[n]) for n, _ in items]
    verdict = parse_json_object(checker.chat(
        compare_segment_messages(source_language, source_items, list(back.items())),
        COMPARE_SCHEMA))
    score, issues = verdict.get("score"), verdict.get("issues")
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    if (not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5
            or not isinstance(issues, list) or not all(isinstance(i, str) for i in issues)):
        raise TranslateError("answer", "The checker's score was not in the expected shape.")
    return score, [i.strip() for i in issues if i.strip()], back


# --- the run ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtitleTranslateResult:
    records: tuple[SubtitleTranslationRecord, ...]  # stored, with ids, in target order
    verifier_available: bool
    message: str  # one line, e.g. "2 verified, 1 flagged"


def _summary(records: Sequence[SubtitleTranslationRecord], verifier_available: bool) -> str:
    counts = Counter(r.status for r in records)
    text = ", ".join(f"{counts[s]} {s}" for s in ("verified", "flagged", "unverified", "failed")
                     if counts[s])
    if not verifier_available and text and counts["failed"] < len(records):
        text += " (the checker was not available)"
    return text or "Nothing was translated"


def srt_for(record: SubtitleTranslationRecord) -> str:
    """The subtitles as an SRT file's text, with the transcript's own times."""
    return to_srt(record.segments)


def translate_subtitles(
    store: Store,
    translator: TextModel,
    checker: Checker | None,
    video_id: str,
    targets: Sequence[str],
    *,
    transcript_id: int | None = None,
    now: datetime | None = None,
    report: Callable[[str], None] | None = None,
    workers: int = 4,
) -> SubtitleTranslateResult:
    """Translate a video's transcript into each target, check every result and store it.

    Uses the given transcript (it must belong to the video) or the newest one. Raises a
    `TranslateError` of kind "auth", storing nothing, when the translator's key is refused.
    """
    if transcript_id is None:
        transcript = store.latest_transcript(video_id)
        if transcript is None:
            raise ValueError("This video has no transcript yet. Transcribe it first.")
    else:
        transcript = store.get_transcript(transcript_id)
        if transcript is None or transcript.video_id != video_id:
            raise ValueError("That transcript does not belong to this video.")
    source_language = transcript.language
    if not source_language:
        raise ValueError("This transcript has no language, so it cannot be translated. Transcribe "
                         "the video again and pick the language.")
    # segment number n = 1-based position in the transcript, so numbers always match it
    texts = {n: " ".join(str(seg.get("text") or "").split())
             for n, seg in enumerate(transcript.segments, start=1)}
    items = [(n, text) for n, text in texts.items() if text]
    if not items:
        raise ValueError("The transcript has no text to translate.")
    targets = list(targets)
    if not targets:
        raise ValueError("Pick at least one language to translate into.")
    for target in targets:
        if target not in LANGUAGES:
            raise ValueError(f"Unknown language {target!r}.")
        if same_language(target, source_language):
            raise ValueError("A video cannot be translated into its own language.")
    if len(set(targets)) != len(targets):
        raise ValueError("Each language can only be picked once.")

    def say(text: str) -> None:
        if report is not None:
            report(text)

    say(f"Translating subtitles into {len(targets)} language" + ("" if len(targets) == 1 else "s"))
    batches = make_batches(items)
    stop = threading.Event()
    jobs = [(target, batch) for target in targets for batch in batches]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = [pool.submit(_one, translator, source_language, target, batch, stop)
                   for target, batch in jobs]
        outcomes = [f.result() for f in futures]
    if any(o.error is not None and o.error.kind == "auth" for o in outcomes):
        raise TranslateError("auth", AUTH_MESSAGE)  # nothing has been stored

    per_target: dict[str, list[_Outcome]] = {target: [] for target in targets}
    for (target, _batch), outcome in zip(jobs, outcomes):
        per_target[target].append(outcome)

    checker_up = bool(checker is not None and checker.available())
    verifier_available = checker_up
    ids: list[int] = []
    for index, target in enumerate(targets, start=1):
        results = per_target[target]
        tokens_in = sum(o.input_tokens for o in results)
        tokens_out = sum(o.output_tokens for o in results)
        base = dict(video_id=video_id, transcript_id=transcript.id, language=target,
                    source_language=source_language, translator=translator.model,
                    input_tokens=tokens_in, output_tokens=tokens_out, id=0, created_at="")
        failure = next((o.error for o in results if o.error is not None), None)
        if failure is not None:
            record = SubtitleTranslationRecord(
                **base, segments=[], verifier="", status="failed", score=None, issues=[],
                problems=[str(failure)], back_translation={})
        else:
            translated: dict[int, str] = {}
            for outcome in results:
                assert outcome.texts is not None
                translated.update(outcome.texts)
            numbers = [n for n, _ in items]
            problems = check_segments(numbers, [texts[n] for n in numbers],
                                      [translated[n] for n in numbers])
            verdict: Verdict | None = None
            back: dict[int, str] = {}
            note = ""  # why the checker could not check this one
            if checker_up:
                say(f"Checking {target} ({index} of {len(targets)})")
                try:
                    assert checker is not None
                    scores: list[int] = []
                    issues: list[str] = []
                    for start in range(0, len(items), CHECK_SEGMENTS):
                        chunk = [(n, translated[n]) for n, _ in items[start:start + CHECK_SEGMENTS]]
                        score, found, chunk_back = _verify_chunk(
                            checker, source_language, target, chunk, texts)
                        scores.append(score)
                        issues.extend(found)
                        back.update(chunk_back)
                    verdict = Verdict(min(scores), tuple(issues), "", "", checker.model)
                except TranslateError as exc:
                    note = str(exc)
                    if exc.kind == "network":  # the checker went away: do not wait for the rest
                        checker_up = False
                        verifier_available = False
                except Exception as exc:  # noqa: BLE001 - a checker bug must not lose paid work
                    note = f"The checker failed unexpectedly ({type(exc).__name__})."
            record = SubtitleTranslationRecord(
                **base,
                # all the transcript's segments; the times are the transcript's own
                segments=[{"start_ms": seg["start_ms"], "end_ms": seg["end_ms"],
                           "text": translated.get(n, "")}
                          for n, seg in enumerate(transcript.segments, start=1)],
                verifier=verdict.model if verdict else "", status=_status(problems, verdict),
                score=verdict.score if verdict else None,
                issues=list(verdict.issues) if verdict else [], problems=problems,
                back_translation=({"segments": [{"n": n, "text": t} for n, t in back.items()]}
                                  if verdict else {"note": note} if note else {}))
        ids.append(store.add_subtitle_translation(record, now))
    records = tuple(r for r in (store.get_subtitle_translation(i) for i in ids) if r is not None)
    return SubtitleTranslateResult(records, verifier_available, _summary(records, verifier_available))
