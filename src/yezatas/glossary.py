"""Translation glossary: keep some terms as they are, or translate them a fixed way."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .translate import LANGUAGES

GLOSSARY_FILE = "glossary.json"
MAX_TERMS = 200
TERM_MAX = 80


class GlossaryError(ValueError):
    """The glossary could not be read; the message is a plain sentence for the creator."""


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    keep: bool
    translations: tuple[tuple[str, str], ...]


def _check_text(value: str, label: str) -> None:
    if len(value) > TERM_MAX:
        raise GlossaryError(f"{label} must be at most {TERM_MAX} characters.")
    if "\n" in value or "\r" in value:
        raise GlossaryError(f"{label} must not contain a line break.")
    if "<" in value or ">" in value:
        raise GlossaryError(f'{label} must not contain "<" or ">".')


def make_term(term: str, keep: bool, translations: Mapping[str, str]) -> GlossaryTerm:
    stripped_term = term.strip() if isinstance(term, str) else ""
    if not stripped_term:
        raise GlossaryError("A term must not be empty.")
    _check_text(stripped_term, "A term")
    if keep and translations:
        raise GlossaryError(f'The term "{stripped_term}" cannot be both kept and translated.')
    if not keep and not translations:
        raise GlossaryError(f'The term "{stripped_term}" must be kept or have a translation.')
    if keep:
        return GlossaryTerm(term=stripped_term, keep=True, translations=())
    # not keep: validate translations
    pairs: list[tuple[str, str]] = []
    for code, trans in translations.items():
        if code not in LANGUAGES:
            raise GlossaryError(f'Unknown language "{code}".')
        if not isinstance(trans, str) or not trans.strip():
            raise GlossaryError(f'The translation for "{code}" must not be empty.')
        stripped_trans = trans.strip()
        _check_text(stripped_trans, "A translation")
        pairs.append((code, stripped_trans))
    pairs.sort(key=lambda p: p[0])
    return GlossaryTerm(term=stripped_term, keep=False, translations=tuple(pairs))


def _term_label(index: int, raw: object) -> str:
    if isinstance(raw, dict):
        name = raw.get("term")
        if isinstance(name, str) and name.strip():
            return f'Glossary entry "{name.strip()}"'
    return f"Glossary entry {index}"


def load_glossary(path: Path | str) -> list[GlossaryTerm]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as error:
        raise GlossaryError(f"Glossary could not be read: {error}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise GlossaryError(f"Glossary is not valid JSON: {error.msg}.") from None
    if not isinstance(data, dict):
        raise GlossaryError("Glossary must be an object with a \"terms\" list.")
    if "terms" not in data:
        raise GlossaryError("Glossary must have a \"terms\" list.")
    raw_terms = data["terms"]
    if not isinstance(raw_terms, list):
        raise GlossaryError("Glossary \"terms\" must be a list.")
    if len(raw_terms) > MAX_TERMS:
        raise GlossaryError(f"Glossary must have at most {MAX_TERMS} terms.")
    terms: list[GlossaryTerm] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_terms, start=1):
        if not isinstance(raw, dict):
            raise GlossaryError(f"Glossary entry {index} must be an object.")
        label = _term_label(index, raw)
        term_raw = raw.get("term")
        if not isinstance(term_raw, str) or not term_raw.strip():
            raise GlossaryError(f"{label} must have a term.")
        term_str = term_raw.strip()
        _check_text(term_str, "A term")
        lowered = term_str.lower()
        if lowered in seen:
            raise GlossaryError(f'The term "{term_str}" appears more than once.')
        seen.add(lowered)
        keep_raw = raw.get("keep", False)
        translations_raw = raw.get("translations", None)
        # Determine keep/translations
        keep = bool(keep_raw) if isinstance(keep_raw, bool) else False
        # If keep is explicitly True, translations should not be present with values
        if keep:
            if isinstance(translations_raw, dict) and translations_raw:
                raise GlossaryError(f'The term "{term_str}" cannot be both kept and translated.')
            try:
                gt = make_term(term_str, True, {})
            except GlossaryError as e:
                raise GlossaryError(f"{label}: {e}") from None
            terms.append(gt)
        else:
            if translations_raw is None or not isinstance(translations_raw, dict) or not translations_raw:
                raise GlossaryError(f"{label} must be kept or have a translation.")
            # Validate translations dict entries
            for code, val in translations_raw.items():
                if code not in LANGUAGES:
                    raise GlossaryError(f"{label} has unknown language \"{code}\".")
                if not isinstance(val, str) or not val.strip():
                    raise GlossaryError(f"{label} has an empty translation for \"{code}\".")
                _check_text(val.strip(), "A translation")
            try:
                gt = make_term(term_str, False, translations_raw)
            except GlossaryError as e:
                raise GlossaryError(f"{label}: {e}") from None
            terms.append(gt)
    return terms


def save_glossary(path: Path | str, terms: Sequence[GlossaryTerm]) -> None:
    if len(terms) > MAX_TERMS:
        raise GlossaryError(f"Glossary must have at most {MAX_TERMS} terms.")
    seen: set[str] = set()
    for gt in terms:
        lowered = gt.term.lower()
        if lowered in seen:
            raise GlossaryError(f'The term "{gt.term}" appears more than once.')
        seen.add(lowered)
        # Re-validate via make_term
        trans_map = {k: v for k, v in gt.translations}
        make_term(gt.term, gt.keep, trans_map)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"terms": []}
    for gt in terms:
        entry: dict = {"term": gt.term}
        if gt.keep:
            entry["keep"] = True
        else:
            entry["translations"] = {k: v for k, v in gt.translations}
        data["terms"].append(entry)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=str(file_path.parent), prefix=".glossary.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, file_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _found(term: str, text: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.IGNORECASE))


def glossary_rules(terms: Sequence[GlossaryTerm], source_text: str, target_language: str) -> str:
    keep_terms: list[str] = []
    fixed: list[tuple[str, str]] = []
    for gt in terms:
        if not _found(gt.term, source_text):
            continue
        if gt.keep:
            keep_terms.append(gt.term)
        else:
            for code, trans in gt.translations:
                if code == target_language:
                    fixed.append((gt.term, trans))
                    break
    parts: list[str] = []
    if keep_terms:
        quoted = ", ".join(f'"{t}"' for t in keep_terms)
        parts.append(f"Keep these terms exactly as written: {quoted}.")
    if fixed:
        quoted = "; ".join(f'"{k}" -> "{v}"' for k, v in fixed)
        parts.append(f"Always translate these terms this way: {quoted}.")
    return " ".join(parts)


def glossary_problems(terms: Sequence[GlossaryTerm], source_text: str, translated_text: str,
                      target_language: str) -> list[str]:
    problems: list[str] = []
    for gt in terms:
        if not _found(gt.term, source_text):
            continue
        if gt.keep:
            if gt.term not in translated_text:
                problems.append(f'The term "{gt.term}" should stay as it is, but it is not in the translation.')
        else:
            for code, trans in gt.translations:
                if code == target_language:
                    if not _found(trans, translated_text):
                        problems.append(
                            f'The term "{gt.term}" should be translated as "{trans}", but "{trans}" is not in the translation.'
                        )
                    break
    return problems
