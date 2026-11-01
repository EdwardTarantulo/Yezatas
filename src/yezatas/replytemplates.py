"""Reply templates: saved reply texts the creator can reuse, with ``{name}`` for the commenter.

Templates are plain text saved through the store. ``check_template`` is what a caller runs before
``Store.save_reply_template``: it trims the name and body and refuses the ones that cannot be used.
``fill`` puts the commenter's name into a body. ``suggest`` ranks templates for one comment by the
words they share — pure code, no model and no network. ``save_template`` cleans a template and
refuses a name that another template already has. Nothing here talks to YouTube.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence

from .store import ReplyTemplate, Store

NAME_MAX = 60
BODY_MAX = 2000

# Words that say nothing about what a comment is about, so they never make a template match.
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "is", "it",
    "for", "on", "my", "your", "you", "i", "me", "we", "this", "that",
    "with", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "can", "could", "should",
    "so", "but", "if", "then", "than", "at", "by", "from", "as",
    "not", "no", "just", "very", "too",
})

_WORD = re.compile(r"\w+")


class TemplateError(ValueError):
    """A template was refused; the message is a plain sentence for the creator."""


def check_template(name: str, body: str) -> tuple[str, str]:
    """The cleaned (name, body) of a template, or TemplateError when it cannot be saved."""
    if not isinstance(name, str):
        raise TemplateError("Enter a template name.")
    if not isinstance(body, str):
        raise TemplateError("Enter a reply text.")
    name = name.strip()
    body = body.strip()
    if not name:
        raise TemplateError("Enter a template name.")
    if not body:
        raise TemplateError("Enter a reply text.")
    if len(name) > NAME_MAX:
        raise TemplateError(f"The template name is {len(name)} characters; the limit is {NAME_MAX}.")
    if len(body) > BODY_MAX:
        raise TemplateError(f"The reply text is {len(body)} characters; the limit is {BODY_MAX}.")
    if body == "{name}":
        raise TemplateError("The reply text must say more than just {name}.")
    return name, body


def save_template(store: Store, name: str, body: str, now: datetime | None = None,
                  template_id: int | None = None) -> int:
    """Clean a template and save it, refusing a name that another template already has."""
    cleaned_name, cleaned_body = check_template(name, body)
    for template in store.list_reply_templates():
        if template.id != template_id and template.name.casefold() == cleaned_name.casefold():
            raise TemplateError(f'There is already a template called "{cleaned_name}".')
    return store.save_reply_template(cleaned_name, cleaned_body, now=now, template_id=template_id)


def fill(body: str, author: str) -> str:
    """The body with every ``{name}`` replaced by the commenter's name ("there" when there is none).

    Only the exact placeholder ``{name}`` is replaced; any other braces stay as typed.
    """
    if not isinstance(body, str):
        return ""
    name = author.strip() if isinstance(author, str) else ""
    return body.replace("{name}", name or "there")


def _words(text: object) -> set[str]:
    """The words of a text that can make a template match: lowercased, at least three letters.

    Splitting is on non-word characters. Stop words and the ``{name}`` placeholder say nothing
    about what a comment is about, so they are left out.
    """
    if not isinstance(text, str):
        return set()
    cleaned = text.casefold().replace("{name}", " ")
    return {word for word in _WORD.findall(cleaned)
            if len(word) >= 3 and word not in STOP_WORDS}


def suggest(comment_text: str, templates: Sequence[ReplyTemplate], limit: int = 3) -> list[ReplyTemplate]:
    """The templates that share a word with the comment, best first (at most `limit`).

    A template scores the number of distinct words it shares with the comment, counted over its
    name and body together. Templates that share nothing are left out; ties keep the order the
    templates were given in. The input list is not changed.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return []
    comment_words = _words(comment_text)
    if not comment_words:
        return []
    scored: list[tuple[int, ReplyTemplate]] = []
    for template in templates or ():
        words = _words(getattr(template, "name", "")) | _words(getattr(template, "body", ""))
        score = len(comment_words & words)
        if score >= 1:
            scored.append((score, template))
    scored.sort(key=lambda pair: -pair[0])  # stable: ties keep the order given
    return [template for _, template in scored[:limit]]
