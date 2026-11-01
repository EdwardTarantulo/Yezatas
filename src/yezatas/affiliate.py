"""Affiliate links: add the creator's own tracking tag to matching links in descriptions.

Only drafts are made, nothing reaches YouTube. `plan_affiliate` is a pure read and writes
nothing, so the creator can preview every change. `apply_affiliate` plans again (the catalog
may have changed since the preview) and drafts only the approved ids that are still in the
plan and not blocked. The drafts go through the normal review, approval and push steps.

No program is built in. The creator writes the rules in `affiliate.json` in the config home:
a list of {"name", "domains", "param", "value"} objects, one per program. A rule tags a link
when the link's host is one of its domains or a subdomain of one, by setting its query
parameter to the rule's value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qsl, unquote_plus, urlencode, urlsplit, urlunsplit

from .bulk import BLOCKED_HAS_DRAFT, MAX_BULK, BulkItem, BulkResult
from .edits import DraftError, save_draft
from .facts import _TRAILING
from .seo import _URL
from .store import Store
from .validate import Draft, Issue, has_errors, validate_draft


class AffiliateError(ValueError):
    """The affiliate rules could not be read; the message is a plain sentence for the creator."""


@dataclass(frozen=True)
class AffiliateRule:
    name: str
    domains: tuple[str, ...]  # lowercase, no leading "www.", no trailing dot
    param: str
    value: str


@dataclass(frozen=True)
class AffiliateChange:
    video_id: str
    description: str  # the new description
    tagged: int  # links changed
    issues: tuple[Issue, ...]  # validate_draft(new, original) issues
    blocked: str  # "" or why it can't be drafted


def _rule_label(index: int, raw: object) -> str:
    """How to name a rule in an error: its name when it has one, else its number in the file."""
    if isinstance(raw, dict):
        name = raw.get("name")
        if isinstance(name, str) and name.strip():
            return f'Affiliate rule "{name.strip()}"'
    return f"Affiliate rule {index}"


def load_rules(path: Path | str) -> list[AffiliateRule]:
    """Read the rules from `path`; a missing file means no rules."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError) as error:
        raise AffiliateError(f"Affiliate rules could not be read: {error}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise AffiliateError(f"Affiliate rules are not valid JSON: {error.msg}.") from None
    if not isinstance(data, list):
        raise AffiliateError("Affiliate rules must be a list of rules.")
    rules: list[AffiliateRule] = []
    for index, raw in enumerate(data, start=1):
        if not isinstance(raw, dict):
            raise AffiliateError(f"Affiliate rule {index} must be an object.")
        name_raw = raw.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise AffiliateError(f"Affiliate rule {index} must have a name.")
        name = name_raw.strip()
        label = _rule_label(index, raw)
        if "domains" not in raw:
            raise AffiliateError(f'{label} has no "domains": list at least one domain.')
        domains_raw = raw["domains"]
        if not isinstance(domains_raw, list):
            raise AffiliateError(f'{label} "domains" must be a list.')
        if not domains_raw:
            raise AffiliateError(f'{label} "domains" must not be empty.')
        domains: list[str] = []
        for entry in domains_raw:
            if not isinstance(entry, str) or not entry.strip():
                raise AffiliateError(f'{label} "domains" must be non-empty strings.')
            domain = entry.strip().lower()
            if domain.startswith("www."):
                domain = domain[4:]
            domain = domain.rstrip(".")
            if not domain:
                raise AffiliateError(f'{label} "domains" must be non-empty strings.')
            domains.append(domain)
        param_raw = raw.get("param")
        if not isinstance(param_raw, str) or not param_raw.strip():
            raise AffiliateError(f'{label} must have a "param" that is not empty.')
        value_raw = raw.get("value")
        if not isinstance(value_raw, str):
            raise AffiliateError(f'{label} "value" must be a string.')
        rules.append(AffiliateRule(name, tuple(domains), param_raw.strip(), value_raw))
    return rules


def save_rules(path: Path | str, rules: Sequence[AffiliateRule]) -> None:
    """Write the rules back to `path` as a JSON list, creating parent dirs."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"name": rule.name, "domains": list(rule.domains), "param": rule.param, "value": rule.value}
        for rule in rules
    ]
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _domain_key(domain: str) -> str:
    """A rule domain as compared with a host: lowercase, no leading "www.", no trailing dot."""
    key = domain.strip().lower()
    if key.startswith("www."):
        key = key[4:]
    return key.rstrip(".")


def _with_param(query: str, param: str, value: str) -> str:
    """The raw query with `param` set to `value`: replaced at its first place, else appended.

    Every other part of the query is kept exactly as the creator wrote it (no re-encoding).
    """
    pair = urlencode([(param, value)])
    result: list[str] = []
    placed = False
    for part in query.split("&") if query else []:
        if unquote_plus(part.split("=", 1)[0]) == param:
            if not placed:
                result.append(pair)
                placed = True
            # later duplicates of the param are dropped
        else:
            result.append(part)
    if not placed:
        result.append(pair)
    return "&".join(result)


def tag_url(url: str, rules: Sequence[AffiliateRule]) -> str:
    """`url` with every matching rule's parameter set; unchanged when no rule matches.

    Only http and https links are changed. A leading "www." link without a scheme is treated
    as https for matching and tagging but keeps its "www." text form.
    """
    if not url or not rules:
        return url
    bare_www = url[:4].lower() == "www."
    if bare_www:
        to_parse = "https://" + url
        given_scheme = ""
    else:
        try:
            probe = urlsplit(url)
        except ValueError:
            return url
        if probe.scheme.lower() not in ("http", "https"):
            return url
        to_parse = url
        given_scheme = url.split("://", 1)[0] if "://" in url else probe.scheme
    try:
        parts = urlsplit(to_parse)
        host = parts.hostname
    except ValueError:
        return url
    if not host:
        return url
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return url
    matching = [
        rule for rule in rules
        if any(host == _domain_key(domain) or host.endswith("." + _domain_key(domain))
               for domain in rule.domains)
    ]
    if not matching:
        return url
    query = parts.query
    for rule in matching:
        query = _with_param(query, rule.param, rule.value)
    if parse_qsl(query, keep_blank_values=True) == parse_qsl(parts.query, keep_blank_values=True):
        return url
    rebuilt = urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
    if bare_www:
        prefix = parts.scheme + "://"
        return rebuilt[len(prefix):] if rebuilt.startswith(prefix) else rebuilt
    if given_scheme and given_scheme != parts.scheme and rebuilt.startswith(parts.scheme + "://"):
        rebuilt = given_scheme + rebuilt[len(parts.scheme):]
    return rebuilt


def _tagged_description(
    description: str, rules: Sequence[AffiliateRule]
) -> tuple[str, int, frozenset[str]]:
    """The tagged description, the count changed, and the original URLs changed."""
    if not description or not rules:
        return description, 0, frozenset()
    tagged = 0
    changed: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        nonlocal tagged
        whole = match.group(0)
        core = whole.rstrip(_TRAILING)
        if not core:
            return whole
        trailing = whole[len(core):]
        new_core = tag_url(core, rules)
        if new_core == core:
            return whole
        tagged += 1
        changed.add(core)
        return new_core + trailing

    return _URL.sub(replace, description), tagged, frozenset(changed)


def plan_affiliate(store: Store, rules: Sequence[AffiliateRule], visibility: str = "public") -> list[AffiliateChange]:
    """What tagging would do to each video it touches (most viewed first). Writes nothing."""
    if visibility not in ("public", "all"):
        raise ValueError("Choose which videos to change: public only, or all.")
    plan: list[AffiliateChange] = []
    for video in store.list_videos():
        if visibility == "public" and video.privacy_status != "public":
            continue
        description, tagged, changed = _tagged_description(video.description, rules)
        if tagged == 0 or description == video.description:
            continue
        issues = tuple(validate_draft(
            Draft(video.title, description, tuple(video.tags)),
            original=Draft(video.title, video.description, tuple(video.tags)),
        ))
        tagged_messages = {
            f'The link "{url}" was in the original description but is missing now.'
            for url in changed
        }
        issues = tuple(
            issue for issue in issues
            if not (issue.code == "link_removed" and issue.message in tagged_messages)
        )
        if store.get_draft(video.video_id) is not None:
            blocked = BLOCKED_HAS_DRAFT
        elif has_errors(issues):
            blocked = " ".join(i.message for i in issues if i.severity == "error")
        else:
            blocked = ""
        plan.append(AffiliateChange(video.video_id, description, tagged, issues, blocked))
    return plan


def apply_affiliate(
    store: Store, rules: Sequence[AffiliateRule], approved_ids: Sequence[str],
    now: datetime | None = None, visibility: str = "public",
) -> BulkResult:
    """Make drafts for the approved ids that are still in a fresh plan and not blocked.

    `visibility` must be the one the preview used, or approved private videos fall out of the plan.
    """
    ids = list(dict.fromkeys(approved_ids))
    if len(ids) > MAX_BULK:
        raise ValueError(
            f"That is {len(ids)} videos; at most {MAX_BULK} can be changed at once. "
            "Choose fewer videos."
        )
    plan = {change.video_id: change for change in plan_affiliate(store, rules, visibility)}
    items: list[BulkItem] = []
    for video_id in ids:
        change = plan.get(video_id)
        if change is None:
            items.append(BulkItem(video_id, "not_in_plan",
                                  "This edit no longer applies to this video."))
            continue
        if change.blocked:
            items.append(BulkItem(video_id, "blocked", change.blocked))
            continue
        video = store.get_video(video_id)
        try:
            if video is None:
                raise DraftError("This video is not in your catalog. Pull your videos first.")
            note = f"Affiliate links: {change.tagged} {'link' if change.tagged == 1 else 'links'} tagged"
            saved = save_draft(store, video_id, video.title, change.description, video.tags,
                               source="affiliate", note=note, now=now)
        except DraftError as error:
            items.append(BulkItem(video_id, "blocked", str(error)))
            continue
        if saved.saved:
            items.append(BulkItem(video_id, "drafted", "Draft made."))
        else:
            items.append(BulkItem(video_id, "blocked", "This edit changes nothing."))
    return BulkResult(
        drafted=sum(i.status == "drafted" for i in items),
        blocked=sum(i.status == "blocked" for i in items),
        not_in_plan=sum(i.status == "not_in_plan" for i in items),
        items=tuple(items),
    )
