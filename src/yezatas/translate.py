"""Translations: Muse Spark translates a video's title and description, code checks that every fact
survived, and Muse Glimmer on the creator's own PC back-translates blind and compares.

The models only propose. A translation is `verified`, `flagged` (a problem or a doubt, held back
unless the creator adds it on purpose), `unverified` (the checker was not available) or `failed`.
Nothing here touches YouTube or a draft; every result is stored in the `translations` table.
API keys are only handed to the `openai` client: never logged, never in an error message or the DB.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

from . import agent, primaryai
from .config import Config, ensure_dirs
from .facts import _EMAIL, _HANDLE, _TIME, extract_urls
from .seo import _HASHTAG, _URL
from .store import Store, TranslationRecord

if TYPE_CHECKING:
    from .glossary import GlossaryTerm

# YouTube language code -> English name (the names go into the prompts).
LANGUAGES: dict[str, str] = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "de": "German", "fr": "French",
    "it": "Italian", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian", "tr": "Turkish", "ja": "Japanese", "ko": "Korean",
    "zh-Hans": "Simplified Chinese", "zh-Hant": "Traditional Chinese", "hi": "Hindi",
    "id": "Indonesian", "vi": "Vietnamese", "ar": "Arabic", "nl": "Dutch", "sv": "Swedish",
    "cs": "Czech",
}
DEFAULT_TARGETS: tuple[str, ...] = ("es", "pt", "de", "fr", "ru")

TRANSLATOR_KEY_ENV = "BASETEN_API_KEY"
MODAL_KEY_ENV = "MODAL_API_KEY"
MAX_TITLE = 100
MAX_DESCRIPTION_BYTES = 5000
PASS_SCORE = 4  # a score of 3 or less is flagged

AUTH_MESSAGE = "The translation API key was refused. Check it in Configuration."


class TranslateError(RuntimeError):
    """A translator or verifier call failed. `kind` is "auth", "rate_limit", "network", "api" or
    "answer" (the model answered, but not in the expected shape)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Translated:
    title: str
    description: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class Translator(Protocol):
    model: str

    def translate(self, source_language: str, target_language: str, title: str,
                  description: str, glossary: str = "") -> Translated: ...


@dataclass(frozen=True)
class Verdict:
    score: int  # 1-5; 5: same meaning, nothing added or dropped
    issues: tuple[str, ...]
    back_title: str  # the blind back-translation
    back_description: str
    model: str


class Verifier(Protocol):
    model: str

    def available(self) -> bool: ...

    def verify(self, source_language: str, target_language: str, source_title: str,
               source_description: str, title: str, description: str) -> Verdict: ...


def source_hash(title: str, description: str) -> str:
    """Identifies the exact source text of a translation (16 hex characters)."""
    data = json.dumps([title, description], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _name(code: str) -> str:
    return LANGUAGES.get(code, code)


def youtube_key(language: str, existing: Sequence[str]) -> str:
    """The key a translation into `language` should use next to the video's `existing` keys.

    Studio often stores regional codes ("de-DE", "pt-BR"): a new German translation replaces that
    one instead of adding a second German. Codes with a script ("zh-Hans") only match exactly.
    """
    if language in existing or "-" in language:
        return language
    regional = [key for key in existing if key.split("-", 1)[0] == language]
    return regional[0] if len(regional) == 1 else language


def same_language(a: str | None, b: str | None) -> bool:
    """True for the same code, or a code and its regional form ("en" and "en-GB")."""
    if a is None or b is None:
        return a == b
    return a == b or ("-" not in b and a.split("-", 1)[0] == b) or (
        "-" not in a and b.split("-", 1)[0] == a)


# --- the translator --------------------------------------------------------------------------------
_KEEP_RULES = (
    "Keep exactly as they are: every URL, every time stamp (like 0:00 or 12:34), every hashtag, "
    "@handle and email address, emoji, line breaks and blank lines, product and brand names, code "
    "and crypto addresses. Do not add or remove anything, and add no notes."
)


def translator_messages(source_language: str, target_language: str, title: str,
                        description: str, glossary: str = "") -> list[dict[str, str]]:
    system = (
        "You are a professional YouTube localizer. Translate the video title and description from "
        f"{_name(source_language)} into {_name(target_language)} for viewers of that language: "
        f"natural, not literal, and keep the creator's tone. {_KEEP_RULES}"
        + (f" {glossary}" if glossary else "")
        + " Answer with exactly <title>...</title> and <description>...</description> and nothing else."
    )
    user = f"<title>{title}</title>\n<description>\n{description}\n</description>"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _between(text: str, tag: str) -> str | None:
    """The text inside the last <tag>...</tag> (reasoning before it is ignored)."""
    start = text.rfind(f"<{tag}>")
    if start < 0:
        return None
    start += len(tag) + 2
    end = text.find(f"</{tag}>", start)
    return None if end < 0 else text[start:end]


def parse_translation(content: str | None) -> tuple[str, str]:
    """(title, description) from a reply with <title> and <description> tags."""
    if not content:
        raise TranslateError("answer", "The translator sent no text.")
    title = _between(content, "title")
    description = _between(content, "description")
    if title is None or description is None:
        raise TranslateError("answer", "The translator's answer had no <title> and <description>.")
    # one newline after the opening tag and one before the closing tag belong to the format
    for lead in ("\r\n", "\n"):
        if description.startswith(lead):
            description = description[len(lead):]
            break
    for tail in ("\r\n", "\n"):
        if description.endswith(tail):
            description = description[:-len(tail)]
            break
    return title.strip(), description


@dataclass(frozen=True)
class Completion:
    """One chat reply: its text (None when the model sent none) and the tokens it cost."""

    content: str | None
    input_tokens: int = 0
    output_tokens: int = 0


class ChatTranslator:
    """OpenAI-compatible chat model: Spark Open on Baseten or Modal, Spark 1.3 on Meta's API, or
    Muse Glimmer through Ollama on this computer (`local`)."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 120,
                 local: bool = False) -> None:
        import openai  # lazy: the module imports without the SDK

        self.model = model
        self.local = local
        self._openai = openai
        self._key = api_key
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _clean(self, text: str) -> str:
        text = " ".join(text.split())[:200]
        return text.replace(self._key, "***") if self._key else text

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        """One chat call: the reply's text (None when it has none) and the tokens used. Every
        failure becomes a `TranslateError` with a plain sentence and no key in it."""
        openai = self._openai
        try:
            response = self._client.chat.completions.create(model=self.model, messages=messages)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise TranslateError("auth", AUTH_MESSAGE) from exc
        except openai.RateLimitError as exc:
            raise TranslateError(
                "rate_limit", "The translation service is limiting requests. Try again in a minute."
            ) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            if self.local:
                raise TranslateError(
                    "network", "Could not reach Muse Glimmer on this computer. Start Ollama with "
                              "that model and try again.") from exc
            raise TranslateError(
                "network", "Could not reach the translation service. Check your internet "
                           "connection and try again.") from exc
        except openai.APIError as exc:
            status = getattr(exc, "status_code", None)
            detail = self._clean(str(getattr(exc, "message", "") or ""))
            text = "The translation service returned an error" + (f" (HTTP {status})" if status else "")
            raise TranslateError("api", f"{text}: {detail}" if detail else f"{text}.") from exc
        if not getattr(response, "choices", None):
            raise TranslateError("api", "The translation service returned no answer.")
        usage = getattr(response, "usage", None)
        return Completion(
            response.choices[0].message.content,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def translate(self, source_language: str, target_language: str, title: str,
                  description: str, glossary: str = "") -> Translated:
        completion = self.complete(
            translator_messages(source_language, target_language, title, description, glossary=glossary))
        new_title, new_description = parse_translation(completion.content)
        return Translated(new_title, new_description, self.model, completion.input_tokens,
                          completion.output_tokens)


# --- the verifier ----------------------------------------------------------------------------------
COMPARE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "issues"],
}


def back_messages(source_language: str, target_language: str, title: str, description: str,
                  *, insist: bool = False) -> list[dict[str, str]]:
    """The blind step: only the translation is shown, never the original.

    Plain text in tags, not JSON: with a JSON format Glimmer sometimes copied the text back
    untranslated. `insist` repeats the task in the user turn for the one retry.
    """
    source, target = _name(source_language), _name(target_language)
    system = (
        f"You are a careful translator. The user gives you a YouTube video title and description "
        f"written in {target}. Translate both into {source}, faithfully and literally. Keep URLs, "
        "time stamps, hashtags, @handles and emoji as they are. Do not add, remove or improve "
        f"anything. Answer with exactly <title>{source} title</title> and "
        f"<description>{source} description</description>."
    )
    user = f"<title>{title}</title>\n<description>\n{description}\n</description>"
    if insist:
        user = (f"Translate this {target} text into {source}. Your answer must be in {source}, "
                f"not in {target}.\n\n{user}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def compare_messages(source_language: str, title: str, description: str, back_title: str,
                     back_description: str) -> list[dict[str, str]]:
    system = (
        f"You compare two versions of a YouTube video title and description, both in "
        f"{_name(source_language)}. Version B is a back-translation of a translation of version A. "
        "Decide whether B has the same meaning as A. Ignore differences in wording and style. "
        "Score from 1 to 5: 5 means the same meaning with nothing added or dropped; 3 or less means "
        "a fact was changed, added or dropped. List each problem in issues as one short sentence "
        'naming the changed, added or dropped fact; use an empty list when there is none. Answer '
        'with JSON: {"score": 5, "issues": []}.'
    )
    user = (f"Version A\nTitle:\n{title}\n\nDescription:\n{description}\n\n"
            f"Version B\nTitle:\n{back_title}\n\nDescription:\n{back_description}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(content: Any) -> dict[str, Any]:
    """The first JSON object in a reply; anything after it (Glimmer's `<|eot|>`) is ignored."""
    if not isinstance(content, str):
        raise TranslateError("answer", "The checker sent no text.")
    text = content.strip()
    start = text.find("{")
    if start < 0:
        raise TranslateError("answer", "The checker's answer was not JSON.")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except ValueError as exc:
        raise TranslateError("answer", "The checker's answer was not valid JSON.") from exc
    if not isinstance(value, dict):
        raise TranslateError("answer", "The checker's answer was not a JSON object.")
    return value


EOT = "<|eot|>"  # Glimmer may end its text with this marker


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def copied_back(source_title: str, source_description: str, title: str, description: str,
                back_title: str, back_description: str) -> bool:
    """True when the "back-translation" is the translation itself: the checker did not translate.

    A part that the translation left unchanged (a brand name as the title, say) proves nothing.
    """
    title_copied = (bool(title.strip()) and not _same(title, source_title)
                    and _same(back_title, title))
    text_copied = (bool(description.strip()) and not _same(description, source_description)
                   and _same(back_description, description))
    return title_copied or text_copied


class OllamaVerifier:
    """Muse Glimmer through Ollama's native API on the creator's own machine (stdlib only)."""

    def __init__(self, base_url: str, model: str, timeout: float = 300,
                 opener: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # localhost: no proxy. `opener` lets tests hand in a fake.
        self._opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def available(self) -> bool:
        """True when Ollama answers and has the model."""
        try:
            with self._opener.open(f"{self.base_url}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            names = set()
            for entry in data["models"]:
                names.update(str(entry.get(key, "")) for key in ("name", "model"))
        except Exception:  # any failure means "not available"
            return False
        return self.model in names or f"{self.model}:latest" in names

    def chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> Any:
        """The reply's text (None when it has none); `schema` asks Ollama for that JSON shape."""
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0},
        }
        if schema is not None:
            payload["format"] = schema
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise TranslateError(
                "api", f"The checker (Ollama) returned an error (HTTP {exc.code}). Is the model "
                       f"{self.model} installed?") from exc
        except (OSError, TimeoutError) as exc:  # URLError, refused connections, timeouts
            raise TranslateError(
                "network", "Could not reach the checker (Ollama). Is it running?") from exc
        except ValueError as exc:
            raise TranslateError("answer", "The checker's reply was not valid JSON.") from exc
        message = data.get("message") if isinstance(data, dict) else None
        return message.get("content") if isinstance(message, dict) else None

    def _back(self, source_language: str, target_language: str, title: str, description: str,
              insist: bool) -> tuple[str, str]:
        content = self.chat(back_messages(source_language, target_language, title, description,
                                          insist=insist))
        if not isinstance(content, str):
            raise TranslateError("answer", "The checker sent no back-translation.")
        try:
            return parse_translation(content.replace(EOT, ""))
        except TranslateError as exc:
            raise TranslateError("answer", "The checker's back-translation had no <title> and "
                                           "<description>.") from exc

    def verify(self, source_language: str, target_language: str, source_title: str,
               source_description: str, title: str, description: str) -> Verdict:
        back_title, back_description = self._back(source_language, target_language, title,
                                                  description, insist=False)
        if copied_back(source_title, source_description, title, description, back_title,
                       back_description):
            back_title, back_description = self._back(source_language, target_language, title,
                                                      description, insist=True)
            if copied_back(source_title, source_description, title, description, back_title,
                           back_description):
                raise TranslateError(
                    "answer", "The checker copied the translation instead of translating it "
                              "back, so it could not check it.")
        verdict = parse_json_object(self.chat(
            compare_messages(source_language, source_title, source_description, back_title,
                             back_description),
            COMPARE_SCHEMA))
        score, issues = verdict.get("score"), verdict.get("issues")
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if (not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5
                or not isinstance(issues, list) or not all(isinstance(i, str) for i in issues)):
            raise TranslateError("answer", "The checker's score was not in the expected shape.")
        return Verdict(score, tuple(i.strip() for i in issues if i.strip()), back_title,
                       back_description, self.model)


# --- the keys and the configured clients -------------------------------------------------------------
def translator_key_source(config: Config, env: Mapping[str, str] | None = None) -> str:
    """Which translator is used, in order: "environment" or "saved" (a Baseten key), "modal" (a Modal
    key and endpoint), "local" (Primary AI is Local), "meta" (the Meta API key) or "" (none)."""
    if env is None:
        env = os.environ
    if env.get(TRANSLATOR_KEY_ENV, "").strip():
        return "environment"
    if config.translator_key_path is not None and config.translator_key_path.is_file():
        return "saved"
    if modal_ready(config, env):
        return "modal"
    if primaryai.is_local(config, env):
        return "local"
    return "meta" if agent.key_source(config, env) else ""


def _read_key(path: Any) -> str | None:
    if path is None:
        return None
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return key or None


def load_translator_key(config: Config, env: Mapping[str, str] | None = None) -> str | None:
    """The Baseten key: BASETEN_API_KEY from the environment, else the saved key file."""
    if env is None:
        env = os.environ
    key = env.get(TRANSLATOR_KEY_ENV, "").strip()
    return key or _read_key(config.translator_key_path)


def _save_key(config: Config, path: Any, prefix: str, key: str) -> None:
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("The key must be one line with no spaces.")
    if path is None:
        raise ValueError("This configuration has no place for the key.")
    ensure_dirs(config)
    fd, tmp_name = tempfile.mkstemp(dir=config.credentials_dir, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
        if sys.platform != "win32":
            os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _delete_key(path: Any) -> bool:
    if path is None:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def save_translator_key(config: Config, key: str) -> None:
    """Save the key atomically next to the other credentials."""
    _save_key(config, config.translator_key_path, "translator_api_key.", key)


def delete_translator_key(config: Config) -> bool:
    """Delete the saved key. Returns True if there was one."""
    return _delete_key(config.translator_key_path)


# --- Modal: the same open model on another host ------------------------------------------------------
def modal_key_source(config: Config, env: Mapping[str, str] | None = None) -> str:
    """Where the Modal key comes from: "environment" (MODAL_API_KEY), "saved" or "" (none)."""
    if env is None:
        env = os.environ
    if env.get(MODAL_KEY_ENV, "").strip():
        return "environment"
    if config.modal_key_path is not None and config.modal_key_path.is_file():
        return "saved"
    return ""


def load_modal_key(config: Config, env: Mapping[str, str] | None = None) -> str | None:
    """The Modal key: MODAL_API_KEY from the environment, else the saved key file."""
    if env is None:
        env = os.environ
    key = env.get(MODAL_KEY_ENV, "").strip()
    return key or _read_key(config.modal_key_path)


def modal_url(config: Config, env: Mapping[str, str] | None = None) -> str:
    """The Modal endpoint (MODAL_ENDPOINT_URL), read live so a test or a restart can change it."""
    if env is None:
        env = os.environ
    return env.get("MODAL_ENDPOINT_URL", "").strip().rstrip("/") or config.modal_url


def modal_ready(config: Config, env: Mapping[str, str] | None = None) -> bool:
    return bool(load_modal_key(config, env)) and bool(modal_url(config, env))


def save_modal_key(config: Config, key: str) -> None:
    """Save the Modal key atomically next to the other credentials."""
    _save_key(config, config.modal_key_path, "modal_api_key.", key)


def delete_modal_key(config: Config) -> bool:
    """Delete the saved Modal key. Returns True if there was one."""
    return _delete_key(config.modal_key_path)


def translator_from_config(config: Config, env: Mapping[str, str] | None = None) -> Translator | None:
    """Spark Open with the Baseten key; else Spark Open on Modal (key and endpoint); else the
    Primary AI (Glimmer through Ollama, or Spark 1.3 with the Meta key); else None."""
    key = load_translator_key(config, env)
    if key:
        return ChatTranslator(key, config.translator_model, config.translator_url)
    if modal_ready(config, env):
        return ChatTranslator(load_modal_key(config, env) or "", config.modal_model,
                              modal_url(config, env))
    found = primaryai.text_model(config, env)
    if found is None:
        return None
    return ChatTranslator(found.api_key, found.model, found.base_url,
                          timeout=primaryai.LOCAL_TIMEOUT if found.local else 120, local=found.local)


def verifier_from_config(config: Config) -> Verifier:
    return OllamaVerifier(config.verifier_url, config.verifier_model)


# --- code checks (no model) ----------------------------------------------------------------------------
def _facts(text: str) -> dict[str, Any]:
    """Links, times, hashtags, handles and emails of a text (the same finders as the fact guard)."""
    urls = set(extract_urls(text))
    rest = _URL.sub(" ", text)
    emails = set(_EMAIL.findall(rest))
    rest = _EMAIL.sub(" ", rest)
    return {
        "link": urls,
        "email": emails,
        "time": Counter(m.group(0) for m in _TIME.finditer(rest)),
        "hashtag": set(_HASHTAG.findall(rest)),
        "handle": {m.group(0).rstrip(".-") for m in _HANDLE.finditer(rest)},
    }


_KINDS = (("link", "link"), ("time", "time"), ("hashtag", "hashtag"), ("handle", "@handle"),
          ("email", "email address"))


def check_translation(source_title: str, source_description: str, title: str,
                      description: str) -> list[str]:
    """Problems found by code alone, as short sentences (empty: none). Deterministic."""
    problems: list[str] = []
    if source_title.strip() and not title.strip():
        problems.append("The translated title is empty.")
    if source_description.strip() and not description.strip():
        problems.append("The translated description is empty.")
    if len(title) > MAX_TITLE:
        problems.append(f"The translated title is {len(title)} characters; YouTube allows "
                        f"{MAX_TITLE}.")
    size = len(description.encode("utf-8"))
    if size > MAX_DESCRIPTION_BYTES:
        problems.append(f"The translated description is {size:,} bytes; YouTube allows "
                        f"{MAX_DESCRIPTION_BYTES:,}.")
    if "<" in title + description or ">" in title + description:
        problems.append("The translation contains < or >, which YouTube does not allow.")
    old = _facts(source_title + "\n" + source_description)
    new = _facts(title + "\n" + description)
    for key, label in _KINDS:
        before, after = old[key], new[key]
        if key == "time":
            missing = sorted((before - after).elements())
            added = sorted((after - before).elements())
        else:
            missing, added = sorted(before - after), sorted(after - before)
        for value in missing:
            problems.append(f'The {label} "{value}" is missing from the translation.')
        for value in added:
            problems.append(f'The translation adds the {label} "{value}".')
    if (title, description) == (source_title, source_description) and (title or description):
        problems.append("The translation is identical to the source; it was not translated.")
    return problems


# --- the run ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TranslateResult:
    records: tuple[TranslationRecord, ...]  # stored, with ids, in target order
    verifier_available: bool
    message: str  # one line, e.g. "4 verified, 1 flagged"


def _status(problems: Sequence[str], verdict: Verdict | None) -> str:
    if problems or (verdict is not None and (verdict.score < PASS_SCORE or verdict.issues)):
        return "flagged"
    return "verified" if verdict is not None else "unverified"


def _summary(records: Sequence[TranslationRecord], verifier_available: bool) -> str:
    counts = Counter(r.status for r in records)
    text = ", ".join(f"{counts[s]} {s}" for s in ("verified", "flagged", "unverified", "failed")
                     if counts[s])
    if not verifier_available and text and counts["failed"] < len(records):
        text += " (the checker was not available)"
    return text or "Nothing was translated"


def _one(translator: Translator, source_language: str, target: str, title: str,
         description: str, glossary: str = "") -> Translated | TranslateError:
    """Runs in a worker thread: translator only, no Store access. Errors are returned, not raised."""
    try:
        if glossary:
            return translator.translate(source_language, target, title, description, glossary=glossary)
        return translator.translate(source_language, target, title, description)
    except TranslateError as exc:
        return exc
    except Exception as exc:  # a bug must not lose the other languages
        return TranslateError("api", f"The translator failed unexpectedly ({type(exc).__name__}).")


def translate_video(
    store: Store,
    translator: Translator,
    verifier: Verifier | None,
    video_id: str,
    source_language: str,
    targets: Sequence[str],
    *,
    glossary: Sequence[GlossaryTerm] = (),
    title: str | None = None,
    description: str | None = None,
    now: datetime | None = None,
    report: Callable[[str], None] | None = None,
    workers: int = 4,
) -> TranslateResult:
    """Translate a video's text into each target, check every result and store it.

    The source text is `title`/`description` if given, else the draft's text, else the catalog's.
    """
    if source_language not in LANGUAGES:
        raise ValueError(f"Unknown source language {source_language!r}.")
    targets = list(targets)
    if not targets:
        raise ValueError("Pick at least one language to translate into.")
    for target in targets:
        if target not in LANGUAGES:
            raise ValueError(f"Unknown language {target!r}.")
        if target == source_language:
            raise ValueError("A video cannot be translated into its own language.")
    if len(set(targets)) != len(targets):
        raise ValueError("Each language can only be picked once.")
    video = store.get_video(video_id)
    if video is None:
        raise ValueError("This video is not in your catalog. Pull your videos first.")
    draft = store.get_draft(video_id)
    current_title, current_description = ((draft.title, draft.description) if draft is not None
                                          else (video.title, video.description))
    source_title = current_title if title is None else title
    source_description = current_description if description is None else description
    if not source_title.strip() and not source_description.strip():
        raise ValueError("There is no text to translate.")

    def say(text: str) -> None:
        if report is not None:
            report(text)

    from .glossary import glossary_problems, glossary_rules

    source_text = source_title + "\n" + source_description
    say(f"Translating {len(targets)} language" + ("" if len(targets) == 1 else "s"))
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets)))) as pool:
        futures = [
            pool.submit(
                _one,
                translator,
                source_language,
                t,
                source_title,
                source_description,
                glossary_rules(glossary, source_text, t),
            )
            for t in targets
        ]
        outcomes = [f.result() for f in futures]

    verifier_available = bool(verifier is not None and verifier.available())
    digest = source_hash(source_title, source_description)
    ids: list[int] = []
    for index, (target, outcome) in enumerate(zip(targets, outcomes), start=1):
        base = dict(video_id=video_id, language=target, source_language=source_language,
                    source_hash=digest, translator=translator.model, id=0, created_at="")
        if isinstance(outcome, TranslateError):
            record = TranslationRecord(
                **base, title="", description="", verifier="", status="failed", score=None,
                issues=[], problems=[str(outcome)], back_translation={}, input_tokens=0,
                output_tokens=0)
        else:
            problems = check_translation(source_title, source_description, outcome.title,
                                         outcome.description)
            problems = list(problems)
            problems.extend(
                glossary_problems(
                    glossary,
                    source_text,
                    outcome.title + "\n" + outcome.description,
                    target,
                )
            )
            verdict: Verdict | None = None
            note = ""  # why the checker could not check this one
            if verifier_available:
                say(f"Checking {target} ({index} of {len(targets)})")
                try:
                    assert verifier is not None
                    verdict = verifier.verify(source_language, target, source_title,
                                              source_description, outcome.title,
                                              outcome.description)
                except TranslateError as exc:
                    note = str(exc)
                    if exc.kind == "network":  # the checker went away: do not wait for the rest
                        verifier_available = False
                except Exception as exc:  # noqa: BLE001 - a checker bug must not lose paid work
                    note = f"The checker failed unexpectedly ({type(exc).__name__})."
            record = TranslationRecord(
                **base, title=outcome.title, description=outcome.description,
                verifier=verdict.model if verdict else "", status=_status(problems, verdict),
                score=verdict.score if verdict else None,
                issues=list(verdict.issues) if verdict else [], problems=problems,
                back_translation=({"title": verdict.back_title,
                                   "description": verdict.back_description} if verdict
                                  else {"note": note} if note else {}),
                input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens)
        ids.append(store.add_translation(record, now))
    records = tuple(r for r in (store.get_translation(i) for i in ids) if r is not None)
    return TranslateResult(records, verifier_available, _summary(records, verifier_available))
