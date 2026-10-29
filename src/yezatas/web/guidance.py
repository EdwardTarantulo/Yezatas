"""Turn errors into plain-language guidance that points at the right setup step.

Each mapper returns (kind, message). The kind lets the setup page put the problem on the
right step; the message is safe to show (it never contains token values).
"""

from __future__ import annotations

from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

from ..auth import MissingClientSecrets
from ..catalog import NoChannelError
from ..quota import QuotaExceeded
from ..youtube import ApiErrorInfo, api_error_info

API_DISABLED_REASONS = frozenset({"accessNotConfigured", "SERVICE_DISABLED"})
QUOTA_REASONS = frozenset({"quotaExceeded", "dailyLimitExceeded"})
RATE_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})
NO_CHANNEL_REASONS = frozenset({"youtubeSignupRequired", "channelNotFound", "channelClosed"})
AUTH_REASONS = frozenset({
    "authError", "invalidCredentials", "insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT",
})

MAX_DETAIL = 300

SIGN_IN_AGAIN = "Sign in again (step 5)."
NO_CHANNEL_TEXT = (
    "This Google account has no YouTube channel. Sign in with the channel's account, or pick "
    "the brand account on Google's screen."
)
QUOTA_TEXT = (
    "Today's YouTube quota is used up. It resets at midnight Pacific time, so try again then."
)


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_DETAIL else text[: MAX_DETAIL - 1] + "…"


def _describe(exc: BaseException) -> str:
    detail = _short(str(exc))
    return f"{exc.__class__.__name__}: {detail}" if detail else exc.__class__.__name__


def classify_api_error(info: ApiErrorInfo) -> str:
    """One of: api_disabled, quota, rate, no_channel, auth, other."""
    reasons = set(info.reasons)
    lowered = info.message.lower()
    if reasons & API_DISABLED_REASONS or "has not been used in project" in lowered:
        return "api_disabled"
    if reasons & QUOTA_REASONS:
        return "quota"
    if reasons & RATE_REASONS:
        return "rate"
    if reasons & NO_CHANNEL_REASONS:
        return "no_channel"
    if info.status == 401 or reasons & AUTH_REASONS:
        return "auth"
    return "other"


def map_api_error(exc: BaseException) -> tuple[str, str]:
    """Guidance for a failed connection check or catalog pull."""
    if isinstance(exc, HttpError):
        info = api_error_info(exc)
        kind = classify_api_error(info)
        if kind == "api_disabled":
            return kind, (
                "The YouTube Data API v3 is not enabled in your Google Cloud project. "
                "Enable it (step 2), wait a minute, then try again. Make sure it is the same "
                "project your OAuth client belongs to."
            )
        if kind == "quota":
            return kind, QUOTA_TEXT
        if kind == "rate":
            return kind, "YouTube says there were too many requests. Wait a minute and try again."
        if kind == "no_channel":
            return kind, NO_CHANNEL_TEXT
        if kind == "auth":
            return kind, "Google rejected the saved sign-in. " + SIGN_IN_AGAIN
        status = f"HTTP {info.status}" if info.status else "HttpError"
        reason = f" ({info.reason})" if info.reason else ""
        return kind, _short(f"YouTube returned an error: {status}{reason}. {info.message}")
    if isinstance(exc, RefreshError):
        return "auth", (
            "Google no longer accepts the saved sign-in. In Testing mode it expires after "
            "7 days, or it may have been revoked. " + SIGN_IN_AGAIN
        )
    if isinstance(exc, NeedsSignIn):
        return "auth", "You are not signed in, or the saved sign-in is no longer valid. " + SIGN_IN_AGAIN
    if isinstance(exc, NoChannelError):
        return "no_channel", NO_CHANNEL_TEXT
    if isinstance(exc, QuotaExceeded):
        return "quota_local", f"Not enough quota left: {_short(str(exc))}. It resets at midnight Pacific time."
    return "other", _describe(exc)


def map_sign_in_error(exc: BaseException) -> tuple[str, str]:
    """Guidance for a failed Google sign-in."""
    if isinstance(exc, MissingClientSecrets):
        return "missing_secrets", "The OAuth client file is missing. Add it in step 4."
    text = f"{getattr(exc, 'error', '')} {exc.__class__.__name__} {exc}".lower()
    if "access_denied" in text or "accessdenied" in text:
        return "access_denied", (
            "Google refused the sign-in (access_denied). The usual cause: the account is not "
            "on the Test users list. Add it in step 3, then try again. If you clicked Cancel "
            "on Google's page, just start again."
        )
    if "invalid_client" in text or "invalidclient" in text:
        return "invalid_client", (
            "Google does not recognise this OAuth client. Download the JSON again from the "
            "Clients page and upload it in step 4."
        )
    return "error", _describe(exc)


class NeedsSignIn(RuntimeError):
    """There is no usable saved sign-in."""
