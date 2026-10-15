"""
Problem reports: validate the form, build the title and body, and deliver them. No Qt in here: the Settings dialog
shows what this module builds, and the EngineBridge runs ``submit`` on a worker thread so the UI never waits for
the network.

Delivery order: an HTTPS POST to the form service in report_config.FORM_ENDPOINT. While that is still the
placeholder, or when the POST fails, the report goes out through the player's mail program (mailto:) instead.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
from typing import Callable, Dict, Optional

import report_config
from appinfo import APP_NAME, APP_VERSION

REPORT_TYPES = ("Bug", "Glitch", "Suggestion")
SUMMARY_MAX = 80
DESCRIPTION_MAX = 4000
REPLY_TO_MAX = 120
BODY_MAX = 12000       # what the service is asked to carry, diagnostics included
MAILTO_URL_MAX = 1990  # Windows hands mailto: links to the mail client through a ~2 000 character limit

_EMAIL = re.compile(r"^[^@\s,;<>()\"']+@[^@\s,;<>()\"']+\.[A-Za-z]{2,}$")


# ------------------------------------------------------------------------------------------------ the form
def valid_email(address: str) -> bool:
    address = address.strip()
    return 0 < len(address) <= REPLY_TO_MAX and bool(_EMAIL.match(address))


def validate(report_type: str, summary: str, description: str, reply_to: str) -> Dict[str, str]:
    """Field name -> message for everything wrong with the form (empty when it is fine to send)."""
    errors: Dict[str, str] = {}
    if report_type not in REPORT_TYPES:
        errors["type"] = "Pick Bug, Glitch or Suggestion."
    summary = summary.strip()
    if not summary:
        errors["summary"] = "Write a one-line summary."
    elif len(summary) > SUMMARY_MAX:
        errors["summary"] = f"Keep the summary under {SUMMARY_MAX} characters."
    description = description.strip()
    if not description:
        errors["description"] = "Describe what happened."
    elif len(description) > DESCRIPTION_MAX:
        errors["description"] = f"Keep the description under {DESCRIPTION_MAX} characters."
    reply_to = reply_to.strip()
    if reply_to and not valid_email(reply_to):
        errors["reply_to"] = "That does not look like an email address."
    return errors


def build_title(report_type: str, summary: str) -> str:
    return f"[{APP_NAME} {APP_VERSION}] {report_type}: {' '.join(summary.split())}"


def report_text(title: str, body: str) -> str:
    """The whole report as one block of text (what the dialog previews and the clipboard fallback copies)."""
    return f"Subject: {title}\n\n{body}"


def build_body(description: str, reply_to: str, diagnostics: Optional[str]) -> str:
    parts = [description.strip()]
    if reply_to.strip():
        parts.append(f"Reply to: {reply_to.strip()}")
    if diagnostics:
        parts.append("--- Diagnostics (shown to the sender before sending) ---\n" + diagnostics.strip())
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------------------------ delivery
def is_placeholder(endpoint: str) -> bool:
    """True until a real https://formspree.io/f/<id> URL is set in report_config.py."""
    return "REPLACE_WITH" in endpoint or not endpoint.startswith("https://")


def is_placeholder_address(address: str) -> bool:
    """True while MAILTO_ADDRESS is still the example one, so there is nobody to mail yet."""
    return not address or address.lower().endswith((".invalid", "@example.com", "@example.org", "@example.net"))


def mailto_url(address: str, title: str, body: str, body_limit: int = report_config.MAILTO_BODY_LIMIT) -> str:
    """
    A mailto: link with the same title and body, URL-encoded. The body is cut to ``body_limit`` characters, and
    further if the encoded link would still be too long for Windows to pass to a mail client.
    """
    note = "\n[...cut: mail links are length limited]"

    def link(text: str) -> str:
        return (
            f"mailto:{urllib.parse.quote(address, safe='@')}"
            f"?subject={urllib.parse.quote(title, safe='')}&body={urllib.parse.quote(text, safe='')}"
        )

    text = body if len(body) <= body_limit else body[: body_limit - len(note)].rstrip() + note
    url = link(text)
    while len(url) > MAILTO_URL_MAX and len(text) > 200:
        text = text[: int(len(text) * 0.85)].rstrip() + note
        url = link(text)
    return url


_NO_REDIRECT = None


def _no_redirect_class():
    """
    The handler that never follows a redirect (the report goes to the one host in the config and nowhere else). Built on
    first use: urllib.request pulls in ssl and http, which a player who never sends a report does not need in memory.
    """
    global _NO_REDIRECT
    if _NO_REDIRECT is None:
        import urllib.request

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        _NO_REDIRECT = NoRedirect
    return _NO_REDIRECT


def __getattr__(name: str):  # keeps ``reporting._NoRedirect`` available (the unit tests use it)
    if name == "_NoRedirect":
        return _no_redirect_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _default_open(request, timeout: float):
    import urllib.request

    return urllib.request.build_opener(_no_redirect_class()).open(request, timeout=timeout)


class ReportError(Exception):
    """A delivery failure with a short reason the player can read."""


def post_report(
    endpoint: str,
    title: str,
    body: str,
    reply_to: str,
    timeout: float = report_config.REQUEST_TIMEOUT_S,
    opener: Optional[Callable] = None,
) -> None:
    """HTTPS POST of the report to a form service. Raises ``ReportError`` with a short reason when it fails."""
    opener = opener or _default_open  # looked up when called, so a test can swap it
    if not endpoint.startswith("https://"):
        raise ReportError("the form address is not an https:// URL")
    fields = {"_subject": title[:200], "message": body[:BODY_MAX], "app_version": APP_VERSION}
    if reply_to.strip():
        fields["email"] = reply_to.strip()
        fields["_replyto"] = reply_to.strip()
    import urllib.request

    request = urllib.request.Request(
        endpoint,
        data=urllib.parse.urlencode(fields).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": f"{APP_NAME}/{APP_VERSION}",
        },
    )
    try:
        response = opener(request, timeout)
        status = getattr(response, "status", None) or response.getcode()
        raw = response.read(65536) if hasattr(response, "read") else b""
        close = getattr(response, "close", None)
        if close:
            close()
    except urllib.error.HTTPError as exc:
        raise ReportError(f"the service answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ReportError(f"could not reach the service ({reason})") from exc
    if not 200 <= int(status) < 300:
        raise ReportError(f"the service answered HTTP {status}")
    try:
        answer = json.loads(raw.decode("utf-8")) if raw else {}
    except (ValueError, UnicodeDecodeError):
        answer = {}
    if isinstance(answer, dict) and (answer.get("ok") is False or answer.get("success") is False or answer.get("errors")):
        raise ReportError("the service rejected the report")


class RateLimiter:
    """One report per ``interval`` seconds (per app run)."""

    def __init__(
        self,
        interval: float = report_config.MIN_SECONDS_BETWEEN_REPORTS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.interval = interval
        self._clock = clock
        self._last: Optional[float] = None

    def wait_seconds(self) -> int:
        if self._last is None:
            return 0
        return max(0, math.ceil(self.interval - (self._clock() - self._last)))

    def mark(self) -> None:
        self._last = self._clock()


def submit(
    title: str,
    body: str,
    reply_to: str,
    limiter: RateLimiter,
    endpoint: Optional[str] = None,
    address: Optional[str] = None,
    opener: Optional[Callable] = None,
) -> Dict:
    """
    Delivers one report and says how it went; call it off the UI thread. The result is a dict:
      {"status": "sent"}                                                  the form service accepted it
      {"status": "fallback", "reason": "no_key"|"failed", "detail": str, "mailto": url, "text": str, "address": str}
                                                                          open the mail link (the report is in it);
                                                                          ``text`` is the same report for the clipboard
      {"status": "rate_limited", "wait_s": n}                             too soon after the previous report
    """
    endpoint = report_config.FORM_ENDPOINT if endpoint is None else endpoint
    address = report_config.MAILTO_ADDRESS if address is None else address
    wait = limiter.wait_seconds()
    if wait:
        return {"status": "rate_limited", "wait_s": wait}
    limiter.mark()
    extra = {"mailto": mailto_url(address, title, body), "text": report_text(title, body), "address": address}
    if is_placeholder(endpoint):
        return {"status": "fallback", "reason": "no_key", "detail": "reporting is not configured yet", **extra}
    try:
        post_report(endpoint, title, body, reply_to, opener=opener)
    except ReportError as exc:
        return {"status": "fallback", "reason": "failed", "detail": str(exc), **extra}
    except Exception as exc:  # never let a surprise in the network stack lose the player's report
        return {"status": "fallback", "reason": "failed", "detail": f"unexpected error ({type(exc).__name__})", **extra}
    return {"status": "sent"}
