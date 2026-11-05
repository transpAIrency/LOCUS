"""Pooled HTTPS transport for typed Jev requests and local credential loading."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shlex
import time
from typing import Any, Mapping
from urllib.parse import urlsplit

from .contract import decode_json

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.14.0"
MAX_RETRY_DELAY_SECONDS = 60.0


class JevError(RuntimeError):
    """An API request failed or did not return a typed JSON response."""


class JevResponseError(JevError):
    """A completed HTTP response contains malformed or nonobject JSON."""


class JevConnectionError(JevError):
    """The API could not be reached before submitting a request body."""


class JevUncertainSubmissionError(JevError):
    """The response was lost after possible submission; never replay automatically."""


class RetryDelayExceeded(JevError):
    """Stop a run when the server asks for a wait beyond the local retry budget."""


class JevHTTPError(JevError):
    """Retain the HTTP status without storing a potentially sensitive body."""

    def __init__(self, status_code: int, error_code: str | None = None) -> None:
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(f"Jev request failed with HTTP status {status_code}")

    @property
    def row_rejection(self) -> bool:
        """Only size/context errors are known to depend on the excerpt."""
        return self.status_code == 413 or (
            self.status_code in {400, 422} and self.error_code == "max_tokens_exceeded"
        )


class EnvironmentError(ValueError):
    """A local environment file or API-key configuration is invalid."""


@dataclass(frozen=True)
class JevResponse:
    """A decoded provider response plus content-addressable provenance."""

    body: dict[str, Any]
    request_sha256: str
    response_sha256: str


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode the exact request bytes hashed for stage-cache lookup."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def retry_delay(
    headers: Mapping[str, str], attempt: int, *, allow_long_delay: bool = False
) -> float:
    """Honor Retry-After; bounded callers stop when the requested wait is too long."""
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                deadline = parsedate_to_datetime(retry_after)
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                delay = (deadline - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = math.nan
        if delay > MAX_RETRY_DELAY_SECONDS and not allow_long_delay:
            raise RetryDelayExceeded(
                f"Server retry delay exceeds the {MAX_RETRY_DELAY_SECONDS:g}-second local limit; resume later."
            )
        if math.isfinite(delay):
            return max(0.0, delay)
    return min(2.0 ** min(attempt - 1, 5), 30.0)


def load_dotenv(path: Path) -> None:
    """Load a narrow dotenv format without overwriting shell-provided values."""

    if not path.exists():
        return
    if not path.is_file():
        raise EnvironmentError(f"Environment-file path is not a file: {path}")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise EnvironmentError(
                f"Invalid .env assignment on line {line_number} of {path}"
            )
        try:
            parts = shlex.split(value, comments=True)
        except ValueError:
            raise EnvironmentError(
                f"Invalid .env quoting on line {line_number} of {path}"
            ) from None
        if len(parts) > 1:
            raise EnvironmentError(
                f"Unquoted whitespace in .env on line {line_number} of {path}"
            )
        value = parts[0] if parts else ""
        os.environ.setdefault(key, value)


def get_api_key(preferred_name: str | None = None) -> tuple[str, str]:
    """Use an explicit variable exclusively; otherwise accept documented aliases."""

    if preferred_name is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", preferred_name
    ):
        raise EnvironmentError(
            "API-key variable name must be a valid environment variable name"
        )
    names = (
        (preferred_name,)
        if preferred_name is not None
        else ("TYPESAFE_API_KEY", "JEV_API", "JEV_API_KEY")
    )
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            cleaned = value.strip()
            if any(
                ord(character) < 33 or ord(character) > 126 for character in cleaned
            ):
                raise EnvironmentError(
                    "The Jev API key must be printable ASCII without internal whitespace"
                )
            return cleaned, name
    if preferred_name is not None:
        raise EnvironmentError(
            "Set the environment variable named by --api-key-env locally"
        )
    raise EnvironmentError(
        "No Jev API key was found. Set TYPESAFE_API_KEY, JEV_API, or the variable "
        "named by --api-key-env locally."
    )


class JevClient:
    """One thread-safe connection pool shared by the bounded row workers.

    HTTPX is imported only when creating a live client. Offline request
    construction remains available without provider dependencies.
    ``transport`` permits complete offline HTTP tests with MockTransport.
    Close this client only after its worker pool has finished.
    """

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = 45.0,
        max_attempts: int = 5,
        transport: Any = None,
        stop_event=None,
        retry_rate_limits: bool = False,
    ) -> None:
        """Validate local settings before creating a shared HTTPS connection pool."""
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(ord(character) < 33 or ord(character) > 126 for character in api_key)
        ):
            raise JevError(
                "The Jev API key must be nonempty printable ASCII without whitespace"
            )
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise JevError(
                "Jev endpoint must be an HTTPS URL without credentials, query, or fragment"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise JevError("timeout_seconds must be finite and positive")
        if type(max_attempts) is not int or max_attempts < 1:
            raise JevError("max_attempts must be at least one")
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.stop_event = stop_event
        self.retry_rate_limits = retry_rate_limits
        try:
            import httpx
        except ImportError:
            raise JevError(
                "Install the Typesafe transport: pip install -e '.[typesafe]'"
            ) from None
        self._httpx = httpx
        self._client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "neurips2026-locus",
            },
        )

    def decide(self, payload: Mapping[str, Any]) -> JevResponse:
        """Submit one typed decision request and return decoded, hashed JSON."""

        body = canonical_json_bytes(payload)
        retryable_codes = {429, 500, 502, 503, 504, 529}
        attempt, rate_limit_attempt = 1, 0
        while attempt <= self.max_attempts:
            if self.stop_event is not None and self.stop_event.is_set():
                raise CancelledError()
            try:
                response = self._client.post(self.endpoint, content=body)
                status = response.status_code
                if status != 200:
                    rate_limited = status == 429 and self.retry_rate_limits
                    if status not in retryable_codes or (
                        attempt == self.max_attempts and not rate_limited
                    ):
                        # Keep only a documented error code, never an error
                        # message that could echo legal text or credentials.
                        error_code = None
                        try:
                            error = decode_json(response.content.decode("utf-8"))
                            if isinstance(error, dict):
                                values = []
                                # The public API uses detail; retain compatibility
                                # with older error envelopes. Messages are not codes.
                                for detail in (
                                    error,
                                    error.get("detail"),
                                    error.get("error"),
                                ):
                                    if isinstance(detail, dict):
                                        values.extend(
                                            detail.get(field)
                                            for field in (
                                                "code",
                                                "type",
                                                "detail",
                                                "error",
                                            )
                                        )
                                if "max_tokens_exceeded" in values:
                                    error_code = "max_tokens_exceeded"
                        except (ValueError, UnicodeError):
                            pass
                        raise JevHTTPError(status, error_code)
                    if self.stop_event is not None and self.stop_event.is_set():
                        raise CancelledError()
                    if rate_limited:
                        rate_limit_attempt += 1
                    delay = retry_delay(
                        response.headers,
                        rate_limit_attempt if rate_limited else attempt,
                        allow_long_delay=rate_limited,
                    )
                    if rate_limited:
                        print(
                            json.dumps(
                                {
                                    "provider": "typesafe",
                                    "phase": "rate_limited",
                                    "retry_in_seconds": delay,
                                }
                            ),
                            flush=True,
                        )
                    if self.stop_event is None:
                        time.sleep(delay)
                    else:
                        self.stop_event.wait(delay)
                    if not rate_limited:
                        attempt += 1
                    continue
                raw_response = response.content
                try:
                    decoded = decode_json(raw_response.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    raise JevResponseError("Jev returned a non-JSON response") from None
                if not isinstance(decoded, dict):
                    raise JevResponseError(
                        "Jev returned a JSON value other than an object"
                    )
                return JevResponse(
                    body=decoded,
                    request_sha256=sha256(body).hexdigest(),
                    response_sha256=sha256(raw_response).hexdigest(),
                )
            except (
                self._httpx.ConnectError,
                self._httpx.ConnectTimeout,
                self._httpx.PoolTimeout,
            ):
                # These failures happen before a request body is sent.
                if attempt == self.max_attempts:
                    raise JevConnectionError("Could not reach the Jev API") from None
                delay = min(2.0 ** (attempt - 1), 30.0)
                if self.stop_event is None:
                    time.sleep(delay)
                else:
                    self.stop_event.wait(delay)
                attempt += 1
            except self._httpx.TransportError:
                # A timeout or disconnect after sending a POST can have a paid
                # response on the server. Preserve local work and require an
                # explicit resume instead of automatically replaying it.
                raise JevUncertainSubmissionError(
                    "Jev response was interrupted after possible submission; inspect the run before resuming"
                ) from None
        raise AssertionError("retry loop must return or raise")

    def close(self) -> None:
        """Release the connection pool after in-flight calls are checkpointed."""
        self._client.close()
