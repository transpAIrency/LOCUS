"""HTTP retry and local credential handling with no real API requests."""

from contextlib import closing
from concurrent.futures import CancelledError
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from hashlib import sha256
import httpx
from pathlib import Path
import sys
import tempfile
from threading import Event
import traceback
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurips2026_locus.jev import (
    EnvironmentError,
    JevClient,
    JevError,
    RetryDelayExceeded,
    retry_delay,
    get_api_key,
    load_dotenv,
)


class JevTests(unittest.TestCase):
    def test_long_retry_header_stops_instead_of_sleeping_or_retrying_early(self):
        for delay in (
            "120",
            "9" * 400,
            format_datetime(datetime.now(timezone.utc) + timedelta(hours=1)),
        ):
            handler = Mock(
                return_value=httpx.Response(429, headers={"Retry-After": delay})
            )
            with (
                self.subTest(delay=delay),
                self.client(handler) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(RetryDelayExceeded, "60-second"):
                    client.decide({})
            self.assertEqual(handler.call_count, 1)
            sleep.assert_not_called()
        self.assertEqual(retry_delay({"Retry-After": "60"}, 1), 60)
        self.assertEqual(retry_delay({"retry-after": "60"}, 1), 60)

    def test_cancellation_stops_http_retry_without_waiting_or_sending_again(self):
        stop = Event()

        def rejected(request):
            stop.set()
            return httpx.Response(429, headers={"Retry-After": "3600"})

        handler = Mock(side_effect=rejected)
        with (
            self.client(handler, stop_event=stop) as client,
            self.assertRaises(CancelledError),
        ):
            client.decide({})
        self.assertEqual(handler.call_count, 1)

    def test_context_rejection_is_distinct_from_auth_or_schema_errors(self):
        from neurips2026_locus.jev import JevHTTPError

        for status, body, row_specific in [
            (
                400,
                {"error": {"code": "max_tokens_exceeded", "message": "private text"}},
                True,
            ),
            (400, {"detail": {"code": "max_tokens_exceeded"}}, True),
            (400, {"detail": {"type": "max_tokens_exceeded"}}, True),
            (400, {"detail": "max_tokens_exceeded"}, True),
            (
                400,
                {"detail": {"code": "invalid_json", "message": "max_tokens_exceeded"}},
                False,
            ),
            (400, {"error": {"code": "invalid_json"}}, False),
            (401, {"error": "private text"}, False),
            (413, {}, True),
            (422, {"error": "private text"}, False),
        ]:
            with (
                self.subTest(status=status),
                self.client(lambda _: httpx.Response(status, json=body)) as client,
            ):
                with self.assertRaises(JevHTTPError) as caught:
                    client.decide({})
                self.assertEqual(caught.exception.row_rejection, row_specific)
                self.assertNotIn("private text", str(caught.exception))

    def client(self, handler, **kwargs):
        return closing(
            JevClient(
                "private-test-key", transport=httpx.MockTransport(handler), **kwargs
            )
        )

    def test_non_json_constants_and_duplicate_keys_are_rejected(self):
        for value in (b'{"answers":{},"answers":{}}', b'{"confidence":NaN}'):
            with self.client(lambda _: httpx.Response(200, content=value)) as client:
                with self.assertRaises(JevError):
                    client.decide({})

    def test_http_roundtrip_preserves_response_usage_and_hashes(self):
        body = {"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 12}}
        handler = Mock(return_value=httpx.Response(200, json=body))
        with self.client(handler) as client:
            result = client.decide({"state": "A rule."})
        self.assertEqual(result.body, body)
        self.assertEqual(len(result.request_sha256), 64)
        self.assertEqual(len(result.response_sha256), 64)
        request = handler.call_args.args[0]
        self.assertEqual(result.request_sha256, sha256(request.content).hexdigest())
        self.assertEqual(request.headers["Authorization"], "Bearer private-test-key")
        self.assertNotIn(b"private-test-key", request.content)

    def test_retryable_http_error_honors_delay_and_does_not_retry_auth_errors(self):
        handler = Mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json={}),
            ]
        )
        with (
            self.client(handler) as client,
            patch("neurips2026_locus.jev.time.sleep") as sleep,
        ):
            client.decide({})
            sleep.assert_called_once_with(2.0)
        handler = Mock(return_value=httpx.Response(401, text="private-test-key"))
        with self.client(handler) as client, self.assertRaises(JevError) as caught:
            client.decide({})
        self.assertEqual(handler.call_count, 1)
        self.assertNotIn("private-test-key", str(caught.exception))
        self.assertNotIn(
            "private-test-key",
            "".join(
                traceback.format_exception(
                    type(caught.exception),
                    caught.exception,
                    caught.exception.__traceback__,
                )
            ),
        )

    def test_connection_retries_are_bounded(self):
        handler = Mock(side_effect=httpx.ConnectTimeout("private-test-key"))
        with (
            self.client(handler, max_attempts=2) as client,
            patch("neurips2026_locus.jev.time.sleep"),
            self.assertRaises(JevError) as caught,
        ):
            client.decide({})
        self.assertEqual(handler.call_count, 2)
        self.assertNotIn("private-test-key", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_possible_submission_is_not_automatically_replayed(self):
        for error in (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ):
            handler = Mock(side_effect=error("private-test-key"))
            with (
                self.subTest(error=error),
                self.client(handler) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(JevError, "possible submission") as caught:
                    client.decide({})
            self.assertEqual(handler.call_count, 1)
            sleep.assert_not_called()
            self.assertNotIn("private-test-key", str(caught.exception))

    def test_rejects_malformed_keys_without_revealing_them(self):
        for key in (
            "private-test-key\nmalformed",
            "private-test-key\r",
            "private-test-key\x00",
            "non-ascii-\u20ac",
            "",
            None,
        ):
            with self.subTest(), self.assertRaises(JevError) as caught:
                JevClient(key)
            self.assertNotIn("private-test-key", str(caught.exception))

    def test_missing_custom_environment_variable_does_not_echo_its_name(self):
        # A user can accidentally pass the key instead of its variable name.
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(EnvironmentError) as caught:
                get_api_key("secret_accidentally_used_as_a_name")
        self.assertNotIn("secret_accidentally_used_as_a_name", str(caught.exception))

    def test_explicit_key_variable_never_falls_back_to_another_account(self):
        """A missing collaborator key must not bill a configured default key."""
        for value in (None, "", "   "):
            environment = {
                "TYPESAFE_API_KEY": "default-account",
                "JEV_API": "legacy-account",
            }
            if value is not None:
                environment["COLLABORATOR_KEY"] = value
            with patch.dict("os.environ", environment, clear=True):
                with self.assertRaises(EnvironmentError):
                    get_api_key("COLLABORATOR_KEY")
                self.assertEqual(get_api_key(), ("default-account", "TYPESAFE_API_KEY"))
        with patch.dict(
            "os.environ",
            {
                "COLLABORATOR_KEY": " collaborator-account ",
                "TYPESAFE_API_KEY": "default-account",
            },
            clear=True,
        ):
            self.assertEqual(
                get_api_key("COLLABORATOR_KEY"),
                ("collaborator-account", "COLLABORATOR_KEY"),
            )

    def test_rejects_redirects_and_unsafe_endpoints(self):
        handler = Mock(
            return_value=httpx.Response(
                307, headers={"Location": "https://elsewhere.invalid"}
            )
        )
        with self.client(handler) as client, self.assertRaisesRegex(JevError, "307"):
            client.decide({})
        self.assertEqual(handler.call_count, 1)
        for endpoint in (
            "http://api.typesafe.ai",
            "https://",
            "https://user:secret@host.invalid/api",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(JevError):
                JevClient("private-test-key", endpoint=endpoint)

    def test_dotenv_handles_quotes_comments_and_shell_precedence_without_expansion(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {}, clear=True),
        ):
            path = Path(directory) / ".env"
            path.write_text('JEV_API="literal-$VALUE" # a comment\n', encoding="utf-8")
            load_dotenv(path)
            self.assertEqual(get_api_key(), ("literal-$VALUE", "JEV_API"))
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "shell-value"}):
                path.write_text("TYPESAFE_API_KEY=file-value\n", encoding="utf-8")
                load_dotenv(path)
                self.assertEqual(get_api_key(), ("shell-value", "TYPESAFE_API_KEY"))
            path.write_text('JEV_API="private-test-key\n', encoding="utf-8")
            with self.assertRaises(EnvironmentError) as caught:
                load_dotenv(path)
            self.assertNotIn("private-test-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
