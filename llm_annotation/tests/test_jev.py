"""HTTP retry and local credential handling with no real API requests."""

from contextlib import closing
from concurrent.futures import CancelledError
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from hashlib import sha256
import httpx
import io
import json
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
    JevHTTPError,
    JevResponseError,
    JevConnectionError,
    JevUncertainSubmissionError,
    RetryDelayExceeded,
    retry_delay,
    get_api_key,
    load_dotenv,
)


class JevTests(unittest.TestCase):
    def test_long_retry_header_stops_instead_of_sleeping_or_retrying_early(self):
        for status in (429, 503):
            for delay in (
                "120",
                "9" * 400,
                format_datetime(datetime.now(timezone.utc) + timedelta(hours=1)),
            ):
                handler = Mock(
                    return_value=httpx.Response(status, headers={"Retry-After": delay})
                )
                with (
                    self.subTest(status=status, delay=delay),
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

    def test_unattended_rate_limits_exceed_attempt_limit_with_identical_request(self):
        raw_body = b'{"model":"jev-fixed","answers":{},"usage":{"input_tokens":12}}'
        handler = Mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "120"}),
                httpx.Response(429, headers={"Retry-After": "120"}),
                httpx.Response(429, headers={"Retry-After": "120"}),
                httpx.Response(200, content=raw_body),
            ]
        )
        payload = {"model": "jev-fixed", "state": {"excerpt": "A legal rule."}}
        with (
            self.client(handler, max_attempts=1, retry_rate_limits=True) as client,
            patch("neurips2026_locus.jev.time.sleep") as sleep,
        ):
            result = client.decide(payload)
        self.assertEqual(handler.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [120] * 3)
        requests = [call.args[0].content for call in handler.call_args_list]
        self.assertEqual(len(set(requests)), 1)
        self.assertEqual(result.request_sha256, sha256(requests[0]).hexdigest())
        self.assertEqual(result.response_sha256, sha256(raw_body).hexdigest())
        self.assertEqual(result.body["usage"], {"input_tokens": 12})

    def test_unattended_rate_limit_wait_is_cancellable_without_another_post(self):
        stop = Event()
        handler = Mock(
            return_value=httpx.Response(429, headers={"Retry-After": "3600"})
        )

        def cancel_wait(delay):
            stop.set()
            return True

        with (
            self.client(
                handler, max_attempts=1, retry_rate_limits=True, stop_event=stop
            ) as client,
            patch.object(stop, "wait", side_effect=cancel_wait) as wait,
            self.assertRaises(CancelledError),
        ):
            client.decide({})
        wait.assert_called_once_with(3600)
        self.assertEqual(handler.call_count, 1)

    def test_default_rate_limit_retries_remain_bounded(self):
        handler = Mock(return_value=httpx.Response(429))
        with (
            self.client(handler, max_attempts=1) as client,
            patch("neurips2026_locus.jev.time.sleep") as sleep,
            self.assertRaises(JevError),
        ):
            client.decide({})
        self.assertEqual(handler.call_count, 1)
        sleep.assert_not_called()

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
            (422, {"error": {"code": "max_tokens_exceeded"}}, True),
            (401, {"error": "private text"}, False),
            (402, {"error": "private text"}, False),
            (403, {"error": "private text"}, False),
            (413, {}, True),
            (422, {"error": "private text"}, False),
        ]:
            with (
                self.subTest(status=status),
                self.client(
                    lambda _: httpx.Response(status, json=body),
                    retry_transport_errors=True,
                ) as client,
            ):
                with self.assertRaises(JevHTTPError) as caught:
                    client.decide({})
                self.assertEqual(caught.exception.row_rejection, row_specific)
                self.assertFalse(caught.exception.transient)
                self.assertNotIn("private text", str(caught.exception))

    def client(self, handler, **kwargs):
        return closing(
            JevClient(
                "private-test-key", transport=httpx.MockTransport(handler), **kwargs
            )
        )

    def test_non_json_constants_and_duplicate_keys_are_rejected(self):
        for value in (b'{"answers":{},"answers":{}}', b'{"confidence":NaN}', b'[]'):
            with self.client(lambda _: httpx.Response(200, content=value)) as client:
                with self.assertRaises(JevResponseError):
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
        for retry_rate_limits in (False, True):
            handler = Mock(side_effect=httpx.ConnectTimeout("private-test-key"))
            with (
                self.subTest(retry_rate_limits=retry_rate_limits),
                self.client(
                    handler, max_attempts=2, retry_rate_limits=retry_rate_limits
                ) as client,
                patch("neurips2026_locus.jev.time.sleep"),
                self.assertRaises(JevConnectionError) as caught,
            ):
                client.decide({})
            self.assertEqual(handler.call_count, 2)
            self.assertEqual(caught.exception.transport_error, "ConnectTimeout")
            self.assertEqual(caught.exception.attempts, 2)
            self.assertNotIn("private-test-key", str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_possible_submission_is_not_automatically_replayed(self):
        for error in (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.CloseError,
            httpx.RemoteProtocolError,
        ):
            for retry_rate_limits in (False, True):
                handler = Mock(side_effect=error("private-test-key"))
                with (
                    self.subTest(error=error, retry_rate_limits=retry_rate_limits),
                    self.client(handler, retry_rate_limits=retry_rate_limits) as client,
                    patch("neurips2026_locus.jev.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(
                        JevUncertainSubmissionError, "possible submission"
                    ) as caught:
                        client.decide({})
                self.assertEqual(handler.call_count, 1)
                sleep.assert_not_called()
                self.assertNotIn("private-test-key", str(caught.exception))

    def test_opted_in_transport_retry_preserves_exact_body_and_response_digests(self):
        body = b'{"model":"jev-fixed","answers":{},"usage":{"input_tokens":12}}'
        payload = {"model": "jev-fixed", "state": {"excerpt": "SOURCE_TEXT_NOT_LOGGED"}}
        for error in (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.CloseError,
            httpx.RemoteProtocolError,
        ):
            handler = Mock(
                side_effect=[
                    error("private-test-key"), httpx.Response(200, content=body)
                ]
            )
            with (
                self.subTest(error=error.__name__),
                self.client(
                    handler, max_attempts=2, retry_transport_errors=True
                ) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
                patch("sys.stdout", new=io.StringIO()) as output,
            ):
                result = client.decide(payload)
            self.assertEqual(handler.call_count, 2)
            sleep.assert_called_once_with(1.0)
            requests = [call.args[0].content for call in handler.call_args_list]
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(result.request_sha256, sha256(requests[0]).hexdigest())
            self.assertEqual(result.response_sha256, sha256(body).hexdigest())
            event = json.loads(output.getvalue())
            self.assertEqual(event["transport_error"], error.__name__)
            self.assertEqual(event["attempt"], 1)
            self.assertEqual(event["max_attempts"], 2)
            self.assertNotIn("private-test-key", output.getvalue())
            self.assertNotIn("SOURCE_TEXT_NOT_LOGGED", output.getvalue())

    def test_opted_in_transport_exhaustion_has_bounded_safe_typed_diagnostics(self):
        handler = Mock(side_effect=httpx.ReadTimeout("private-test-key"))
        payload = {"state": "SOURCE_TEXT_NOT_LOGGED"}
        with (
            self.client(handler, max_attempts=3, retry_transport_errors=True) as client,
            patch("neurips2026_locus.jev.time.sleep") as sleep,
            patch("sys.stdout", new=io.StringIO()) as output,
            self.assertRaises(JevUncertainSubmissionError) as caught,
        ):
            client.decide(payload)
        self.assertEqual(handler.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])
        self.assertEqual(caught.exception.transport_error, "ReadTimeout")
        self.assertEqual(caught.exception.attempts, 3)
        self.assertTrue(caught.exception.__suppress_context__)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([event["attempt"] for event in events], [1, 2])
        diagnostics = output.getvalue() + "".join(
            traceback.format_exception(
                type(caught.exception), caught.exception, caught.exception.__traceback__
            )
        )
        self.assertNotIn("private-test-key", diagnostics)
        self.assertNotIn("SOURCE_TEXT_NOT_LOGGED", diagnostics)

    def test_opted_in_transport_retry_can_be_cancelled_during_backoff(self):
        stop = Event()
        handler = Mock(side_effect=httpx.ReadTimeout("private-test-key"))

        def cancel_wait(delay):
            stop.set()
            return True

        with (
            self.client(
                handler, max_attempts=5, retry_transport_errors=True, stop_event=stop
            ) as client,
            patch.object(stop, "wait", side_effect=cancel_wait) as wait,
            patch("neurips2026_locus.jev.time.sleep") as sleep,
            patch("sys.stdout", new=io.StringIO()),
            self.assertRaises(CancelledError),
        ):
            client.decide({})
        wait.assert_called_once_with(1.0)
        sleep.assert_not_called()
        self.assertEqual(handler.call_count, 1)

    def test_opted_in_recovery_does_not_retry_unknown_local_protocol_errors(self):
        for error in (httpx.LocalProtocolError, httpx.UnsupportedProtocol):
            handler = Mock(side_effect=error("private-test-key"))
            with (
                self.subTest(error=error.__name__),
                self.client(
                    handler, max_attempts=5, retry_transport_errors=True
                ) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
                self.assertRaises(JevError) as caught,
            ):
                client.decide({})
            self.assertEqual(type(caught.exception), JevError)
            self.assertEqual(handler.call_count, 1)
            sleep.assert_not_called()
            self.assertNotIn("private-test-key", str(caught.exception))

    def test_transient_http_recovery_keeps_exact_body_and_response_hashes(self):
        body = b'{"model":"jev-fixed","answers":{}}'
        for status in (408, 500, 502, 503, 504, 529):
            handler = Mock(
                side_effect=[
                    httpx.Response(status, headers={"Retry-After": "2"}),
                    httpx.Response(200, content=body),
                ]
            )
            with (
                self.subTest(status=status),
                self.client(
                    handler, max_attempts=2, retry_transport_errors=True
                ) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
            ):
                result = client.decide({"state": {"excerpt": "A legal rule."}})
            self.assertEqual(handler.call_count, 2)
            sleep.assert_called_once_with(2.0)
            requests = [call.args[0].content for call in handler.call_args_list]
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(result.request_sha256, sha256(requests[0]).hexdigest())
            self.assertEqual(result.response_sha256, sha256(body).hexdigest())

    def test_exhausted_transient_http_errors_keep_status_and_attempt_count(self):
        for status in (408, 500, 502, 503, 504, 529):
            handler = Mock(
                return_value=httpx.Response(
                    status, json={"message": "private-test-key"}
                )
            )
            with (
                self.subTest(status=status),
                self.client(
                    handler, max_attempts=2, retry_transport_errors=True
                ) as client,
                patch("neurips2026_locus.jev.time.sleep") as sleep,
                patch("sys.stdout", new=io.StringIO()) as output,
                self.assertRaises(JevHTTPError) as caught,
            ):
                client.decide({})
            self.assertEqual(handler.call_count, 2)
            sleep.assert_called_once_with(1.0)
            self.assertTrue(caught.exception.transient)
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(caught.exception.attempts, 2)
            self.assertNotIn("private-test-key", str(caught.exception))
            self.assertNotIn("private-test-key", output.getvalue())

    def test_production_long_http_retry_after_is_honored_and_cancellable(self):
        for cancel in (False, True):
            events = []
            stop = Event() if cancel else None

            def respond(request):
                events.append("post")
                if events.count("post") == 1:
                    return httpx.Response(503, headers={"Retry-After": "120"})
                return httpx.Response(200, json={})

            def wait(delay):
                events.append(("wait", delay))
                if stop is not None:
                    stop.set()
                return stop is not None

            handler = Mock(side_effect=respond)
            with (
                self.subTest(cancel=cancel),
                self.client(
                    handler,
                    max_attempts=2,
                    retry_transport_errors=True,
                    stop_event=stop,
                ) as client,
                patch("neurips2026_locus.jev.time.sleep", side_effect=wait) as sleep,
                patch("sys.stdout", new=io.StringIO()),
            ):
                if stop is None:
                    client.decide({})
                    sleep.assert_called_once_with(120.0)
                else:
                    with (
                        patch.object(stop, "wait", side_effect=wait) as stop_wait,
                        self.assertRaises(CancelledError),
                    ):
                        client.decide({})
                    stop_wait.assert_called_once_with(120.0)
                    sleep.assert_not_called()
            self.assertEqual(
                events,
                ["post", ("wait", 120.0)] + ([] if cancel else ["post"]),
            )

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
