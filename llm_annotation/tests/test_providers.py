"""Provider fixtures and mocked SDKs: these tests never submit paid requests."""

from contextlib import closing
import json
import os
from pathlib import Path
import sys
import tempfile
import warnings
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurips2026_locus import google_batch, openai_batch
from neurips2026_locus.contract import CLASSIFICATION_SCHEMA

CLASSIFICATION = {
    "primary_function": "Rules",
    "subcategory": "Zoning",
    "confidence": "High",
}


def openai_result():
    return {
        "custom_id": "row-1",
        "error": None,
        "response": {
            "status_code": 200,
            "request_id": "req-1",
            "body": {
                "model": "gpt-6-luna",
                "usage": {"input_tokens": 200, "output_tokens": 40},
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": json.dumps(CLASSIFICATION)}
                        ],
                    }
                ],
            },
        },
    }


def google_result():
    return {
        "key": "row-1",
        "response": {
            "modelVersion": "gemini-3.8-flash",
            "usageMetadata": {
                "promptTokenCount": 200,
                "candidatesTokenCount": 40,
                "thoughtsTokenCount": 12,
            },
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"thought": True, "text": "This is not the final answer."},
                            {"text": json.dumps(CLASSIFICATION)},
                        ]
                    },
                }
            ],
        },
    }


class RequestTests(unittest.TestCase):
    def test_separate_policy_and_text_preserve_original_message_boundaries(self):
        kwargs = dict(
            model="test-model",
            system_prompt="system",
            user_prompt=["policy", "legal text"],
            schema=CLASSIFICATION_SCHEMA,
        )
        openai_messages = openai_batch.build_request("row-1", **kwargs)["body"]["input"]
        self.assertEqual(
            openai_messages,
            [
                {"role": "developer", "content": "system"},
                {"role": "user", "content": "policy"},
                {"role": "user", "content": "legal text"},
            ],
        )
        google_messages = google_batch.build_request("row-1", **kwargs)["request"][
            "contents"
        ]
        self.assertEqual(
            google_messages,
            [
                {"role": "user", "parts": [{"text": "policy"}]},
                {"role": "user", "parts": [{"text": "legal text"}]},
            ],
        )

    def test_openai_strict_schema_and_luna_low_settings(self):
        request = openai_batch.build_request(
            "row-1",
            model="gpt-6-luna",
            system_prompt="system",
            user_prompt="policy and legal text",
            schema=CLASSIFICATION_SCHEMA,
            max_output_tokens=300,
        )
        self.assertEqual(request["custom_id"], "row-1")
        self.assertEqual(request["url"], "/v1/responses")
        body = request["body"]
        self.assertEqual(
            body["input"],
            [
                {"role": "developer", "content": "system"},
                {"role": "user", "content": "policy and legal text"},
            ],
        )
        self.assertEqual(body["text"]["format"]["schema"], CLASSIFICATION_SCHEMA)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["max_output_tokens"], 300)
        self.assertEqual(body["reasoning"]["effort"], "low")
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)

    def test_google_native_json_schema_without_model_in_each_request(self):
        request = google_batch.build_request(
            "row-1",
            model="gemini-3.8-flash",
            system_prompt="system",
            user_prompt="policy and legal text",
            schema=CLASSIFICATION_SCHEMA,
            max_output_tokens=300,
        )
        self.assertEqual(request["key"], "row-1")
        body = request["request"]
        self.assertNotIn("model", body)
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "system")
        self.assertEqual(
            body["contents"][0]["parts"][0]["text"], "policy and legal text"
        )
        self.assertEqual(
            body["generationConfig"],
            {
                "responseMimeType": "application/json",
                "responseJsonSchema": CLASSIFICATION_SCHEMA,
                "maxOutputTokens": 300,
                "thinkingConfig": {"thinkingLevel": "LOW"},
            },
        )


class InstalledSDKTests(unittest.TestCase):
    def test_google_real_sdk_decodes_untranslated_partial_success_and_downloads_results(
        self,
    ):
        try:
            import httpx
            from google import genai
        except ImportError:
            self.skipTest("Install the Google SDK extra for the HTTP decoding check")
        real_client = genai.Client
        requests = []

        def respond(request):
            requests.append(request)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.host, "generativelanguage.googleapis.com")
            if "/batches/" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "name": "batches/offline",
                        "done": True,
                        "metadata": {
                            "displayName": "offline",
                            "model": "models/gemini-3.8-flash",
                            "state": "BATCH_STATE_PARTIALLY_SUCCEEDED",
                            "output": {"responsesFile": "files/results"},
                        },
                    },
                )
            return httpx.Response(
                200, content=(json.dumps(google_result()) + "\n").encode()
            )

        with closing(httpx.Client(transport=httpx.MockTransport(respond))) as http:

            def construct(**kwargs):
                kwargs["http_options"] = kwargs["http_options"].model_copy(
                    update={"httpx_client": http}
                )
                return real_client(**kwargs)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "socket.socket.connect",
                    side_effect=AssertionError("No real network allowed"),
                ),
                patch.object(genai, "Client", side_effect=construct),
                closing(google_batch.GoogleBatchClient("offline-dummy-key")) as client,
                warnings.catch_warnings(),
            ):
                # SDK 1.75 warns on this raw enum but preserves the value.
                warnings.filterwarnings(
                    "ignore",
                    message="BATCH_STATE_PARTIALLY_SUCCEEDED is not a valid JobState",
                    category=UserWarning,
                )
                state = client.status("batches/offline")
                self.assertTrue(state["done"])
                self.assertTrue(state["failed"])
                self.assertEqual(state["status"], "partially_succeeded")
                self.assertEqual(state["output_files"], ["files/results"])
                self.assertEqual(
                    list(client.iter_results("files/results")), [google_result()]
                )
        self.assertEqual(len(requests), 2)

    def test_real_sdk_configuration_constructs_offline_despite_conflicting_environment(
        self,
    ):
        try:
            import openai
            from google import genai
        except ImportError:
            self.skipTest(
                "Install both provider SDK extras for this configuration check"
            )
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_BASE_URL": "https://example.invalid",
                    "GOOGLE_GENAI_USE_VERTEXAI": "true",
                },
                clear=True,
            ),
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("SDK configuration must stay offline"),
            ),
            closing(
                openai_batch.OpenAIBatchClient("offline-dummy-key")
            ) as openai_client,
            closing(
                google_batch.GoogleBatchClient("offline-dummy-key")
            ) as google_client,
        ):
            self.assertIsInstance(openai_client.client, openai.OpenAI)
            self.assertEqual(
                str(openai_client.client.base_url).rstrip("/"),
                "https://api.openai.com/v1",
            )
            self.assertIsInstance(google_client.client, genai.Client)
            self.assertFalse(google_client.client.vertexai)


class ParseTests(unittest.TestCase):
    def test_success_keeps_labels_and_reported_usage(self):
        for module, fixture in (
            (openai_batch, openai_result()),
            (google_batch, google_result()),
        ):
            with self.subTest(provider=module.__name__):
                custom_id, classification, metadata, error = module.parse_result(
                    fixture
                )
                self.assertEqual(custom_id, "row-1")
                self.assertEqual(classification, CLASSIFICATION)
                self.assertIsNone(error)
                self.assertEqual(metadata["input_tokens"], 200)
                self.assertEqual(metadata["output_tokens"], 40)
                self.assertTrue(metadata["response_model"])
        self.assertEqual(
            google_batch.parse_result(google_result())[2]["reasoning_tokens"], 12
        )

    def test_missing_identifier_raises_instead_of_matching_by_position(self):
        for module in (openai_batch, google_batch):
            with self.subTest(provider=module.__name__), self.assertRaises(ValueError):
                module.parse_result({})

    def test_provider_errors_remain_errors(self):
        for module, id_key in ((openai_batch, "custom_id"), (google_batch, "key")):
            result = module.parse_result(
                {id_key: "row-1", "error": {"code": "expired"}}
            )
            self.assertIsNone(result[1])
            self.assertIn("expired", result[3])

    def test_openai_refusal_truncation_http_and_malformed_json_are_not_labels(self):
        fixtures = []
        for content in ("not JSON", "[]", None):
            row = openai_result()
            row["response"]["body"]["output"][0]["content"][0]["text"] = content
            fixtures.append(row)
        row = openai_result()
        row["response"]["body"]["output"][0]["content"] = [
            {"type": "refusal", "refusal": "Cannot answer"}
        ]
        fixtures.append(row)
        row = openai_result()
        row["response"]["body"]["status"] = "incomplete"
        fixtures.append(row)
        row = openai_result()
        row["response"]["status_code"] = 429
        fixtures.append(row)
        for row in fixtures:
            with self.subTest(row=row):
                result = openai_batch.parse_result(row)
                self.assertIsNone(result[1])
                self.assertTrue(result[3])

    def test_google_block_truncation_and_malformed_json_are_not_labels(self):
        fixtures = []
        for finish in ("MAX_TOKENS", "SAFETY", None):
            row = google_result()
            row["response"]["candidates"][0]["finishReason"] = finish
            fixtures.append(row)
        row = google_result()
        row["response"]["promptFeedback"] = {"blockReason": "SAFETY"}
        fixtures.append(row)
        for content in ("not JSON", "[]"):
            row = google_result()
            row["response"]["candidates"][0]["content"]["parts"] = [{"text": content}]
            fixtures.append(row)
        for row in fixtures:
            with self.subTest(row=row):
                result = google_batch.parse_result(row)
                self.assertIsNone(result[1])
                self.assertTrue(result[3])

    def test_missing_usage_is_unknown_not_zero(self):
        first = openai_result()
        second = google_result()
        del first["response"]["body"]["usage"]
        del second["response"]["usageMetadata"]
        for module, row in ((openai_batch, first), (google_batch, second)):
            metadata = module.parse_result(row)[2]
            self.assertIsNone(metadata["input_tokens"])
            self.assertIsNone(metadata["output_tokens"])


class ClientTests(unittest.TestCase):
    def test_openai_sdk_has_no_automatic_retries_and_streams_result_lines(self):
        sdk = MagicMock()
        module = SimpleNamespace(OpenAI=Mock(return_value=sdk))
        with patch.dict(sys.modules, {"openai": module}):
            client = openai_batch.OpenAIBatchClient("test-key", timeout=20)
        module.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            timeout=20,
            max_retries=0,
        )
        sdk.batches.create.return_value.id = "batch-1"
        self.assertEqual(client.submit("file-1", "gpt-6-luna", "gold"), "batch-1")
        sdk.batches.create.assert_called_once_with(
            input_file_id="file-1",
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": "gold"},
        )
        response = (
            sdk.files.with_streaming_response.content.return_value.__enter__.return_value
        )
        response.iter_lines.return_value = ["", json.dumps(openai_result())]
        self.assertEqual(list(client.iter_results("result-1")), [openai_result()])
        client.close()
        sdk.close.assert_called_once()

    def test_google_sdk_one_attempt_milliseconds_and_file_download(self):
        sdk = MagicMock()
        types = SimpleNamespace(
            HttpOptions=Mock(side_effect=lambda **kw: kw),
            HttpRetryOptions=Mock(side_effect=lambda **kw: kw),
        )
        genai = SimpleNamespace(Client=Mock(return_value=sdk), types=types)
        with patch.dict(
            sys.modules, {"google": SimpleNamespace(genai=genai), "google.genai": genai}
        ):
            client = google_batch.GoogleBatchClient("test-key", timeout=20)
        genai.Client.assert_called_once_with(
            api_key="test-key",
            vertexai=False,
            http_options={
                "base_url": "https://generativelanguage.googleapis.com",
                "api_version": "v1beta",
                "timeout": 20000,
                "retry_options": {"attempts": 1},
            },
        )
        sdk.batches.create.return_value.name = "batches/1"
        self.assertEqual(
            client.submit("files/1", "gemini-3.8-flash", "gold"), "batches/1"
        )
        sdk.batches.create.assert_called_once_with(
            model="gemini-3.8-flash", src="files/1", config={"display_name": "gold"}
        )
        sdk.files.download.return_value = (
            json.dumps(google_result()) + "\n\n"
        ).encode()
        self.assertEqual(list(client.iter_results("files/result-1")), [google_result()])
        client.close()
        sdk.close.assert_called_once()

    def test_terminal_statuses_keep_partial_output_and_error_files(self):
        client = object.__new__(openai_batch.OpenAIBatchClient)
        client.client = Mock()
        client.client.batches.retrieve.return_value = SimpleNamespace(
            id="batch-1",
            status="expired",
            output_file_id="results",
            error_file_id="errors",
            input_file_id="file-input",
            metadata={"description": "gold"},
        )
        self.assertEqual(
            client.status("batch-1"),
            {
                "id": "batch-1",
                "status": "expired",
                "done": True,
                "failed": True,
                "input_file_id": "file-input",
                "model": None,
                "display_name": "gold",
                "output_files": ["results"],
                "error_files": ["errors"],
            },
        )
        client = object.__new__(google_batch.GoogleBatchClient)
        client.client = Mock()
        client.client.batches.get.return_value = SimpleNamespace(
            name="batches/1",
            state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            dest=SimpleNamespace(file_name="files/results"),
            src=None,
            model="models/gemini-3.8-flash",
            display_name="gold",
        )
        self.assertEqual(
            client.status("batches/1"),
            {
                "id": "batches/1",
                "status": "completed",
                "done": True,
                "failed": False,
                "input_file_id": None,
                "model": "models/gemini-3.8-flash",
                "display_name": "gold",
                "output_files": ["files/results"],
                "error_files": [],
            },
        )

    def test_all_provider_statuses(self):
        openai = object.__new__(openai_batch.OpenAIBatchClient)
        openai.client = Mock()
        for state in (
            "validating",
            "in_progress",
            "finalizing",
            "cancelling",
            "completed",
            "failed",
            "expired",
            "cancelled",
        ):
            openai.client.batches.retrieve.return_value = SimpleNamespace(
                id="batch-1",
                status=state,
                output_file_id=None,
                error_file_id=None,
                input_file_id="file-input",
                metadata=None,
            )
            with self.subTest(provider="openai", state=state):
                status = openai.status("batch-1")
                self.assertEqual(
                    status["done"],
                    state in {"completed", "failed", "expired", "cancelled"},
                )
                self.assertEqual(
                    status["failed"], state in {"failed", "expired", "cancelled"}
                )
                self.assertEqual(status["output_files"], [])
                self.assertEqual(status["error_files"], [])
        google = object.__new__(google_batch.GoogleBatchClient)
        google.client = Mock()
        for state in (
            "QUEUED",
            "PENDING",
            "RUNNING",
            "PAUSED",
            "UPDATING",
            "CANCELLING",
            "SUCCEEDED",
            "FAILED",
            "EXPIRED",
            "CANCELLED",
            "PARTIALLY_SUCCEEDED",
        ):
            google.client.batches.get.return_value = SimpleNamespace(
                name="batches/1",
                state=SimpleNamespace(name="JOB_STATE_" + state),
                dest=None,
                src=None,
                model="models/gemini-3.8-flash",
                display_name="gold",
            )
            with self.subTest(provider="google", state=state):
                status = google.status("batches/1")
                self.assertEqual(
                    status["done"],
                    state
                    in {
                        "SUCCEEDED",
                        "FAILED",
                        "EXPIRED",
                        "CANCELLED",
                        "PARTIALLY_SUCCEEDED",
                    },
                )
                self.assertEqual(
                    status["failed"],
                    state in {"FAILED", "EXPIRED", "CANCELLED", "PARTIALLY_SUCCEEDED"},
                )
                self.assertEqual(status["output_files"], [])
                self.assertEqual(status["error_files"], [])

    def test_upload_uses_batch_purpose_or_jsonl_mime_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            client = object.__new__(openai_batch.OpenAIBatchClient)
            client.client = Mock()
            client.client.files.create.return_value.id = "file-1"
            self.assertEqual(client.upload(path), "file-1")
            self.assertEqual(
                client.client.files.create.call_args.kwargs["purpose"], "batch"
            )
            self.assertTrue(client.client.files.create.call_args.kwargs["file"].closed)
            client = object.__new__(google_batch.GoogleBatchClient)
            client.client = Mock()
            client.client.files.upload.return_value.name = "files/1"
            self.assertEqual(client.upload(path), "files/1")
            client.client.files.upload.assert_called_once_with(
                file=str(path),
                config={"mime_type": "jsonl", "display_name": "requests.jsonl"},
            )


if __name__ == "__main__":
    unittest.main()
