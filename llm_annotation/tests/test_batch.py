"""Offline integration of both batch providers and their real response parsers."""

from contextlib import ExitStack, closing
import csv
import io
import httpx
import json
from hashlib import sha256
from pathlib import Path
import sqlite3
import signal
import sys
import tempfile
import unittest
from threading import Event
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurips2026_locus import batch, pipeline


class FakeBatchService:
    """Remote service substitute; retains uploaded requests and returns JSONL."""

    def __init__(self, provider):
        self.provider = provider
        self.uploads = {}
        self.jobs = {}
        self.submissions = 0
        self.fail_submit_once = False
        self.google_file_ids = False
        self.deleted_files = []

    def upload(self, path):
        file_id = f"input-{len(self.uploads) + 1}"
        if self.google_file_ids:
            file_id = "files/" + file_id
        self.uploads[file_id] = [
            json.loads(line) for line in path.read_text().splitlines()
        ]
        return file_id

    def submit(self, file_id, model, display_name):
        self.submissions += 1
        job_id = f"job-{self.submissions}"
        results = []
        for request in reversed(self.uploads[file_id]):
            key = request["custom_id"] if self.provider == "openai" else request["key"]
            classification = {
                "primary_function": "Rules",
                "subcategory": "Other",
                "confidence": "High",
            }
            if self.provider == "openai":
                results.append(
                    {
                        "custom_id": key,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "model": model,
                                "usage": {"input_tokens": 200, "output_tokens": 40},
                                "status": "completed",
                                "output": [
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [
                                            {
                                                "type": "output_text",
                                                "text": json.dumps(classification),
                                            }
                                        ],
                                    }
                                ],
                            },
                        },
                    }
                )
            else:
                results.append(
                    {
                        "key": key,
                        "response": {
                            "modelVersion": model,
                            "usageMetadata": {
                                "promptTokenCount": 200,
                                "candidatesTokenCount": 40,
                            },
                            "candidates": [
                                {
                                    "finishReason": "STOP",
                                    "content": {
                                        "parts": [{"text": json.dumps(classification)}]
                                    },
                                }
                            ],
                        },
                    }
                )
        self.jobs[job_id] = {
            "status": {
                "id": job_id,
                "status": "completed",
                "done": True,
                "failed": False,
                "output_files": [
                    ("files/" if self.google_file_ids else "") + job_id + "-results"
                ],
                "error_files": [],
                "display_name": display_name,
                "input_file_id": file_id if self.provider == "openai" else None,
                "model": "models/" + model if self.provider == "google" else None,
            },
            "results": results,
        }
        if self.fail_submit_once:
            self.fail_submit_once = False
            raise TimeoutError("The service accepted the job but the response was lost")
        return job_id

    def status(self, job_id):
        return dict(self.jobs[job_id]["status"])

    def iter_results(self, file_id):
        job_id = file_id.removeprefix("files/").removesuffix("-results")
        yield from self.jobs[job_id]["results"]

    def delete_file(self, file_id):
        self.deleted_files.append(file_id)

    def close(self):
        pass


class BatchTests(unittest.TestCase):
    @staticmethod
    def quota_rejection(provider, *, token_queue=False):
        message = (
            "Enqueued token limit reached; wait for existing batch jobs to finish."
            if token_queue
            else "Too many requests per minute."
        )
        rejection = RuntimeError(message)
        setattr(rejection, "status_code" if provider == "openai" else "code", 429)
        rejection.body = {
            "error": {
                "code": (
                    "token_limit_exceeded" if token_queue else "rate_limit_exceeded"
                ),
                "message": message,
            }
        }
        rejection.request_id = "quota-request-fixture"
        rejection.response = httpx.Response(
            429,
            headers={"Retry-After": "60", "x-request-id": "quota-request-fixture"},
        )
        return rejection

    def test_provider_error_context_keeps_model_guidance_and_scrubs_credentials(self):
        error = RuntimeError("opaque SDK exception")
        error.status_code = 400
        error.request_id = "request-fixture"
        error.body = {
            "error": {
                "code": "invalid_model",
                "message": (
                    "Model fixture-unavailable does not support batch processing. "
                    "API key: sk-fixture-secret-value; "
                    "Authorization: Bearer fixture-bearer-secret. " + "x" * 1000
                ),
            }
        }
        details = batch.provider_error_details(error)
        self.assertEqual(details["http_status"], 400)
        self.assertEqual(details["code"], "invalid_model")
        self.assertEqual(details["request_id"], "request-fixture")
        self.assertIn("fixture-unavailable", details["message"])
        self.assertIn("does not support batch", details["message"])
        self.assertNotIn("sk-fixture-secret-value", details["message"])
        self.assertNotIn("fixture-bearer-secret", details["message"])
        self.assertLessEqual(len(details["message"]), 600)

    def test_legacy_prompt_prepared_batches_keep_their_requests_and_resume(self):
        from test_pipeline import (
            convert_fixture_to_legacy_prompt_ledger,
            prompt_ledger_snapshot,
        )

        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                self.run_action("prepare", resume=False)
                prepared = self.query(
                    "SELECT id,request_sha256 FROM batch_requests ORDER BY id"
                )
                historical, _ = convert_fixture_to_legacy_prompt_ledger(self.ledger)
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(
                    self.query(
                        "SELECT id,request_sha256 FROM batch_requests ORDER BY id"
                    ),
                    prepared,
                )
                self.assertEqual(prompt_ledger_snapshot(self.ledger)[0], historical)
                self.assertEqual(
                    self.run_action("prepare", "--output", str(self.output)), 0
                )
                for line in self.output.read_text().splitlines():
                    annotation = json.loads(line)[f"classification_{provider}"]
                    self.assertEqual(
                        annotation["prompt_version"], "oct2026_codebook_short"
                    )
                    self.assertNotIn("contract_version", annotation)
                    self.assertNotIn("provenance", annotation)
                    self.assertEqual(
                        annotation["run_details"]["request_submission"], "batch"
                    )
                    self.assertEqual(annotation["run_details"]["calls_per_record"], 1)
                    self.assertNotIn("processing_mode", annotation["run_details"])

    def test_read_retry_stops_on_excessive_server_delay(self):
        error = httpx.HTTPStatusError(
            "Rate limited",
            request=httpx.Request("GET", "https://example.invalid"),
            response=httpx.Response(429, headers={"Retry-After": "3600"}),
        )
        operation = Mock(side_effect=error)
        with (
            patch.object(batch.time, "sleep") as sleep,
            self.assertRaisesRegex(batch.JevError, "60-second"),
        ):
            batch.retry_provider_read(operation, provider="google")
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    def test_cancellation_after_upload_prevents_paid_creation_and_can_resume(self):
        for provider in ("openai", "google"):
            for slots in (1, 3):
                with (
                    self.subTest(provider=provider, slots=slots),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    service, _ = self.fixture(Path(temporary), provider, count=6)
                    self.run_action("prepare", resume=False)
                    stop = Event()
                    upload = service.upload

                    def cancel_after_upload(path):
                        file_id = upload(path)
                        stop.set()
                        return file_id

                    with patch.object(
                        service, "upload", side_effect=cancel_after_upload
                    ):
                        code = pipeline.main(
                            [
                                *self.common,
                                "--resume",
                                "--batch-action",
                                "submit",
                                "--batch-size",
                                "2",
                                "--max-inflight-batches",
                                str(slots),
                            ],
                            stop_event=stop,
                        )
                    self.assertEqual(service.submissions, 0)
                    self.assertEqual(code, 130)
                    self.assertEqual(
                        self.query(
                            "SELECT COUNT(*) FROM batch_jobs WHERE state='submitting'"
                        ),
                        [(0,)],
                    )
                    self.assertEqual(
                        self.query(
                            "SELECT COUNT(*) FROM batch_requests WHERE error IS NOT NULL"
                        ),
                        [(0,)],
                    )
                    self.assertEqual(
                        self.run_action(
                            "submit", "--batch-size", "2", "--max-inflight-batches", "3"
                        ),
                        0,
                    )
                    self.assertEqual(service.submissions, 3)

    def test_cancellation_during_creation_retains_job_id_and_stops_later_jobs(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=6)
                self.run_action("prepare", resume=False)
                stop = Event()
                submit = service.submit

                def cancel_during_creation(*args):
                    remote_id = submit(*args)
                    stop.set()
                    return remote_id

                with patch.object(
                    service, "submit", side_effect=cancel_during_creation
                ):
                    code = pipeline.main(
                        [
                            *self.common,
                            "--resume",
                            "--batch-action",
                            "submit",
                            "--batch-size",
                            "2",
                            "--max-inflight-batches",
                            "3",
                        ],
                        stop_event=stop,
                    )
                self.assertEqual(service.submissions, 1)
                self.assertEqual(code, 130)
                self.assertEqual(
                    self.query("SELECT state,remote_id FROM batch_jobs"),
                    [("submitted", "job-1")],
                )

    def test_safe_read_retries_wrapped_connections_but_not_invalid_data(self):
        error = RuntimeError("private-test-key")
        error.__cause__ = httpx.ConnectError("private-test-key")
        operation = Mock(side_effect=[error, {"done": False}])
        with (
            patch("sys.stdout", new=io.StringIO()) as output,
            patch.object(batch.time, "sleep") as sleep,
        ):
            self.assertEqual(
                batch.retry_provider_read(operation, provider="openai"), {"done": False}
            )
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertNotIn("private-test-key", output.getvalue())
        operation = Mock(side_effect=ValueError("Invalid result data"))
        with patch.object(batch.time, "sleep") as sleep, self.assertRaises(ValueError):
            batch.retry_provider_read(operation, provider="google")
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()

    def test_invalid_stream_is_explicitly_closed_before_collection_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "openai")
            self.run_action("submit", resume=False)
            closed = []

            def invalid_stream():
                try:
                    yield {"custom_id": "unknown-id", "error": {"code": "fixture"}}
                finally:
                    closed.append(True)

            stream = invalid_stream()
            self.addCleanup(stream.close)
            with patch.object(service, "iter_results", return_value=stream):
                with self.assertRaisesRegex(
                    pipeline.InputError, "unknown or duplicate"
                ):
                    self.run_action("collect")
            # The test retains the generator, so garbage collection cannot close it.
            self.assertEqual(closed, [True])
            self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(0,)])

    def test_unknown_provider_never_silently_selects_google(self):
        for provider in ("typesafe", "jev", "typo"):
            with (
                self.subTest(provider=provider),
                self.assertRaises(pipeline.InputError),
            ):
                batch.adapter(provider)
            with self.assertRaises(pipeline.InputError):
                batch.api_key(provider, None)

    def test_missing_custom_key_variable_does_not_echo_an_accidental_secret(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaises(pipeline.InputError) as caught,
        ):
            batch.api_key("openai", "secret_accidentally_used_as_a_name")
        self.assertNotIn("secret_accidentally_used_as_a_name", str(caught.exception))

    def test_openai_defaults_and_shared_three_field_schema(self):
        args = pipeline.build_parser().parse_args(
            ["--provider", "openai", "--input", "unused.jsonl"]
        )
        config = batch.configuration(args, hash_input=False)
        self.assertEqual(config["model"], "gpt-6-luna")
        body = batch.request_for(
            {"content": "A license is required."}, "row-1", config
        )["body"]
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["max_output_tokens"], 4096)
        self.assertNotIn("temperature", body)
        self.assertEqual(
            set(body["text"]["format"]["schema"]["properties"]),
            {"primary_function", "subcategory", "confidence"},
        )

    def fixture(self, directory, provider, count=2):
        self.input = directory / "input.jsonl"
        self.ledger = directory / "ledger.sqlite3"
        self.output = directory / "output.jsonl"
        records = [
            {
                "audit_id": f"row-{i}",
                "content": f"Board {i} must hold a public hearing before issuing permits.",
                "classification_human": {"secret_gold_label": "NOT MODEL INPUT"},
            }
            for i in range(count)
        ]
        self.input.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        env = directory / ".env"
        env.write_text("", encoding="utf-8")
        self.common = [
            "--provider",
            provider,
            "--input",
            str(self.input),
            "--ledger",
            str(self.ledger),
            "--env-file",
            str(env),
            "--max-records",
            str(count),
        ]
        service = FakeBatchService(provider)
        constructor = Mock(return_value=service)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "test-only", "GEMINI_API_KEY": "test-only"},
                clear=True,
            )
        )
        target = "OpenAIBatchClient" if provider == "openai" else "GoogleBatchClient"
        stack.enter_context(patch.object(batch.adapter(provider), target, constructor))
        stack.enter_context(patch("sys.stdout", new=io.StringIO()))
        return service, constructor

    def run_action(self, action, *extra, resume=True):
        argv = [*self.common, "--batch-action", action, *extra]
        if resume:
            argv.append("--resume")
        return batch.run(pipeline.build_parser().parse_args(argv))

    def query(self, statement):
        with closing(sqlite3.connect(self.ledger)) as db, db:
            return db.execute(statement).fetchall()

    def test_both_providers_out_of_order_and_export(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, constructor = self.fixture(Path(temporary), provider)
                self.assertEqual(self.run_action("prepare", resume=False), 0)
                constructor.assert_not_called()
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(service.submissions, 1)
                with patch.object(
                    batch,
                    "load_dotenv",
                    side_effect=AssertionError("export must not load secrets"),
                ):
                    args = pipeline.build_parser().parse_args(
                        [*self.common, "--export-only", "--output", str(self.output)]
                    )
                    self.assertEqual(batch.run(args), 0)
                rows = [
                    json.loads(line) for line in self.output.read_text().splitlines()
                ]
                self.assertEqual([row["audit_id"] for row in rows], ["row-0", "row-1"])
                for row in rows:
                    self.assertEqual(
                        set(row), {"audit_id", f"classification_{provider}"}
                    )
                    annotation = row[f"classification_{provider}"]
                    self.assertEqual(
                        annotation["final"],
                        {
                            "primary_function": "Rules",
                            "subcategory": "Other",
                            "confidence": "High",
                        },
                    )
                    self.assertNotIn("review", annotation)
                    self.assertEqual(annotation["run_details"]["provider"], provider)
                    self.assertNotIn("provenance", annotation)
                    self.assertEqual(
                        annotation["run_details"]["request_submission"], "batch"
                    )
                    self.assertEqual(annotation["run_details"]["calls_per_record"], 1)
                    self.assertNotIn("processing_mode", annotation["run_details"])
                self.assertNotIn("NOT MODEL INPUT", json.dumps(service.uploads))

    def test_missing_and_error_results_are_retained_not_fabricated(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=3)
                self.run_action("submit", resume=False)
                with closing(sqlite3.connect(self.ledger)) as db:
                    self.assertEqual(batch.summary(db)["retry_passes"], 0)
                results = service.jobs["job-1"]["results"]
                results.pop()
                key = "custom_id" if provider == "openai" else "key"
                results[0] = {
                    key: results[0][key],
                    "error": {"code": "failed", "message": "fixture error"},
                }
                self.assertEqual(self.run_action("collect"), batch.ROW_ERRORS_EXIT_CODE)
                self.assertEqual(
                    self.query(
                        "SELECT COUNT(*) FROM batch_requests WHERE error IS NOT NULL"
                    ),
                    [(2,)],
                )
                self.assertEqual(
                    self.query(
                        "SELECT status,COUNT(*) FROM annotations GROUP BY status "
                        "ORDER BY status"
                    ),
                    [("annotated", 1), ("failed", 2)],
                )
                self.assertEqual(
                    self.run_action("prepare", "--output", str(self.output)),
                    batch.ROW_ERRORS_EXIT_CODE,
                )
                self.assertFalse(self.output.exists())
                with self.assertRaisesRegex(pipeline.InputError, "failed"):
                    self.run_action(
                        "prepare", "--output", str(self.output), "--export-only"
                    )
                with patch("sys.stdout", new=io.StringIO()) as output:
                    self.run_action(
                        "prepare",
                        "--output",
                        str(self.output),
                        "--allow-partial-export",
                    )
                self.assertEqual(len(self.output.read_text().splitlines()), 1)
                exported = json.loads(output.getvalue().splitlines()[-1])["export"]
                self.assertEqual(exported["failed"], 2)
                self.assertEqual(exported["missing"], 0)

    def test_interrupted_result_stream_retries_after_rollback_without_new_model_work(
        self,
    ):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=3)
                self.run_action("submit", resume=False)
                original = service.iter_results
                attempts = 0

                def interrupted(file_id):
                    nonlocal attempts
                    attempts += 1
                    values = iter(original(file_id))
                    yield next(values)
                    if attempts == 1:
                        raise httpx.ReadTimeout("private-test-key")
                    yield from values

                with (
                    patch.object(service, "iter_results", interrupted),
                    patch.object(batch.time, "sleep") as sleep,
                ):
                    self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(attempts, 2)
                sleep.assert_called_once_with(1.0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(3,)])
                self.assertEqual(service.submissions, 1)

    def test_transient_read_attempts_are_bounded_and_authentication_is_not_retried(
        self,
    ):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                self.run_action("submit", resume=False)
                for status, attempts in ((503, 3), (403, 1)):
                    error = RuntimeError("private-test-key")
                    error.status_code = status
                    with (
                        patch.object(service, "status", side_effect=error) as get,
                        patch.object(batch.time, "sleep") as sleep,
                        patch.object(
                            batch.time,
                            "time",
                            return_value=1000 if status == 503 else 10000,
                        ),
                    ):
                        if status == 503:
                            with patch("sys.stdout", new=io.StringIO()) as output:
                                self.assertEqual(
                                    self.run_action("status"),
                                    batch.RETRYABLE_OPERATION_EXIT_CODE,
                                )
                            self.assertNotIn("private-test-key", output.getvalue())
                        else:
                            with self.assertRaises(pipeline.InputError) as caught:
                                self.run_action("status")
                            self.assertNotIn(
                                "private-test-key", str(caught.exception)
                            )
                    self.assertEqual(get.call_count, attempts)
                    self.assertEqual(sleep.call_count, attempts - 1)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(
                    self.query("SELECT state FROM batch_jobs"), [("submitted",)]
                )

    def test_unknown_or_duplicate_response_rolls_back_entire_shard(self):
        for provider in ("openai", "google"):
            for problem in ("unknown", "duplicate"):
                with (
                    self.subTest(provider=provider, problem=problem),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    service, _ = self.fixture(Path(temporary), provider)
                    self.run_action("submit", resume=False)
                    results = service.jobs["job-1"]["results"]
                    malformed = json.loads(json.dumps(results[0]))
                    if problem == "unknown":
                        malformed["custom_id" if provider == "openai" else "key"] = (
                            "unrequested-id"
                        )
                    results.append(malformed)
                    with self.assertRaisesRegex(
                        pipeline.InputError, "unknown or duplicate"
                    ):
                        self.run_action("collect")
                    self.assertEqual(
                        self.query(
                            "SELECT COUNT(*) FROM batch_requests WHERE result_json IS NOT NULL OR error IS NOT NULL"
                        ),
                        [(0,)],
                    )
                    self.assertEqual(
                        self.query("SELECT state FROM batch_jobs"), [("submitted",)]
                    )
                    results.pop()
                    self.assertEqual(self.run_action("collect"), 0)
                    self.assertEqual(
                        self.query("SELECT COUNT(*) FROM annotations"), [(2,)]
                    )

    def test_changed_resume_configuration_fails_before_client_creation(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                _, constructor = self.fixture(Path(temporary), provider)
                self.run_action("prepare", resume=False)
                for extra in (
                    ("--model", "different-model"),
                    ("--max-output-tokens", "512"),
                    ("--max-text-chars", "20"),
                ):
                    with (
                        self.subTest(extra=extra),
                        self.assertRaisesRegex(
                            pipeline.InputError, "metadata does not match"
                        ),
                    ):
                        self.run_action("submit", *extra)
                constructor.assert_not_called()

    def test_old_topic_protocol_and_changed_effort_cannot_resume(self):
        for key, old_value in (
            ("prompt_version", "locus-generative-1"),
            ("reasoning_effort", "none"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                _, constructor = self.fixture(Path(temporary), "openai")
                self.run_action("prepare", resume=False)
                with closing(sqlite3.connect(self.ledger)) as db, db:
                    stored = pipeline.ledger_metadata(db)
                    stored[key] = old_value
                    db.execute(
                        "UPDATE metadata SET value=? WHERE key='run_metadata'",
                        (json.dumps(stored),),
                    )
                with self.assertRaisesRegex(
                    pipeline.InputError, "metadata does not match"
                ):
                    self.run_action("submit")
                constructor.assert_not_called()

    def test_uncertain_submit_blocks_repost_and_attachment_checks_identity(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                service.fail_submit_once = True
                with self.assertRaisesRegex(pipeline.InputError, "operation failed"):
                    self.run_action("submit", resume=False)
                self.assertEqual(
                    self.query("SELECT state, remote_id FROM batch_jobs"),
                    [("submitting", None)],
                )
                with self.assertRaisesRegex(
                    pipeline.InputError, "uncertain submission"
                ):
                    self.run_action("submit")
                self.assertEqual(service.submissions, 1)
                status = service.jobs["job-1"]["status"]
                for field in (
                    "display_name",
                    "input_file_id" if provider == "openai" else "model",
                ):
                    original = status[field]
                    status[field] = "incorrect"
                    with self.assertRaisesRegex(pipeline.InputError, "does not match"):
                        self.run_action(
                            "attach", "--batch-number", "1", "--batch-id", "job-1"
                        )
                    status[field] = original
                self.assertEqual(
                    self.run_action(
                        "attach", "--batch-number", "1", "--batch-id", "job-1"
                    ),
                    0,
                )
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(service.submissions, 1)

    def test_google_key_accepts_either_documented_name(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": " test-only "}, clear=True):
            self.assertEqual(batch.api_key("google", None), "test-only")
            with self.assertRaisesRegex(
                pipeline.InputError, "Set OPENAI_API_KEY locally"
            ):
                batch.api_key("openai", None)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                pipeline.InputError, "Set GEMINI_API_KEY or GOOGLE_API_KEY"
            ):
                batch.api_key("google", None)

    def test_server_error_after_creation_is_uncertain_and_never_auto_reposted(self):
        for provider, attribute in (("openai", "status_code"), ("google", "code")):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                submit = service.submit

                def accepted_then_failed(*args):
                    submit(*args)
                    error = RuntimeError("The create response failed after acceptance")
                    setattr(error, attribute, 500)
                    raise error

                with patch.object(service, "submit", side_effect=accepted_then_failed):
                    with self.assertRaises(pipeline.InputError):
                        self.run_action("submit", resume=False)
                self.assertEqual(
                    self.query("SELECT state,remote_id FROM batch_jobs"),
                    [("submitting", None)],
                )
                with self.assertRaisesRegex(
                    pipeline.InputError, "uncertain submission"
                ):
                    self.run_action("submit")
                self.assertEqual(service.submissions, 1)
                self.assertEqual(len(service.uploads), 1)

    def test_definite_create_rejections_do_not_require_an_impossible_remote_attachment(
        self,
    ):
        for provider, attribute in (("openai", "status_code"), ("google", "code")):
            for code in (400,):
                with (
                    self.subTest(provider=provider, code=code),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    service, _ = self.fixture(Path(temporary), provider)
                    rejection = RuntimeError("fixture rejection")
                    setattr(rejection, attribute, code)
                    with patch.object(service, "submit", side_effect=rejection):
                        with self.assertRaisesRegex(
                            pipeline.InputError, "rejected batch"
                        ) as raised:
                            self.run_action("submit", resume=False)
                    self.assertIn(f'"http_status":{code}', str(raised.exception))
                    self.assertEqual(
                        self.query("SELECT state FROM batch_jobs"), [("preparing",)]
                    )
                    self.assertEqual(self.run_action("submit"), 0)
                    self.assertEqual(self.run_action("collect"), 0)
                    self.assertEqual(service.submissions, 1)

    def test_definite_quota_rejection_waits_then_reuses_uploaded_file_once(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                with patch.object(batch.time, "time", return_value=1000):
                    with patch.object(
                        service, "submit", side_effect=self.quota_rejection(provider)
                    ) as submit:
                        self.assertEqual(self.run_action("submit", resume=False), 0)
                    self.assertEqual(submit.call_count, 1)
                    self.assertEqual(
                        self.query("SELECT state,file_id FROM batch_jobs"),
                        [("quota_wait", "input-1")],
                    )
                    assigned = self.query(
                        "SELECT id,job_number FROM batch_requests ORDER BY id"
                    )
                    self.assertTrue(all(number == 1 for _, number in assigned))
                    quota = json.loads(
                        self.query(
                            "SELECT value FROM metadata WHERE key='batch_quota:1'"
                        )[0][0]
                    )
                    self.assertGreaterEqual(quota["retry_at"], 1060)
                    self.assertEqual(quota["attempts"], 1)
                    self.assertFalse(quota["shrink"])
                    self.assertEqual(quota["error"]["http_status"], 429)
                    self.assertEqual(quota["error"]["code"], "rate_limit_exceeded")
                    self.assertEqual(
                        quota["error"]["request_id"], "quota-request-fixture"
                    )
                    with closing(sqlite3.connect(self.ledger)) as db:
                        progress = batch.summary(db)
                    self.assertEqual(len(progress["deferred_jobs"]), 1)
                    self.assertIsNotNone(progress["next_retry_at_utc"])
                    with patch.object(service, "submit") as submit:
                        self.assertEqual(self.run_action("submit"), 0)
                        submit.assert_not_called()
                    self.assertEqual(
                        self.query(
                            "SELECT id,job_number FROM batch_requests ORDER BY id"
                        ),
                        assigned,
                    )
                self.assertEqual(len(service.uploads), 1)
                self.assertEqual(service.submissions, 0)
                with patch.object(
                    batch.time, "time", return_value=quota["retry_at"] + 1
                ):
                    self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(len(service.uploads), 1)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(
                    self.query("SELECT state,remote_id FROM batch_jobs"),
                    [("submitted", "job-1")],
                )
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])

    def test_rejected_token_queue_shard_shrinks_without_duplicate_accepted_rows(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=4)
                with (
                    patch.object(batch.time, "time", return_value=1000),
                    patch.object(
                        service,
                        "submit",
                        side_effect=self.quota_rejection(provider, token_queue=True),
                    ),
                ):
                    self.assertEqual(self.run_action("submit", resume=False), 0)
                quota = json.loads(
                    self.query(
                        "SELECT value FROM metadata WHERE key='batch_quota:1'"
                    )[0][0]
                )
                self.assertTrue(quota["shrink"])
                self.assertEqual(len(service.uploads["input-1"]), 4)
                self.assertEqual(service.submissions, 0)
                with patch.object(
                    batch.time, "time", return_value=quota["retry_at"] + 1
                ):
                    self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(len(service.uploads["input-2"]), 2)
                self.assertEqual(
                    self.query("SELECT state FROM batch_jobs ORDER BY number"),
                    [("quota_split",), ("submitted",)],
                )
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(len(service.uploads["input-3"]), 2)
                self.assertEqual(self.run_action("collect"), 0)
                key = "custom_id" if provider == "openai" else "key"
                accepted_ids = [
                    row[key]
                    for file in ("input-2", "input-3")
                    for row in service.uploads[file]
                ]
                self.assertEqual(len(accepted_ids), 4)
                self.assertEqual(len(set(accepted_ids)), 4)
                self.assertEqual(
                    set(accepted_ids),
                    {row[0] for row in self.query("SELECT id FROM batch_requests")},
                )
                self.assertEqual(service.submissions, 2)
                self.assertEqual(
                    self.query(
                        "SELECT status,COUNT(*) FROM annotations GROUP BY status"
                    ),
                    [("annotated", 4)],
                )

    def test_manual_output_waits_for_batches_and_preserves_previous_output(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                self.assertEqual(
                    self.run_action(
                        "submit", "--output", str(self.output), resume=False
                    ),
                    0,
                )
                self.assertFalse(self.output.exists())
                service.jobs["job-1"]["status"].update(
                    status="in_progress", done=False
                )
                self.assertEqual(
                    self.run_action("collect", "--output", str(self.output)), 0
                )
                self.assertFalse(self.output.exists())
                self.output.write_text("previous complete export\n")
                self.assertEqual(
                    self.run_action(
                        "collect",
                        "--output",
                        str(self.output),
                        "--overwrite-output",
                    ),
                    0,
                )
                self.assertEqual(self.output.read_text(), "previous complete export\n")
                service.jobs["job-1"]["status"].update(status="completed", done=True)
                self.assertEqual(
                    self.run_action(
                        "collect",
                        "--output",
                        str(self.output),
                        "--overwrite-output",
                    ),
                    0,
                )
                self.assertEqual(len(self.output.read_text().splitlines()), 2)
                self.assertEqual(service.submissions, 1)

    def test_single_row_token_quota_rejections_back_off_without_uploading_again(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=1)
                rejection = self.quota_rejection(provider, token_queue=True)
                rejection.response = httpx.Response(429)
                now = 1000
                delays = []
                for attempt in (1, 2):
                    with (
                        patch.object(batch.time, "time", return_value=now),
                        patch.object(service, "submit", side_effect=rejection),
                    ):
                        self.assertEqual(
                            self.run_action("submit", resume=attempt != 1), 0
                        )
                    quota = json.loads(
                        self.query(
                            "SELECT value FROM metadata WHERE key='batch_quota:1'"
                        )[0][0]
                    )
                    self.assertEqual(quota["attempts"], attempt)
                    delays.append(quota["retry_at"] - now)
                    now = quota["retry_at"] + 1
                self.assertGreaterEqual(delays[0], 60)
                self.assertGreaterEqual(delays[1], 2 * delays[0])
                self.assertEqual(len(service.uploads), 1)
                self.assertEqual(service.submissions, 0)
                with patch.object(batch.time, "time", return_value=now):
                    self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(len(service.uploads), 1)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(self.run_action("collect"), 0)

    def test_long_read_retry_after_embargo_blocks_remote_actions_until_due(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, constructor = self.fixture(Path(temporary), provider)
                self.assertEqual(self.run_action("submit", resume=False), 0)
                original_rows = self.query(
                    "SELECT id,job_number,result_json,error FROM batch_requests "
                    "ORDER BY id"
                )
                error = self.quota_rejection(provider)
                error.response = httpx.Response(429, headers={"Retry-After": "3600"})
                with patch.object(batch.time, "time", return_value=1000):
                    with (
                        patch.object(service, "status", side_effect=error) as status,
                        patch.object(batch.time, "sleep") as sleep,
                    ):
                        self.assertEqual(
                            self.run_action("collect"),
                            batch.RETRYABLE_OPERATION_EXIT_CODE,
                        )
                    self.assertEqual(status.call_count, 1)
                    sleep.assert_not_called()
                    with closing(sqlite3.connect(self.ledger)) as db:
                        progress = batch.summary(db)
                    retry_at = progress["operation_retry_at"]
                    self.assertGreaterEqual(retry_at, 4600)
                    self.assertIsNotNone(progress["operation_retry_at_utc"])
                    self.assertIsNone(progress["next_retry_at"])
                    embargo = json.loads(
                        self.query(
                            "SELECT value FROM metadata WHERE key='batch_operation_retry'"
                        )[0][0]
                    )
                    self.assertEqual(embargo["action"], "collect")
                    before = constructor.call_count
                    for action in ("collect", "status", "submit", "retry"):
                        with self.subTest(deferred_action=action):
                            self.assertEqual(
                                self.run_action(action),
                                batch.RETRYABLE_OPERATION_EXIT_CODE,
                            )
                    self.assertEqual(constructor.call_count, before)
                self.assertEqual(
                    self.query(
                        "SELECT id,job_number,result_json,error FROM batch_requests "
                        "ORDER BY id"
                    ),
                    original_rows,
                )
                self.assertEqual(
                    self.query("SELECT state,remote_id FROM batch_jobs"),
                    [("submitted", "job-1")],
                )
                self.assertEqual(service.submissions, 1)
                with patch.object(batch.time, "time", return_value=retry_at + 1):
                    self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])
                with closing(sqlite3.connect(self.ledger)) as db:
                    self.assertIsNone(batch.summary(db)["operation_retry_at"])
                self.assertEqual(service.submissions, 1)

    def test_google_cleanup_is_opt_in_and_limited_to_committed_job_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "google")
            service.google_file_ids = True
            self.assertEqual(self.run_action("submit", resume=False), 0)
            service.jobs["job-1"]["status"]["error_files"] = ["files/job-1-results"]
            service.uploads["files/unrelated"] = []
            self.assertEqual(self.run_action("collect"), 0)
            self.assertEqual(service.deleted_files, [])
            with closing(sqlite3.connect(self.ledger)) as db:
                self.assertEqual(batch.summary(db)["pending_cleanup_files"], 2)
            self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])
            self.assertEqual(
                self.run_action("collect", "--cleanup-remote-files"), 0
            )
            self.assertEqual(
                set(service.deleted_files),
                {"files/input-1", "files/job-1-results"},
            )
            self.assertEqual(len(service.deleted_files), 2)
            with closing(sqlite3.connect(self.ledger)) as db:
                self.assertEqual(batch.summary(db)["pending_cleanup_files"], 0)
            self.assertEqual(
                self.run_action("collect", "--cleanup-remote-files"), 0
            )
            self.assertEqual(len(service.deleted_files), 2)
            self.assertEqual(service.submissions, 1)

    def test_google_long_output_is_saved_as_skipped_without_blocking_export(self):
        google_client_class = batch.google_batch.GoogleBatchClient
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "google")
            service.google_file_ids = True
            self.assertEqual(self.run_action("submit", resume=False), 0)
            output_id = "files/" + "b" * 41
            service.jobs["job-1"]["status"]["output_files"] = [output_id]
            results = service.jobs["job-1"]["results"]

            def output_rows(file_id):
                self.assertEqual(file_id, output_id)
                yield from results

            with patch.object(service, "iter_results", side_effect=output_rows):
                self.assertEqual(self.run_action("collect"), 0)
            saved = self.query(
                "SELECT source_id,annotation_json FROM annotations ORDER BY source_id"
            )
            cleanup_client = object.__new__(google_client_class)
            cleanup_client.client = Mock()
            with patch.object(service, "delete_file", cleanup_client.delete_file):
                self.assertEqual(
                    self.run_action(
                        "collect", "--cleanup-remote-files", "--output", str(self.output)
                    ),
                    0,
                )
                self.assertEqual(
                    self.run_action("collect", "--cleanup-remote-files"), 0
                )
            cleanup_client.client.files.delete.assert_called_once_with(
                name="files/input-1"
            )
            cleanup = json.loads(
                self.query("SELECT value FROM metadata WHERE key='batch_cleanup:1'")[0][0]
            )
            self.assertEqual(cleanup["pending"], [])
            self.assertEqual(cleanup["deleted"], ["files/input-1"])
            self.assertEqual(cleanup["skipped"], [output_id])
            self.assertEqual(
                self.query(
                    "SELECT source_id,annotation_json FROM annotations ORDER BY source_id"
                ),
                saved,
            )
            self.assertEqual(len(self.output.read_text().splitlines()), 2)
            self.assertEqual(service.submissions, 1)
            with closing(sqlite3.connect(self.ledger)) as db:
                self.assertEqual(batch.summary(db)["pending_cleanup_files"], 0)

    def test_google_cleanup_never_runs_before_atomic_collection_commits(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "google")
            service.google_file_ids = True
            self.run_action("submit", resume=False)
            results = service.jobs["job-1"]["results"]
            results.append(dict(results[0]))
            with self.assertRaisesRegex(pipeline.InputError, "duplicate"):
                self.run_action("collect", "--cleanup-remote-files")
            self.assertEqual(service.deleted_files, [])
            self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(0,)])
            self.assertEqual(
                self.query(
                    "SELECT COUNT(*) FROM metadata WHERE key LIKE 'batch_cleanup:%'"
                ),
                [(0,)],
            )
            results.pop()
            self.assertEqual(
                self.run_action("collect", "--cleanup-remote-files"), 0
            )
            self.assertEqual(len(service.deleted_files), 2)
            self.assertEqual(service.submissions, 1)

    def test_google_cleanup_failure_keeps_annotations_and_resumes_remaining_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "google")
            service.google_file_ids = True
            self.run_action("submit", resume=False)
            delete = service.delete_file

            def delete_first_then_fail(file_id):
                self.assertEqual(
                    self.query("SELECT COUNT(*) FROM annotations"), [(2,)]
                )
                self.assertEqual(
                    self.query("SELECT state FROM batch_jobs"), [("collected",)]
                )
                if service.deleted_files:
                    raise httpx.ConnectTimeout("temporary deletion failure")
                delete(file_id)

            with (
                patch.object(batch.time, "time", return_value=1000),
                patch.object(batch.time, "sleep"),
                patch.object(service, "delete_file", side_effect=delete_first_then_fail),
            ):
                self.assertEqual(
                    self.run_action("collect", "--cleanup-remote-files"),
                    batch.RETRYABLE_OPERATION_EXIT_CODE,
                )
            self.assertEqual(len(service.deleted_files), 1)
            saved = self.query(
                "SELECT source_id,annotation_json FROM annotations ORDER BY source_id"
            )
            cleanup = json.loads(
                self.query("SELECT value FROM metadata WHERE key='batch_cleanup:1'")[0][0]
            )
            self.assertEqual(len(cleanup["pending"]), 1)
            self.assertEqual(cleanup["deleted"], service.deleted_files)
            with closing(sqlite3.connect(self.ledger)) as db:
                progress = batch.summary(db)
            self.assertEqual(progress["pending_cleanup_files"], 1)
            with patch.object(
                batch.time, "time", return_value=progress["operation_retry_at"] + 1
            ):
                self.assertEqual(
                    self.run_action("collect", "--cleanup-remote-files"), 0
                )
            self.assertEqual(len(service.deleted_files), 2)
            self.assertEqual(len(set(service.deleted_files)), 2)
            self.assertEqual(
                self.query(
                    "SELECT source_id,annotation_json FROM annotations ORDER BY source_id"
                ),
                saved,
            )
            self.assertEqual(service.submissions, 1)

    def test_collection_preserves_paid_results_when_the_external_input_is_unavailable(
        self,
    ):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                self.fixture(Path(temporary), provider)
                self.run_action("submit", resume=False)
                saved = self.input.read_bytes()
                self.input.unlink()
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])
                self.input.write_bytes(saved)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])

    def test_uncertain_reset_is_explicit_local_and_does_not_resubmit(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, constructor = self.fixture(Path(temporary), provider)
                with patch.object(
                    service, "submit", side_effect=TimeoutError("No response")
                ):
                    with self.assertRaises(pipeline.InputError):
                        self.run_action("submit", resume=False)
                before = constructor.call_count
                with patch.object(
                    batch,
                    "load_dotenv",
                    side_effect=AssertionError("local reset must not load keys"),
                ):
                    with self.assertRaisesRegex(
                        pipeline.InputError, "requires --batch-number"
                    ):
                        self.run_action("reset-uncertain")
                    self.assertEqual(
                        self.run_action("reset-uncertain", "--batch-number", "1"), 0
                    )
                    with self.assertRaisesRegex(
                        pipeline.InputError, "Only an uncertain"
                    ):
                        self.run_action("reset-uncertain", "--batch-number", "1")
                self.assertEqual(constructor.call_count, before)
                self.assertEqual(service.submissions, 0)
                self.assertEqual(
                    self.query("SELECT state FROM batch_jobs"), [("abandoned",)]
                )
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(service.submissions, 1)

    def test_dry_run_and_prepare_need_no_key_or_sdk(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                _, constructor = self.fixture(Path(temporary), provider)
                with (
                    patch.dict("os.environ", {}, clear=True),
                    patch.object(
                        batch,
                        "load_dotenv",
                        side_effect=AssertionError("must not load secrets"),
                    ),
                ):
                    args = pipeline.build_parser().parse_args(
                        [*self.common, "--dry-run"]
                    )
                    self.assertEqual(batch.run(args), 0)
                    self.assertFalse(self.ledger.exists())
                    with self.assertRaisesRegex(pipeline.InputError, "50000"):
                        self.run_action(
                            "prepare", "--batch-size", "50001", resume=False
                        )
                    self.assertEqual(
                        self.run_action(
                            "prepare", "--batch-size", "50000", resume=False
                        ),
                        0,
                    )
                constructor.assert_not_called()

    def test_selected_maximum_stays_fixed_on_resume(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=3)
                self.run_action("prepare", "--max-records", "1", resume=False)
                self.assertEqual(
                    self.query("SELECT COUNT(*) FROM batch_requests"), [(1,)]
                )
                with self.assertRaisesRegex(
                    pipeline.InputError, "metadata does not match"
                ):
                    self.run_action("submit", "--max-records", "3")
                self.run_action("submit", "--max-records", "1")
                self.run_action("collect", "--max-records", "1")
                self.assertEqual(
                    self.query("SELECT COUNT(*) FROM batch_requests"), [(1,)]
                )
                self.assertEqual(sum(map(len, service.uploads.values())), 1)

    def test_attachment_name_distinguishes_different_request_contents(self):
        for provider in ("openai", "google"):
            names = []
            for cap in (20, 40):
                with (
                    self.subTest(provider=provider, cap=cap),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    service, _ = self.fixture(Path(temporary), provider)
                    self.run_action(
                        "submit", "--max-text-chars", str(cap), resume=False
                    )
                    name = service.jobs["job-1"]["status"]["display_name"]
                    payload = "".join(
                        pipeline.canonical_json(row) + "\n"
                        for row in service.uploads["input-1"]
                    )
                    self.assertTrue(
                        name.startswith(
                            f"locus-{sha256(payload.encode()).hexdigest()}-"
                        )
                    )
                    names.append(name)
            self.assertNotEqual(
                names[0],
                names[1],
                "Recovery identity must bind actual uploaded requests, not only source and policy",
            )

    def test_explicit_retry_resubmits_failed_requests_only(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=3)
                self.run_action("submit", resume=False)
                job = service.jobs["job-1"]
                key_field = "custom_id" if provider == "openai" else "key"
                failed_id = job["results"][0][key_field]
                job["results"][0] = {
                    key_field: failed_id,
                    "error": {"code": "unavailable"},
                }
                self.assertEqual(self.run_action("collect"), batch.ROW_ERRORS_EXIT_CODE)
                completed_before = self.query(
                    "SELECT id, result_json FROM batch_requests WHERE result_json IS NOT NULL ORDER BY id"
                )
                self.assertEqual(
                    self.query(
                        "SELECT COUNT(*) FROM annotations WHERE status='failed'"
                    ),
                    [(1,)],
                )
                completed_annotations = self.query(
                    "SELECT source_id,annotation_json FROM annotations "
                    "WHERE status='annotated' ORDER BY source_id"
                )
                self.assertEqual(len(completed_annotations), 2)
                self.assertEqual(self.run_action("submit"), batch.ROW_ERRORS_EXIT_CODE)
                self.assertEqual(service.submissions, 1)
                with closing(sqlite3.connect(self.ledger)) as db, db:
                    db.execute(
                        "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                        (
                            "batch_operation_retry",
                            json.dumps(
                                {
                                    "retry_at": 2000,
                                    "attempts": 1,
                                    "action": "collect",
                                    "error": {},
                                }
                            ),
                        ),
                    )
                with patch.object(batch.time, "time", return_value=1000):
                    self.assertEqual(
                        self.run_action("retry"), batch.RETRYABLE_OPERATION_EXIT_CODE
                    )
                with closing(sqlite3.connect(self.ledger)) as db:
                    self.assertEqual(batch.summary(db)["retry_passes"], 0)
                with patch.object(batch.time, "time", return_value=2001):
                    self.assertEqual(self.run_action("retry"), 0)
                with closing(sqlite3.connect(self.ledger)) as db:
                    self.assertEqual(batch.summary(db)["retry_passes"], 1)
                self.assertEqual(
                    [r[key_field] for r in service.uploads["input-2"]], [failed_id]
                )
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(3,)])
                self.assertEqual(
                    self.query(
                        "SELECT COUNT(*) FROM annotations WHERE status='failed'"
                    ),
                    [(0,)],
                )
                after_annotations = dict(
                    self.query("SELECT source_id,annotation_json FROM annotations")
                )
                for source_id, encoded in completed_annotations:
                    self.assertEqual(after_annotations[source_id], encoded)
                self.assertEqual(
                    self.query(
                        "SELECT COUNT(*) FROM batch_requests WHERE error IS NOT NULL"
                    ),
                    [(0,)],
                )
                completed_after = dict(
                    self.query(
                        "SELECT id, result_json FROM batch_requests WHERE result_json IS NOT NULL"
                    )
                )
                for request_id, result in completed_before:
                    self.assertEqual(completed_after[request_id], result)
                self.assertEqual(self.run_action("retry"), 0)
                self.assertEqual(service.submissions, 2)
                with closing(sqlite3.connect(self.ledger)) as db:
                    self.assertEqual(batch.summary(db)["retry_passes"], 1)

    def test_pending_status_and_collection_do_not_submit_or_mark_missing(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider)
                self.run_action("submit", resume=False)
                status = service.jobs["job-1"]["status"]
                status.update(status="in_progress", done=False)
                with patch.object(
                    service,
                    "iter_results",
                    side_effect=AssertionError(
                        "pending results must not be downloaded"
                    ),
                ):
                    self.assertEqual(self.run_action("status"), 0)
                    self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(
                    self.query(
                        "SELECT COUNT(*) FROM batch_requests WHERE result_json IS NOT NULL OR error IS NOT NULL"
                    ),
                    [(0,)],
                )
                self.assertEqual(
                    self.query("SELECT state FROM batch_jobs"), [("submitted",)]
                )
                status.update(status="completed", done=True)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])

    def test_unknown_google_state_keeps_paid_job_and_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "google")
            self.assertEqual(self.run_action("submit", resume=False), 0)
            failure = batch.google_batch.GoogleBatchStatusError(
                "Google batch job-1 returned an unknown state; "
                "check its provider status before resuming"
            )
            with patch.object(service, "status", side_effect=failure):
                with self.assertRaises(pipeline.InputError) as raised:
                    self.run_action("collect")
            message = str(raised.exception)
            for expected in ("job-1", "invalid_batch_status", "check its provider"):
                self.assertIn(expected, message)
            self.assertEqual(
                self.query("SELECT state,remote_id FROM batch_jobs"),
                [("submitted", "job-1")],
            )
            self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(0,)])
            self.assertEqual(self.run_action("collect"), 0)
            self.assertEqual(service.submissions, 1)
            self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(2,)])

    def test_smaller_shard_resume_keeps_successes_after_upload_timeout(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=5)
                self.assertEqual(
                    self.run_action("submit", "--batch-size", "2", resume=False), 0
                )
                self.assertEqual(self.run_action("collect"), 0)
                saved = self.query(
                    "SELECT source_id,annotation_json FROM annotations ORDER BY source_id"
                )
                self.assertEqual(len(saved), 2)
                with (
                    patch.object(batch.time, "time", return_value=1000),
                    patch.object(
                        service,
                        "upload",
                        side_effect=httpx.WriteTimeout("temporary upload timeout"),
                    ),
                ):
                    self.assertEqual(
                        self.run_action("submit", "--batch-size", "2"),
                        batch.RETRYABLE_OPERATION_EXIT_CODE,
                    )
                self.assertEqual(service.submissions, 1)
                with closing(sqlite3.connect(self.ledger)) as db:
                    retry_at = batch.summary(db)["operation_retry_at"]
                with patch.object(batch.time, "time", return_value=retry_at + 1):
                    self.assertEqual(self.run_action("submit", "--batch-size", "1"), 0)
                self.assertEqual(self.run_action("collect"), 0)
                for _ in range(2):
                    self.assertEqual(self.run_action("submit", "--batch-size", "1"), 0)
                    self.assertEqual(self.run_action("collect"), 0)
                key = "custom_id" if provider == "openai" else "key"
                accepted = [
                    row[key] for rows in service.uploads.values() for row in rows
                ]
                self.assertEqual(len(accepted), 5)
                self.assertEqual(len(set(accepted)), 5)
                self.assertEqual(
                    [len(rows) for rows in service.uploads.values()], [2, 1, 1, 1]
                )
                self.assertEqual(service.submissions, 4)
                annotations = dict(
                    self.query("SELECT source_id,annotation_json FROM annotations")
                )
                self.assertEqual(len(annotations), 5)
                for source_id, encoded in saved:
                    self.assertEqual(annotations[source_id], encoded)
                self.assertEqual(self.run_action("submit", "--batch-size", "1"), 0)
                self.assertEqual(service.submissions, 4)

    def test_utf8_byte_budget_splits_shards_and_rejects_oversized_rows_before_calls(
        self,
    ):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                service, _ = self.fixture(root, provider, count=3)
                records = [
                    json.loads(line) for line in self.input.read_text().splitlines()
                ]
                for row in records:
                    row["content"] += "\U0001f642" * 4096
                self.input.write_text(
                    "".join(
                        json.dumps(row, ensure_ascii=False) + "\n" for row in records
                    ),
                    encoding="utf-8",
                )
                config = batch.configuration(
                    pipeline.build_parser().parse_args(self.common), hash_input=False
                )
                source_id, _ = pipeline.source_identity(records[0], 0)
                request = batch.request_for(records[0], source_id, config)
                request_bytes = (
                    len(pipeline.canonical_json(request).encode("utf-8")) + 1
                )
                with patch.object(batch, "MAX_SHARD_BYTES", request_bytes * 2):
                    self.run_action("prepare", resume=False)
                    self.run_action("submit")
                    self.run_action("collect")
                    self.run_action("submit")
                    self.run_action("collect")
                self.assertEqual(
                    [len(value) for value in service.uploads.values()], [2, 1]
                )
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(3,)])
                for rows in service.uploads.values():
                    size = sum(
                        len(pipeline.canonical_json(row).encode("utf-8")) + 1
                        for row in rows
                    )
                    self.assertLessEqual(size, request_bytes * 2)

                oversized = root / "oversized"
                oversized.mkdir()
                _, constructor = self.fixture(oversized, provider, count=1)
                with patch.object(batch, "MAX_SHARD_BYTES", 1):
                    with self.assertRaisesRegex(pipeline.InputError, "byte limit"):
                        self.run_action("prepare", resume=False)
                constructor.assert_not_called()

    def test_shards_are_bounded_and_indexed_without_repeated_input_scans(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                service, _ = self.fixture(Path(temporary), provider, count=3)
                flags = ("--batch-size", "1", "--max-inflight-batches", "2")
                self.run_action("prepare", *flags, resume=False)
                # Preparation is the only full parsing pass; poll/submit seek the index.
                with patch.object(
                    batch, "indexed_jsonl", side_effect=AssertionError("no full scan")
                ):
                    self.run_action("submit", *flags)
                    self.assertEqual(service.submissions, 2)
                    self.run_action("submit", *flags)
                    self.assertEqual(service.submissions, 2)
                    with patch.object(
                        batch,
                        "sha256_file",
                        side_effect=AssertionError("no poll rehash"),
                    ):
                        self.run_action("collect", *flags)
                    self.assertEqual(
                        self.query("SELECT COUNT(*) FROM annotations"), [(2,)]
                    )
                    self.run_action("submit", *flags)
                    self.run_action("collect", *flags)
                self.assertEqual(service.submissions, 3)
                self.assertEqual(
                    [len(requests) for requests in service.uploads.values()], [1] * 3
                )
                self.assertEqual(self.query("SELECT COUNT(*) FROM annotations"), [(3,)])

    def test_pending_selection_has_no_corpus_sort_and_migrates_existing_ledgers(self):
        with tempfile.TemporaryDirectory() as temporary:
            service, _ = self.fixture(Path(temporary), "openai", count=3)
            self.run_action("prepare", resume=False)
            with closing(sqlite3.connect(self.ledger)) as db, db:
                before = db.execute(
                    "SELECT id, request_sha256 FROM batch_requests ORDER BY id"
                ).fetchall()
                db.execute("DROP INDEX batch_request_pending_ordered")
                db.execute(
                    """CREATE INDEX batch_request_pending ON batch_requests(source_index)
                              WHERE job_number IS NULL AND result_json IS NULL AND error IS NULL"""
                )
            self.assertEqual(self.run_action("submit", "--batch-size", "1"), 0)
            with closing(sqlite3.connect(self.ledger)) as db:
                plan = [
                    row[3]
                    for row in db.execute(
                        """EXPLAIN QUERY PLAN SELECT id FROM batch_requests
                    WHERE job_number IS NULL AND result_json IS NULL AND error IS NULL
                    ORDER BY source_index LIMIT 5000"""
                    )
                ]
                self.assertTrue(
                    any("batch_request_pending_ordered" in line for line in plan), plan
                )
                self.assertFalse(any("TEMP B-TREE" in line for line in plan), plan)
                self.assertEqual(
                    db.execute(
                        "SELECT id, request_sha256 FROM batch_requests ORDER BY id"
                    ).fetchall(),
                    before,
                )
            self.assertEqual(service.submissions, 1)

    def test_prepare_and_export_do_not_parse_beyond_selected_record_limit(self):
        for provider in ("openai", "google"):
            with (
                self.subTest(provider=provider),
                tempfile.TemporaryDirectory() as temporary,
            ):
                _, constructor = self.fixture(Path(temporary), provider, count=1)
                original = self.input.read_text(encoding="utf-8")
                self.input.write_text(original + "not valid JSON\n", encoding="utf-8")
                self.assertEqual(
                    self.run_action("prepare", "--max-records", "1", resume=False), 0
                )
                self.assertEqual(
                    self.query("SELECT COUNT(*) FROM batch_requests"), [(1,)]
                )
                constructor.assert_not_called()
                self.assertEqual(self.run_action("submit"), 0)
                self.assertEqual(self.run_action("collect"), 0)
                self.assertEqual(
                    self.run_action("prepare", "--output", str(self.output)), 0
                )
                self.assertEqual(len(self.output.read_text().splitlines()), 1)


class RunnerTests(unittest.TestCase):
    """Exercise one provider per invocation, with every paid client replaced."""

    def setUp(self):
        from neurips2026_locus import runner

        self.runner = runner
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.input = self.directory / "input.jsonl"
        self.env = self.directory / ".env"
        self.env.write_text("", encoding="utf-8")
        self.records = [
            {"audit_id": str(i), "content": f"Test fixture rule {i}."}
            for i in range(250)
        ]
        self.write_rows()
        self.args = [
            "human_audit",
            "--input",
            str(self.input),
            "--env-file",
            str(self.env),
            "--provider",
            "google",
            "--model",
            "fake-jev-1",
            "--check-every",
            "0.00001",
        ]
        from test_pipeline import FakeJevClient

        FakeJevClient.instances = []
        self.jev_client = FakeJevClient
        self.openai = FakeBatchService("openai")
        self.google = FakeBatchService("google")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.runner, "ROOT", self.directory))
        self.stack.enter_context(patch.object(pipeline, "JevClient", FakeJevClient))
        self.openai_constructor = self.stack.enter_context(
            patch.object(
                batch.openai_batch, "OpenAIBatchClient", return_value=self.openai
            )
        )
        self.google_constructor = self.stack.enter_context(
            patch.object(
                batch.google_batch, "GoogleBatchClient", return_value=self.google
            )
        )
        self.stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "test",
                    "GEMINI_API_KEY": "test",
                    "TYPESAFE_API_KEY": "test",
                },
                clear=True,
            )
        )
        self.stack.enter_context(
            patch.object(self.runner.importlib.util, "find_spec", return_value=object())
        )
        self.stack.enter_context(patch("sys.stdout", new=io.StringIO()))
        self.stack.enter_context(patch("sys.stderr", new=io.StringIO()))

    def write_rows(self):
        self.input.write_text(
            "".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8"
        )

    def test_selected_provider_completes_and_resumes_without_calling_peers(self):
        self.records[0].update(header="STRUCTURAL-FIXTURE: Chapter 1", content="")
        self.write_rows()
        for provider in self.runner.PROVIDERS:
            before = (
                sum(len(c.requests) for c in self.jev_client.instances),
                self.openai.submissions,
                self.google.submissions,
            )
            name = "only_" + provider
            arguments = [*self.args, "--provider", provider, "--name", name]
            self.assertEqual(self.runner.main(arguments), 0)
            after = (
                sum(len(c.requests) for c in self.jev_client.instances),
                self.openai.submissions,
                self.google.submissions,
            )
            self.assertEqual(
                tuple(new - old for old, new in zip(before, after)),
                tuple(
                    (
                        (500 if selected == "typesafe" else 1)
                        if selected == provider
                        else 0
                    )
                    for selected in self.runner.PROVIDERS
                ),
            )
            output = self.directory / "outputs" / f"{name}_{provider}.jsonl"
            self.assertEqual(len(output.read_text().splitlines()), 250)
            self.assertEqual(
                len(list((self.directory / "outputs").glob(f"{name}_*.jsonl"))), 1
            )
            self.assertEqual(self.runner.main(arguments), 0)
            self.assertEqual(
                (
                    sum(len(c.requests) for c in self.jev_client.instances),
                    self.openai.submissions,
                    self.google.submissions,
                ),
                after,
            )

    def test_default_and_explicit_timeouts_reach_the_selected_provider_client(self):
        for provider in self.runner.PROVIDERS:
            for timeout in (None, 12.5):
                with self.subTest(provider=provider, timeout=timeout):
                    label = "default" if timeout is None else "custom"
                    arguments = [
                        *self.args,
                        "--provider", provider,
                        "--name", f"timeout_{provider}_{label}",
                    ]
                    if timeout is not None:
                        arguments += ["--timeout", str(timeout)]
                    expected = timeout or (45.0 if provider == "typesafe" else 600.0)
                    if provider == "typesafe":
                        with patch.object(
                            pipeline, "JevClient", wraps=self.jev_client
                        ) as client:
                            self.assertEqual(self.runner.main(arguments), 0)
                        calls = client.call_args_list
                        option = "timeout_seconds"
                    else:
                        client = getattr(self, f"{provider}_constructor")
                        before = client.call_count
                        self.assertEqual(self.runner.main(arguments), 0)
                        calls = client.call_args_list[before:]
                        option = "timeout"
                    self.assertTrue(calls)
                    self.assertTrue(
                        all(call.kwargs[option] == expected for call in calls)
                    )

    def test_dataset_defaults_and_input_overrides_keep_distinct_run_names(self):
        (self.directory / "pyproject.toml").write_text("# Source-checkout marker\n")
        audit = self.runner.parse_args(
            ["human_audit", "--provider", "google", "--env-file", str(self.env)]
        )
        self.assertEqual(audit.dataset, "human_audit")
        self.assertEqual(audit.name, "human_audit")
        self.assertEqual(
            audit.input,
            (
                self.directory.parent
                / "human_annotation"
                / "LOCUS_snorkel_labels.csv"
            ).resolve(),
        )
        self.assertEqual(audit.expect_rows, 250)
        self.assertIsNone(audit.max_text_chars)
        self.assertEqual(audit.batch_size, 5000)
        self.assertEqual(audit.max_inflight_batches, 1)
        self.assertEqual(audit.check_every, 0.5)
        self.assertEqual(audit.workers, 4)
        self.assertFalse(audit.cleanup_remote_files)
        self.assertEqual(audit.max_retry_passes, 2)
        self.assertEqual(audit.timeout, 600.0)
        self.assertNotIn("--max-text-chars", self.runner.arguments(audit))
        for provider in self.runner.PROVIDERS:
            with self.subTest(provider=provider):
                arguments = [
                    "LOCUS",
                    "--provider",
                    provider,
                    "--env-file",
                    str(self.env),
                ]
                corpus = self.runner.parse_args(arguments)
                self.assertEqual(corpus.dataset, "LOCUS")
                self.assertEqual(corpus.name, "LOCUS")
                self.assertEqual(
                    corpus.input,
                    (
                        self.directory / "data" / "locus_v1.1_harmonized.jsonl"
                    ).resolve(),
                )
                self.assertIsNone(corpus.expect_rows)
                self.assertEqual(corpus.max_text_chars, 16000)
                self.assertEqual(
                    corpus.batch_size, 20000 if provider == "google" else 50000
                )
                self.assertEqual(
                    corpus.max_inflight_batches,
                    {"typesafe": 1, "openai": 256, "google": 100}[provider],
                )
                self.assertEqual(corpus.check_every, 0.5)
                self.assertEqual(corpus.workers, 16)
                self.assertFalse(corpus.cleanup_remote_files)
                self.assertEqual(corpus.max_retry_passes, 2)
                self.assertEqual(
                    corpus.timeout, 45.0 if provider == "typesafe" else 600.0
                )
                no_retries = self.runner.parse_args(
                    [*arguments, "--max-retry-passes", "0"]
                )
                self.assertEqual(no_retries.max_retry_passes, 0)
                with self.assertRaises(SystemExit):
                    self.runner.parse_args([*arguments, "--max-retry-passes", "-1"])
                forwarded = self.runner.arguments(corpus)
                self.assertNotIn("--cleanup-remote-files", forwarded)
                self.assertEqual(
                    forwarded[forwarded.index("--max-text-chars") + 1], "16000"
                )
                if provider != "typesafe":
                    self.assertNotIn("--workers", forwarded)
                else:
                    self.assertEqual(
                        forwarded[forwarded.index("--workers") + 1], "16"
                    )
                if provider == "google":
                    cleanup = self.runner.parse_args(
                        [*arguments, "--cleanup-remote-files"]
                    )
                    self.assertIn(
                        "--cleanup-remote-files", self.runner.arguments(cleanup)
                    )
                else:
                    with self.assertRaises(SystemExit):
                        self.runner.parse_args([*arguments, "--cleanup-remote-files"])
                override = self.runner.parse_args(
                    [
                        *arguments,
                        "--input",
                        str(self.input),
                        "--max-text-chars",
                        "8000",
                        "--batch-size",
                        "123",
                        "--max-inflight-batches",
                        "3",
                        "--check-every",
                        "2",
                        "--workers",
                        "7",
                        "--timeout",
                        "12.5",
                    ]
                )
                self.assertEqual(override.input, self.input.resolve())
                self.assertEqual(override.max_text_chars, 8000)
                self.assertEqual(override.batch_size, 123)
                self.assertEqual(override.max_inflight_batches, 3)
                self.assertEqual(override.check_every, 2)
                self.assertEqual(override.workers, 7)
                self.assertEqual(override.timeout, 12.5)
                forwarded = self.runner.arguments(override)
                self.assertEqual(
                    forwarded[forwarded.index("--max-text-chars") + 1], "8000"
                )
                work_dir = (self.directory / "external-state").resolve()
                relocated = self.runner.parse_args(
                    [*arguments, "--work-dir", str(work_dir)]
                )
                self.assertEqual(relocated.input, corpus.input)
                self.assertEqual(
                    self.runner.ledger_path(relocated),
                    work_dir / "runs" / f"LOCUS_{provider}.sqlite3",
                )
                self.assertEqual(
                    self.runner.output_path(relocated),
                    work_dir / "outputs" / f"LOCUS_{provider}.jsonl",
                )
        for bad in ("sample", "corpus", "unknown"):
            with self.subTest(dataset=bad), self.assertRaises(SystemExit):
                self.runner.parse_args([bad])

    def test_exactly_one_provider_is_required(self):
        for arguments in (
            ["human_audit"],
            ["human_audit", "--providers", "google", "openai"],
            ["human_audit", "--provider", "all"],
            ["human_audit", "--provider", "google", "openai"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.runner.parse_args(arguments)

    def test_blank_run_names_and_malformed_model_overrides_fail_before_calls(self):
        for option, value in (
            ("--name", ""),
            ("--model", ""),
            ("--model", " bad-model "),
            ("--model", "bad\nmodel"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self.runner.parse_args([*self.args, option, value])
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))

    def test_unexportable_json_input_is_rejected_before_any_provider_is_paid(self):
        lines = self.input.read_text().splitlines()
        for metadata in ("1e400", '"\\ud800"', '{"\\udfff":"value"}'):
            lines[0] = '{"audit_id":"0","content":"text","metadata":' + metadata + "}"
            self.input.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(self.runner.main(self.args), 2)
            self.assertEqual(self.jev_client.instances, [])
            self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))

    def test_unused_typesafe_environment_does_not_affect_a_google_only_run(self):
        arguments = list(self.args)
        model_flag = arguments.index("--model")
        del arguments[model_flag : model_flag + 2]
        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test", "TYPESAFE_MODEL_ID": "invalid model"},
            clear=True,
        ):
            self.assertEqual(self.runner.main(arguments), 0)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 1))

    def test_explicit_key_environment_never_falls_back_and_only_selects_one_provider(
        self,
    ):
        for provider in self.runner.PROVIDERS:
            with (
                self.subTest(provider=provider),
                patch.object(pipeline, "iter_jsonl") as read,
            ):
                self.assertEqual(
                    self.runner.main(
                        [
                            *self.args,
                            "--provider",
                            provider,
                            "--api-key-env",
                            "ABSENT_KEY",
                        ]
                    ),
                    2,
                )
                read.assert_not_called()
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))
        with patch.dict("os.environ", {"COLLABORATOR_KEY": "test-only"}, clear=True):
            self.assertEqual(
                self.runner.main(
                    [
                        *self.args,
                        "--provider",
                        "google",
                        "--api-key-env",
                        "COLLABORATOR_KEY",
                    ]
                ),
                0,
            )
        self.assertEqual(self.openai.submissions, 0)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual(self.google.submissions, 1)

    def test_disabled_auto_retry_allows_manual_retry_but_not_after_collection_failure(
        self,
    ):
        arguments = [
            *self.args, "--provider", "openai", "--max-retry-passes", "0"
        ]
        submit = self.openai.submit

        def fail_one(*args):
            job_id = submit(*args)
            result = self.openai.jobs[job_id]["results"][0]
            self.openai.jobs[job_id]["results"][0] = {
                "custom_id": result["custom_id"],
                "error": {"code": "fixture"},
            }
            return job_id

        with patch.object(self.openai, "submit", side_effect=fail_one):
            self.assertEqual(self.runner.main(arguments), 2)
        self.assertEqual(self.google.submissions, 0)
        self.assertEqual(self.jev_client.instances, [])
        self.assertFalse((self.directory / "outputs/human_audit_openai.jsonl").exists())
        self.assertIn("rows still failed after 0 retry passes", sys.stdout.getvalue())
        self.assertNotIn("collection failed", sys.stdout.getvalue())
        # A later transport/collection failure must not be mistaken for the
        # existing row error and must not trigger new paid work.
        with patch.object(
            batch,
            "collect_results",
            side_effect=pipeline.InputError("Read interrupted"),
        ):
            self.assertEqual(
                self.runner.main(
                    [*arguments, "--retry-failed"]
                ),
                2,
            )
        self.assertEqual(self.openai.submissions, 1)
        self.assertIn("collection failed", sys.stdout.getvalue())
        self.assertEqual(
            self.runner.main([*arguments, "--retry-failed"]), 0
        )
        self.assertEqual(len(self.openai.uploads["input-2"]), 1)

    def test_native_errors_finish_the_first_pass_before_retrying_only_failed_rows(self):
        self.records = [
            {"audit_id": f"row-{i}", "content": f"Fixture rule {i}."}
            for i in range(4)
        ]
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        for provider in ("openai", "google"):
            with self.subTest(provider=provider):
                service = getattr(self, provider)
                submit = service.submit
                key = "custom_id" if provider == "openai" else "key"
                failed_id = None

                def fail_first_batch(file_id, *args):
                    nonlocal failed_id
                    if service.submissions == 2:
                        attempted = [
                            row[key]
                            for name in ("input-1", "input-2")
                            for row in service.uploads[name]
                        ]
                        self.assertEqual(len(set(attempted)), 4)
                        self.assertEqual(
                            [row[key] for row in service.uploads[file_id]], [failed_id]
                        )
                    job_id = submit(file_id, *args)
                    if service.submissions == 1:
                        result = service.jobs[job_id]["results"][0]
                        failed_id = result[key]
                        service.jobs[job_id]["results"][0] = {
                            key: failed_id,
                            "error": {"code": "temporary_row_failure"},
                        }
                    return job_id

                with patch.object(service, "submit", side_effect=fail_first_batch):
                    self.assertEqual(
                        self.runner.main(
                            [
                                *arguments,
                                "--provider", provider,
                                "--name", f"first_pass_{provider}",
                                "--batch-size", "2",
                                "--max-inflight-batches", "1",
                            ]
                        ),
                        0,
                    )
                self.assertEqual(
                    [len(rows) for rows in service.uploads.values()], [2, 2, 1]
                )
                ledger = (
                    self.directory
                    / "runs"
                    / f"first_pass_{provider}_{provider}.sqlite3"
                )
                with closing(sqlite3.connect(ledger)) as db:
                    progress = batch.summary(db)
                self.assertEqual(progress["retry_passes"], 1)
                self.assertEqual(progress["annotated"], 4)
                self.assertEqual(progress["errors"], 0)

    def test_native_retry_budget_persists_and_manual_retry_adds_only_one_pass(self):
        self.records = [
            {"audit_id": f"row-{i}", "content": f"Fixture rule {i}."}
            for i in range(3)
        ]
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        arguments += ["--batch-size", "2", "--max-inflight-batches", "1"]
        service = self.google
        submit = service.submit
        failed_id = batch.request_id(pipeline.source_identity(self.records[0], 0)[0])

        def always_fail_one(*args):
            job_id = submit(*args)
            for index, row in enumerate(service.jobs[job_id]["results"]):
                if row["key"] == failed_id:
                    service.jobs[job_id]["results"][index] = {
                        "key": failed_id, "error": {"code": "persistent_row_failure"}
                    }
            return job_id

        ledger = self.directory / "runs" / "LOCUS_google.sqlite3"
        output = self.directory / "outputs" / "LOCUS_google.jsonl"
        with patch.object(service, "submit", side_effect=always_fail_one):
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(service.submissions, 4)
            with closing(sqlite3.connect(ledger)) as db:
                self.assertEqual(batch.summary(db)["retry_passes"], 2)
                successes = db.execute(
                    "SELECT source_id,annotation_json FROM annotations "
                    "WHERE status='annotated' ORDER BY source_id"
                ).fetchall()
            self.assertFalse(output.exists())
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(service.submissions, 4)
            self.assertEqual(
                self.runner.main([*arguments, "--retry-failed"]), 2
            )
            self.assertEqual(service.submissions, 5)
            with closing(sqlite3.connect(ledger)) as db:
                self.assertEqual(batch.summary(db)["retry_passes"], 3)
        self.assertFalse(output.exists())
        self.assertEqual(self.runner.main([*arguments, "--retry-failed"]), 0)
        self.assertEqual(service.submissions, 6)
        self.assertEqual(
            [row["key"] for row in service.uploads["input-6"]], [failed_id]
        )
        self.assertEqual(len(output.read_text().splitlines()), 3)
        with closing(sqlite3.connect(ledger)) as db:
            self.assertEqual(batch.summary(db)["retry_passes"], 4)
            annotations = dict(
                db.execute("SELECT source_id,annotation_json FROM annotations")
            )
        for source_id, encoded in successes:
            self.assertEqual(annotations[source_id], encoded)

    def test_jev_row_errors_finish_then_retry_with_a_persisted_budget(self):
        from neurips2026_locus.jev import JevResponseError

        self.records = [{"content": f"rule {i}"} for i in range(3)]
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        arguments += ["--provider", "typesafe", "--workers", "1"]
        decide = self.jev_client.decide
        attempted = []

        def fail_one(client, payload):
            text = payload["state"]["excerpt"]
            attempted.append(text)
            if text == "rule 0":
                client.requests.append(payload)
                raise JevResponseError("Invalid JSON in received response")
            return decide(client, payload)

        ledger = self.directory / "runs" / "LOCUS_typesafe.sqlite3"
        output = self.directory / "outputs" / "LOCUS_typesafe.jsonl"
        with patch.object(self.jev_client, "decide", fail_one):
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(attempted.count("rule 0"), 3)
            self.assertEqual(
                attempted[:5], ["rule 0", "rule 1", "rule 1", "rule 2", "rule 2"]
            )
            self.assertEqual(attempted[5:], ["rule 0", "rule 0"])
            before = len(attempted)
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(len(attempted), before)
            self.assertEqual(
                self.runner.main([*arguments, "--retry-failed"]), 2
            )
            self.assertEqual(attempted.count("rule 0"), 4)
        self.assertFalse(output.exists())
        self.assertEqual(self.runner.main([*arguments, "--retry-failed"]), 0)
        self.assertEqual(len(output.read_text().splitlines()), 3)
        with closing(sqlite3.connect(ledger)) as db:
            self.assertEqual(
                db.execute(
                    "SELECT value FROM metadata WHERE key='row_retry_passes'"
                ).fetchone(),
                ("4",),
            )

    def test_jev_automatic_transport_passes_reuse_the_saved_classification_stage(self):
        from neurips2026_locus.jev import JevUncertainSubmissionError

        self.records = [{"content": f"rule {i}"} for i in range(3)]
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        arguments += ["--provider", "typesafe", "--workers", "1"]
        decide = self.jev_client.decide
        stages = []

        def fail_confidence(client, payload):
            excerpt = payload["state"]["excerpt"]
            confidence = "confidence_label" in payload["questions"]
            stages.append((excerpt, confidence))
            if excerpt == "rule 0" and confidence:
                client.requests.append(payload)
                raise JevUncertainSubmissionError(
                    "Bounded transport recovery exhausted",
                    transport_error="ReadError",
                    attempts=5,
                )
            return decide(client, payload)

        output = self.directory / "outputs" / "LOCUS_typesafe.jsonl"
        ledger = self.directory / "runs" / "LOCUS_typesafe.sqlite3"
        with patch.object(self.jev_client, "decide", fail_confidence):
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(stages.count(("rule 0", False)), 1)
            self.assertEqual(stages.count(("rule 0", True)), 3)
            self.assertIn(("rule 2", True), stages[:6])
            before = list(stages)
            self.assertEqual(self.runner.main(arguments), 2)
            self.assertEqual(stages, before)
        self.assertFalse(output.exists())
        with closing(sqlite3.connect(ledger)) as db:
            self.assertEqual(self.runner.read_retry_passes(db), 2)
            saved_stages = db.execute(
                "SELECT source_id,request_sha256,response_json,response_sha256 "
                "FROM responses ORDER BY request_sha256"
            ).fetchall()
        self.assertEqual(self.runner.main([*arguments, "--retry-failed"]), 0)
        requests = self.jev_client.instances[-1].requests
        self.assertEqual(len(requests), 1)
        self.assertIn("confidence_label", requests[0]["questions"])
        self.assertEqual(len(output.read_text().splitlines()), 3)
        with closing(sqlite3.connect(ledger)) as db:
            self.assertEqual(self.runner.read_retry_passes(db), 3)
            recovered = set(
                db.execute(
                    "SELECT source_id,request_sha256,response_json,response_sha256 "
                    "FROM responses"
                )
            )
        self.assertTrue(set(saved_stages).issubset(recovered))

    def test_jev_global_errors_never_enter_automatic_row_retries(self):
        from neurips2026_locus.jev import (
            JevError, JevHTTPError
        )

        self.records = [{"content": f"rule {i}"} for i in range(3)]
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        for index, error in enumerate(
            (
                JevError("Unsupported local HTTP configuration"),
                JevHTTPError(400),
                JevHTTPError(401),
                pipeline.InputError("Source hash mismatch"),
            )
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(self.jev_client, "decide", side_effect=error) as send,
            ):
                self.assertEqual(
                    self.runner.main(
                        [
                            *arguments,
                            "--provider", "typesafe",
                            "--workers", "1",
                            "--name", f"fatal_{index}",
                        ]
                    ),
                    2,
                )
                self.assertGreaterEqual(send.call_count, 1)
                self.assertLessEqual(send.call_count, 2)
                excerpts = [
                    call.args[-1]["state"]["excerpt"] for call in send.call_args_list
                ]
                self.assertEqual(len(set(excerpts)), len(excerpts))
            ledger = self.directory / "runs" / f"fatal_{index}_typesafe.sqlite3"
            with closing(sqlite3.connect(ledger)) as db:
                self.assertEqual(
                    db.execute("SELECT COUNT(*) FROM annotations").fetchone(), (0,)
                )
                self.assertEqual(self.runner.read_retry_passes(db), 0)

    def test_transient_collection_exhaustion_recovers_without_resubmitting(self):
        for provider in ("openai", "google"):
            with self.subTest(provider=provider):
                service = getattr(self, provider)
                status = service.status
                failures = 0
                now = 1000

                def transient_then_success(job_id):
                    nonlocal failures
                    if failures < 3:
                        failures += 1
                        error = RuntimeError("temporary provider service failure")
                        error.status_code = 503
                        raise error
                    return status(job_id)

                def advance_wait(args, stop, progress):
                    nonlocal now
                    now = max(now + 1, (progress.get("operation_retry_at") or now) + 1)
                    return False

                with (
                    patch.object(service, "status", side_effect=transient_then_success),
                    patch.object(batch.time, "sleep"),
                    patch.object(Event, "wait", return_value=False),
                    patch.object(batch.time, "time", side_effect=lambda: now),
                    patch.object(
                        self.runner, "wait_for_batches", side_effect=advance_wait
                    ),
                ):
                    self.assertEqual(
                        self.runner.main([*self.args, "--provider", provider]), 0
                    )
                self.assertEqual(failures, 3)
                self.assertEqual(service.submissions, 1)
                self.assertEqual(len(service.uploads), 1)
                output = self.directory / "outputs" / f"human_audit_{provider}.jsonl"
                self.assertEqual(len(output.read_text().splitlines()), 250)

    def test_operation_retry_deadline_overrides_polling_and_sooner_quota_retry(self):
        args = self.runner.parse_args([*self.args, "--check-every", "0.5"])
        stop = Mock(spec=Event)
        stop.wait.return_value = True
        with patch.object(self.runner.time, "time", return_value=1000):
            self.assertTrue(
                self.runner.wait_for_batches(
                    args,
                    stop,
                    {"next_retry_at": 1005, "operation_retry_at": 4600},
                )
            )
        stop.wait.assert_called_once_with(3600)

    def test_coordinator_waits_for_quota_then_completes_the_same_uploaded_job(self):
        service = self.openai
        submit = service.submit
        attempts = 0
        now = 1000
        deferred = []

        def quota_then_success(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise BatchTests.quota_rejection("openai")
            return submit(*args)

        def advance_wait(args, stop, progress):
            nonlocal now
            if progress.get("next_retry_at") is not None:
                deferred.append(progress["deferred_jobs"])
                now = progress["next_retry_at"] + 1
            return False

        with (
            patch.object(service, "submit", side_effect=quota_then_success),
            patch.object(batch.time, "time", side_effect=lambda: now),
            patch.object(self.runner, "wait_for_batches", side_effect=advance_wait),
        ):
            self.assertEqual(
                self.runner.main([*self.args, "--provider", "openai"]), 0
            )
        self.assertEqual(attempts, 2)
        self.assertEqual(service.submissions, 1)
        self.assertEqual(len(service.uploads), 1)
        self.assertEqual(len(deferred), 1)
        self.assertEqual(len(deferred[0]), 1)
        output = self.directory / "outputs" / "human_audit_openai.jsonl"
        self.assertEqual(len(output.read_text().splitlines()), 250)

    def test_transient_upload_failure_recovers_before_creating_a_model_job(self):
        upload = self.google.upload
        attempts = 0
        now = 1000

        def fail_then_upload(path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectTimeout("temporary upload connection failure")
            return upload(path)

        def advance_wait(args, stop, progress):
            nonlocal now
            now = max(now + 1, (progress.get("operation_retry_at") or now) + 1)
            return False

        with (
            patch.object(self.google, "upload", side_effect=fail_then_upload),
            patch.object(batch.time, "time", side_effect=lambda: now),
            patch.object(self.runner, "wait_for_batches", side_effect=advance_wait),
        ):
            self.assertEqual(self.runner.main(self.args), 0)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(self.google.uploads), 1)
        self.assertEqual(self.google.submissions, 1)
        output = self.directory / "outputs" / "human_audit_google.jsonl"
        self.assertEqual(len(output.read_text().splitlines()), 250)

    def test_dry_run_has_no_writes_or_calls(self):
        before = sorted(self.directory.iterdir())
        self.assertEqual(self.runner.main([*self.args, "--dry-run"]), 0)
        self.assertEqual(sorted(self.directory.iterdir()), before)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))

    def test_missing_local_setup_fails_before_a_large_input_scan(self):
        for missing in ("key", "dependency"):
            with (
                self.subTest(missing=missing),
                patch.object(pipeline, "iter_jsonl") as read,
            ):
                if missing == "key":
                    with patch.dict("os.environ", {}, clear=True):
                        self.assertEqual(self.runner.main(self.args), 2)
                else:
                    with patch.object(
                        self.runner.importlib.util, "find_spec", return_value=None
                    ):
                        self.assertEqual(self.runner.main(self.args), 2)
                read.assert_not_called()
        self.assertFalse((self.directory / "runs").exists())

    def test_sample_rejects_whitespace_padded_ids_before_any_calls(self):
        for key in (" ", "0 "):
            self.records[0]["audit_id"] = key
            self.write_rows()
            self.assertEqual(self.runner.main(self.args), 2)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))

    def test_preflight_reports_character_counts_and_exact_opt_in_truncation(self):
        with patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(
                self.runner.main([*self.args, "--max-text-chars", "5", "--dry-run"]), 0
            )
        report = json.loads(output.getvalue())
        lengths = [len(row["content"]) for row in self.records]
        self.assertEqual(report["excerpt_characters"], sum(lengths))
        self.assertEqual(report["maximum_excerpt_characters"], max(lengths))
        self.assertEqual(report["truncated_rows"], 250)
        self.records[0]["content"] = "x" * 16001
        self.write_rows()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        for provider in self.runner.PROVIDERS:
            with (
                self.subTest(provider=provider),
                patch("sys.stdout", new=io.StringIO()) as output,
            ):
                self.assertEqual(
                    self.runner.main(
                        [*arguments, "--provider", provider, "--dry-run"]
                    ),
                    0,
                )
                report = json.loads(output.getvalue())
                self.assertEqual(report["max_text_chars"], 16000)
                self.assertEqual(report["maximum_excerpt_characters"], 16001)
                self.assertEqual(report["truncated_rows"], 1)

    def test_wrong_size_duplicate_ids_and_prompt_overlap_fail_before_calls(self):
        self.records.pop()
        self.write_rows()
        self.assertEqual(self.runner.main(self.args), 2)
        self.records.append(dict(self.records[0]))
        self.write_rows()
        self.assertEqual(self.runner.main(self.args), 2)
        self.records[-1]["audit_id"] = "249"
        from neurips2026_locus.contract import prompt_examples

        self.records[0]["content"] = prompt_examples()[0]
        self.write_rows()
        self.assertEqual(self.runner.main(self.args), 2)
        self.assertEqual(self.jev_client.instances, [])
        self.assertFalse((self.directory / "runs").exists())

    def test_corpus_default_and_override_check_declared_count_before_calls(self):
        with (
            patch.object(pipeline, "JevClient") as typesafe_client,
            patch.object(batch.openai_batch, "OpenAIBatchClient") as openai_client,
            patch.object(batch.google_batch, "GoogleBatchClient") as google_client,
        ):
            for provider in self.runner.PROVIDERS:
                with self.subTest(provider=provider):
                    self.assertEqual(
                        self.runner.main(
                            [
                                "LOCUS",
                                "--provider",
                                provider,
                                "--env-file",
                                str(self.env),
                            ]
                        ),
                        2,
                    )
            typesafe_client.assert_not_called()
            openai_client.assert_not_called()
            google_client.assert_not_called()
        self.assertFalse((self.directory / "runs").exists())
        args = [
            "LOCUS",
            "--input",
            str(self.input),
            "--env-file",
            str(self.env),
            "--provider",
            "google",
            "--dry-run",
        ]
        self.assertEqual(self.runner.parse_args(args).expect_rows, None)
        self.assertEqual(self.runner.main(args), 0)
        default = self.directory / "data" / "locus_v1.1_harmonized.jsonl"
        default.parent.mkdir()
        default.write_bytes(self.input.read_bytes())
        for provider in self.runner.PROVIDERS:
            with self.subTest(default_provider=provider):
                self.assertEqual(
                    self.runner.main(
                        [
                            "LOCUS",
                            "--provider",
                            provider,
                            "--env-file",
                            str(self.env),
                            "--dry-run",
                        ]
                    ),
                    0,
                )
        self.assertEqual(self.runner.main([*args, "--expect-rows", "250"]), 0)
        self.assertEqual(self.runner.main([*args, "--expect-rows", "2000000"]), 2)

    def test_ctrl_c_after_upload_or_during_creation_stops_one_provider_safely(self):
        previous_handler = signal.getsignal(signal.SIGINT)
        for provider in ("google", "openai"):
            service = getattr(self, provider)
            for phase in ("upload", "submit"):
                with self.subTest(provider=provider, phase=phase):
                    before = service.submissions
                    operation = getattr(service, phase)

                    def interrupt_after_operation(*args):
                        result = operation(*args)
                        signal.raise_signal(signal.SIGINT)
                        return result

                    name = provider + "_interrupt_" + phase
                    with patch.object(
                        service, phase, side_effect=interrupt_after_operation
                    ):
                        code = self.runner.main(
                            [
                                *self.args,
                                "--provider",
                                provider,
                                "--name",
                                name,
                                "--max-inflight-batches",
                                "3",
                                "--batch-size",
                                "100",
                            ]
                        )
                    self.assertEqual(code, 130)
                    self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
                    self.assertEqual(
                        service.submissions - before, int(phase == "submit")
                    )
                    ledger = self.directory / "runs" / f"{name}_{provider}.sqlite3"
                    with closing(sqlite3.connect(ledger)) as db:
                        self.assertEqual(
                            db.execute(
                                "SELECT COUNT(*) FROM batch_jobs WHERE state='submitting'"
                            ).fetchone(),
                            (0,),
                        )
                        if phase == "submit":
                            self.assertEqual(
                                db.execute(
                                    "SELECT COUNT(*) FROM batch_jobs WHERE remote_id IS NOT NULL AND state='submitted'"
                                ).fetchone(),
                                (1,),
                            )
                    self.assertFalse(
                        (
                            self.directory / "outputs" / f"{name}_{provider}.jsonl"
                        ).exists()
                    )
                    # A subsequent invocation reuses a created job and submits
                    # only the remaining rows; an upload alone can be retried.
                    self.assertEqual(
                        self.runner.main(
                            [
                                *self.args,
                                "--provider",
                                provider,
                                "--name",
                                name,
                                "--max-inflight-batches",
                                "3",
                                "--batch-size",
                                "100",
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(service.submissions - before, 3)

    def test_locus_exports_only_document_section_ids_and_complete_annotations(self):
        document_id = "doc_" + "0" * 24
        source_metadata = {
            "document_id": document_id,
            "file_path": "/private/corpus/source.pdf",
            "state": "al",
            "vendor": "fixture-vendor",
            "document_kind": "code",
            "jurisdiction": "Fixture town",
            "jurisdiction_geoid": "0100123",
            "jurisdiction_fips": "0100123",
            "geography_type": "place",
            "city_county": "Fixture town",
            "county_fips": "01001",
            "county_names": "Fixture County",
            "located_in_county_fips": "01001",
            "located_in_county_names": "Fixture County",
            "shared_by_governments": 1,
            "heading_path": ["CHAPTER 1 / GENERAL"],
            "pages": [1, 2],
            "words": 4,
            "flags": ["HARMONIZED_METADATA_NOT_MODEL_INPUT"],
        }
        self.records = [
            {
                **source_metadata,
                "index": 17,
                "header": "STRUCTURAL-FIXTURE: Chapter 1",
                "content": "",
            },
            {
                **source_metadata,
                "index": 23,
                "header": "License requirement",
                "content": "A license is required: SOURCE_TEXT_NOT_EXPORTED.",
            },
        ]
        self.write_rows()
        source_bytes = self.input.read_bytes()
        arguments = [*self.args]
        arguments[0] = "LOCUS"
        for provider in self.runner.PROVIDERS:
            with self.subTest(provider=provider):
                name = "corpus_" + provider
                self.assertEqual(
                    self.runner.main(
                        [
                            *arguments,
                            "--provider",
                            provider,
                            "--name",
                            name,
                            "--expect-rows",
                            "2",
                        ]
                    ),
                    0,
                )
                output = self.directory / "outputs" / f"{name}_{provider}.jsonl"
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                field = pipeline.PROVIDER_ANNOTATION_FIELDS[provider]
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    [(row["document_id"], row["index"]) for row in rows],
                    [(document_id, 17), (document_id, 23)],
                )
                ledger = self.directory / "runs" / f"{name}_{provider}.sqlite3"
                with closing(sqlite3.connect(ledger)) as db:
                    stored = {
                        source_id: json.loads(encoded)
                        for source_id, encoded in db.execute(
                            "SELECT source_id,annotation_json FROM annotations"
                        )
                    }
                for index, (source, row) in enumerate(zip(self.records, rows)):
                    self.assertEqual(set(row), {"document_id", "index", field})
                    source_id, digest = pipeline.source_identity(source, index)
                    self.assertEqual(row[field], stored[source_id])
                    self.assertEqual(row[field]["source"]["sha256"], digest)
                    self.assertEqual(row[field]["final"]["confidence"], "High")
                if provider == "typesafe":
                    payloads = [
                        request
                        for client in self.jev_client.instances
                        for request in client.requests
                    ]
                else:
                    payloads = list(getattr(self, provider).uploads.values())
                self.assertNotIn(
                    "HARMONIZED_METADATA_NOT_MODEL_INPUT", json.dumps(payloads)
                )
                self.assertNotIn("SOURCE_TEXT_NOT_EXPORTED", output.read_text())
                self.assertNotIn("/private/corpus", output.read_text())
                self.assertEqual(self.input.read_bytes(), source_bytes)


class CSVRunnerTests(unittest.TestCase):
    """The human CSV supplies text; its annotations never become model input."""

    write_rows = RunnerTests.write_rows

    def setUp(self):
        RunnerTests.setUp(self)
        self.component = self.directory / "project" / "llm_annotation"
        self.component.mkdir(parents=True)
        (self.component / "pyproject.toml").write_text("# Source-checkout marker\n")
        self.stack.enter_context(patch.object(self.runner, "ROOT", self.component))
        self.csv_path = (
            self.component.parent
            / "human_annotation"
            / "LOCUS_snorkel_labels.csv"
        )
        self.csv_path.parent.mkdir()
        self.csv_rows = [
            {
                "audit_id": row["audit_id"],
                "text": row["content"],
                "function": "HUMAN_ONLY_NOT_MODEL_INPUT",
                "topic": "HUMAN_ONLY_NOT_MODEL_INPUT",
                "confidence": "HUMAN_ONLY_NOT_MODEL_INPUT",
                "function_eligible": "0",
                "ec1_notes": "HUMAN_ONLY_NOT_MODEL_INPUT",
            }
            for row in self.records
        ]
        self.write_csv()
        self.args = [
            "human_audit",
            "--env-file",
            str(self.env),
            "--provider",
            "google",
            "--model",
            "fake-jev-1",
            "--check-every",
            "0.00001",
        ]

    def write_csv(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.csv_rows[0]))
            writer.writeheader()
            writer.writerows(self.csv_rows)

    def cache_path(self, provider="google", name="human_audit"):
        return self.component / "runs" / f"{name}_{provider}.input.jsonl"

    def test_csv_default_projects_all_text_without_human_labels_for_each_provider(self):
        self.csv_rows[0]["text"] = 'Quoted "words", café, and commas.\nAnother rule.'
        self.write_csv()
        work_dir = (self.directory / "external-state").resolve()
        for provider in self.runner.PROVIDERS:
            with self.subTest(provider=provider):
                arguments = [
                    *self.args,
                    "--provider",
                    provider,
                    "--work-dir",
                    str(work_dir),
                ]
                self.assertEqual(
                    self.runner.main(arguments), 0
                )
                cache = work_dir / "runs" / f"human_audit_{provider}.input.jsonl"
                rows = [json.loads(line) for line in cache.read_text().splitlines()]
                self.assertEqual(len(rows), 250)
                for source, projected in zip(self.csv_rows, rows):
                    self.assertEqual(
                        projected,
                        {
                            "audit_id": source["audit_id"],
                            "header": "",
                            "content": source["text"],
                        },
                    )
                self.assertEqual(
                    cache.read_text(),
                    "".join(pipeline.canonical_json(row) + "\n" for row in rows),
                )
                if provider == "typesafe":
                    payloads = [
                        request
                        for client in self.jev_client.instances
                        for request in client.requests
                    ]
                    self.assertEqual(len(payloads), 500)
                else:
                    payloads = list(getattr(self, provider).uploads.values())
                    self.assertEqual(len(payloads[-1]), 250)
                self.assertNotIn("HUMAN_ONLY_NOT_MODEL_INPUT", json.dumps(payloads))
                output = work_dir / "outputs" / f"human_audit_{provider}.jsonl"
                published = [
                    json.loads(line) for line in output.read_text().splitlines()
                ]
                self.assertEqual(len(published), 250)
                field = pipeline.PROVIDER_ANNOTATION_FIELDS[provider]
                self.assertEqual(
                    [row["audit_id"] for row in published],
                    [row["audit_id"] for row in self.csv_rows],
                )
                for row in published:
                    self.assertEqual(set(row), {"audit_id", field})
                self.assertNotIn(self.csv_rows[0]["text"], output.read_text())
                self.assertNotIn("HUMAN_ONLY_NOT_MODEL_INPUT", output.read_text())
                before = (
                    sum(len(client.requests) for client in self.jev_client.instances),
                    self.openai.submissions,
                    self.google.submissions,
                )
                self.assertEqual(
                    self.runner.main(arguments), 0
                )
                self.assertEqual(
                    (
                        sum(
                            len(client.requests)
                            for client in self.jev_client.instances
                        ),
                        self.openai.submissions,
                        self.google.submissions,
                    ),
                    before,
                )
        self.assertFalse((self.component / "runs").exists())
        self.assertFalse((self.component / "outputs").exists())

    def test_csv_dry_runs_do_not_materialize_inputs_or_other_files(self):
        def snapshot():
            return {
                str(path.relative_to(self.directory)): path.read_bytes()
                for path in self.directory.rglob("*")
                if path.is_file()
            }

        before = snapshot()
        for provider in self.runner.PROVIDERS:
            self.assertEqual(
                self.runner.main([*self.args, "--provider", provider, "--dry-run"]), 0
            )
            self.assertEqual(snapshot(), before)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))

    def test_bad_csv_fails_before_materializing_or_paying(self):
        original = json.loads(json.dumps(self.csv_rows))
        for case in (
            "missing_text_column",
            "duplicate_header",
            "missing_value",
            "extra_value",
            "unterminated_quote",
            "duplicate_id",
            "padded_id",
            "blank_text",
            "wrong_count",
            "invalid_utf8",
        ):
            with self.subTest(case=case):
                self.csv_rows = json.loads(json.dumps(original))
                if case == "duplicate_id":
                    self.csv_rows[-1]["audit_id"] = self.csv_rows[0]["audit_id"]
                elif case == "padded_id":
                    self.csv_rows[0]["audit_id"] += " "
                elif case == "blank_text":
                    self.csv_rows[0]["text"] = " \n "
                elif case == "wrong_count":
                    self.csv_rows.pop()
                self.write_csv()
                if case in {
                    "missing_text_column",
                    "duplicate_header",
                    "missing_value",
                    "extra_value",
                    "unterminated_quote",
                }:
                    stream = io.StringIO(newline="")
                    writer = csv.writer(stream, lineterminator="\n")
                    if case == "missing_text_column":
                        writer.writerow(["audit_id", "not_text"])
                        writer.writerows(
                            (row["audit_id"], row["text"]) for row in self.csv_rows
                        )
                    elif case == "duplicate_header":
                        writer.writerow(["audit_id", "text", "audit_id"])
                        writer.writerows(
                            (row["audit_id"], row["text"], row["audit_id"])
                            for row in self.csv_rows
                        )
                    else:
                        writer.writerow(["audit_id", "text"])
                        writer.writerows(
                            (row["audit_id"], row["text"])
                            for row in self.csv_rows[:-1]
                        )
                        last = self.csv_rows[-1]
                        if case == "missing_value":
                            writer.writerow([last["audit_id"]])
                        elif case == "extra_value":
                            writer.writerow([last["audit_id"], last["text"], "EXTRA"])
                        else:
                            stream.write(last["audit_id"] + ',"unterminated text')
                    self.csv_path.write_text(stream.getvalue(), encoding="utf-8")
                elif case == "invalid_utf8":
                    self.csv_path.write_bytes(b"audit_id,text\n0,\xff\n")
                self.assertEqual(self.runner.main(self.args), 2)
                self.assertEqual(self.jev_client.instances, [])
                self.assertEqual(
                    (self.openai.submissions, self.google.submissions), (0, 0)
                )
                self.assertFalse((self.component / "runs").exists())

    def test_label_only_changes_resume_but_text_or_order_changes_do_not(self):
        self.assertEqual(self.runner.main(self.args), 0)
        cache_before = self.cache_path().read_bytes()
        self.csv_rows[0]["function"] = "A_CHANGED_HUMAN_LABEL"
        self.csv_rows[0]["ec1_notes"] = "New human notes only"
        self.write_csv()
        self.assertEqual(self.runner.main(self.args), 0)
        self.assertEqual(self.google.submissions, 1)
        self.assertEqual(self.cache_path().read_bytes(), cache_before)
        original = json.loads(json.dumps(self.csv_rows))
        for change in ("text", "order"):
            with self.subTest(change=change):
                self.csv_rows = json.loads(json.dumps(original))
                if change == "text":
                    self.csv_rows[0]["text"] += " Changed law."
                else:
                    self.csv_rows[0], self.csv_rows[1] = (
                        self.csv_rows[1],
                        self.csv_rows[0],
                    )
                self.write_csv()
                with patch.object(batch.google_batch, "GoogleBatchClient") as client:
                    self.assertEqual(self.runner.main(self.args), 2)
                client.assert_not_called()
                self.assertEqual(self.google.submissions, 1)
                self.assertEqual(self.cache_path().read_bytes(), cache_before)

    def test_csv_changes_during_or_after_preflight_stop_before_materialization(self):
        for phase in ("reading", "materializing"):
            with self.subTest(phase=phase):
                self.write_csv()
                if phase == "reading":
                    target = "_human_csv_records"
                    original = self.runner._human_csv_records

                    def change_csv(*args, **kwargs):
                        records = original(*args, **kwargs)
                        with self.csv_path.open("a") as handle:
                            handle.write("\n")
                        return records

                else:
                    target = "_materialize_human_input"
                    original = self.runner._materialize_human_input

                    def change_csv(*args, **kwargs):
                        with self.csv_path.open("a") as handle:
                            handle.write("\n")
                        return original(*args, **kwargs)

                with patch.object(self.runner, target, side_effect=change_csv):
                    self.assertEqual(self.runner.main(self.args), 2)
                self.assertEqual(self.jev_client.instances, [])
                self.assertEqual(
                    (self.openai.submissions, self.google.submissions), (0, 0)
                )
                self.assertFalse((self.component / "runs").exists())

    def test_missing_csv_fails_before_paying(self):
        self.csv_path.unlink()
        self.assertEqual(self.runner.main(self.args), 2)
        self.assertEqual(self.jev_client.instances, [])
        self.assertEqual((self.openai.submissions, self.google.submissions), (0, 0))
        self.assertFalse((self.component / "runs").exists())


if __name__ == "__main__":
    unittest.main()
