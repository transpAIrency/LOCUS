"""Offline contract tests for the resumable Jev annotation pipeline.

These tests replace the network client with a deterministic typed-response
fixture.  They exercise the release guarantees that matter for a large run:
Structural rows are retained, historical header-only exclusions are explicit,
and a SQLite ledger cannot be accidentally reused for incompatible work.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
import json
import io
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurips2026_locus import pipeline  # noqa: E402
from neurips2026_locus.files import opaque_document_id  # noqa: E402
from neurips2026_locus.jev import JevResponse  # noqa: E402


class FakeJevClient:
    """A local Jev substitute that makes no HTTP requests."""

    instances: list["FakeJevClient"] = []

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str,
        timeout_seconds: float,
        max_attempts: int,
        stop_event=None,
    ) -> None:
        self.endpoint = endpoint
        self.requests: list[dict[str, object]] = []
        type(self).instances.append(self)

    def decide(self, payload: dict[str, object]) -> JevResponse:
        self.requests.append(payload)
        state = payload["state"]
        assert isinstance(state, dict)
        text = str(state["excerpt"])
        if "structural-fixture" in text.casefold():
            primary_function = "Structural"
            sub_category = "Other"
        else:
            primary_function = "Rules"
            sub_category = "Other"
        body = {
            "model": "fake-jev-1",
            "answers": {
                "primary_function": {
                    "choice": primary_function,
                    "confidence": 0.99,
                },
                "subcategory": {
                    "choice": sub_category,
                    "confidence": 0.98,
                },
            },
        }
        if "confidence_label" in payload["questions"]:
            body["answers"] = {
                "confidence_label": {"choice": "High", "confidence": 0.01}
            }
        return JevResponse(
            body=body,
            request_sha256=sha256(
                pipeline.canonical_json(payload).encode("utf-8")
            ).hexdigest(),
            response_sha256=sha256(
                pipeline.canonical_json(body).encode("utf-8")
            ).hexdigest(),
        )

    def close(self):
        """Match the pooled transport's lifecycle without opening a socket."""
        pass


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def annotation_kwargs(
    input_path: Path,
    ledger_path: Path,
    *,
    resume: bool = False,
    model: str = "fake-jev-1",
) -> dict[str, object]:
    return {
        "input_path": input_path,
        "ledger_path": ledger_path,
        "resume": resume,
        "model": model,
        "endpoint": "https://fake.invalid/v1/systemone",
        "api_key": "test-key",
        "timeout_seconds": 1.0,
        "max_attempts": 1,
        "header_only_policy": "exclude",
        "max_text_chars": None,
        "max_records": None,
    }


class PipelineTests(unittest.TestCase):
    def test_transport_hash_mismatch_is_rejected_before_cache_or_confidence(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        decide = FakeJevClient.decide

        def corrupt_hash(client, payload):
            response = decide(client, payload)
            return JevResponse(response.body, "0" * 64, response.response_sha256)

        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", corrupt_hash),
        ):
            with self.assertRaisesRegex(pipeline.InputError, "request hash"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        self.assertEqual(len(FakeJevClient.instances[-1].requests), 1)
        with closing(sqlite3.connect(self.ledger_path)) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM responses").fetchone(), (0,)
            )

    def test_public_export_scrubs_review_directories_and_unlabelled_absolute_metadata(
        self,
    ):
        legal = "An owner may refer to /Users/example in an example or https://example.org/rules."
        record = {
            "header": legal,
            "content": legal,
            "classification": {"logic": legal},
            "review_run_dir": "/Users/private/run-123",
            "review_directories": ["../runs/a"],
            "metadata": {
                "working_folder": "C:\\Private\\run",
                "unknown": "file:///private/run",
                "nested": [{"review_root": "/private/run"}],
                "url": "https://example.org/rules",
                "opaque": "unchanged",
            },
        }
        before = json.loads(json.dumps(record))
        identity = pipeline.source_identity(record, 0)
        result = pipeline.public_source_record(record)
        self.assertNotIn("review_run_dir", result)
        self.assertNotIn("review_directories", result)
        self.assertNotIn("working_folder", result["metadata"])
        self.assertNotIn("review_root", result["metadata"]["nested"][0])
        self.assertNotIn("file:///private", json.dumps(result["metadata"]))
        self.assertEqual(result["metadata"]["url"], record["metadata"]["url"])
        self.assertEqual(result["metadata"]["opaque"], "unchanged")
        self.assertNotIn("document_id", result)
        self.assertEqual(result["content"], legal)
        self.assertEqual(result["header"], legal)
        self.assertEqual(result["classification"], record["classification"])
        self.assertEqual(record, before)
        self.assertEqual(pipeline.source_identity(record, 0), identity)

    def test_verified_release_ledgers_resume_without_rewriting_their_fingerprints(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        self._annotate()
        for old in (
            "fe48aee9b30b0b70d0bf7974b45fc549c2a05625fbb4e99eec2bf85d261252b6",
            "7ad95a0934acce30a75a547b01e0f43a6ab14fc2bb82ef34a1b0a9fb8b7305d1",
        ):
            with closing(sqlite3.connect(self.ledger_path)) as db, db:
                db.execute(
                    "UPDATE metadata SET value=? WHERE key='request_builder_sha256'",
                    (old,),
                )
            self.assertEqual(self._annotate(resume=True)["already_complete"], 1)
            self.assertEqual(FakeJevClient.instances[-1].requests, [])
            with closing(sqlite3.connect(self.ledger_path)) as db:
                self.assertEqual(
                    db.execute(
                        "SELECT value FROM metadata WHERE key='request_builder_sha256'"
                    ).fetchone(),
                    (old,),
                )

    def test_long_retry_delay_is_a_global_stop_not_a_failed_legal_row(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(
                FakeJevClient,
                "decide",
                side_effect=pipeline.RetryDelayExceeded("Resume later"),
            ),
        ):
            with self.assertRaisesRegex(pipeline.InputError, "Resume later"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        with closing(sqlite3.connect(self.ledger_path)) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM annotations").fetchone(), (0,)
            )
        self.assertEqual(self._annotate(resume=True)["annotated"], 1)

    def test_export_permissions_respect_normal_file_creation_and_restrictive_umask(
        self,
    ):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        self._annotate()
        script = "import os,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from neurips2026_locus.pipeline import export_augmented_jsonl; os.umask(int(sys.argv[2],8)); export_augmented_jsonl(input_path=Path(sys.argv[3]),ledger_path=Path(sys.argv[4]),output_path=Path(sys.argv[5]),include_excluded_header_only=False,allow_partial=False,overwrite=False)"
        for mask, expected in (("022", 0o644), ("077", 0o600)):
            target = self.directory / f"export_{mask}.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    script,
                    str(PROJECT_ROOT / "src"),
                    mask,
                    str(self.input_path),
                    str(self.ledger_path),
                    str(target),
                ],
                check=True,
                timeout=10,
            )
            self.assertEqual(target.stat().st_mode & 0o777, expected)

    def test_direct_cli_ctrl_c_checkpoints_without_starting_confidence_or_export(self):
        output_path = self.directory / "interrupted.jsonl"
        write_jsonl(
            self.input_path, [{"content": "First rule."}, {"content": "Second rule."}]
        )
        entered, release = Event(), Event()
        real_wait = pipeline.wait
        interrupted = False
        previous_handler = signal.getsignal(signal.SIGINT)

        class BlockingClient(FakeJevClient):
            def decide(self, payload):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test did not release the running request")
                return super().decide(payload)

        def interrupt_once(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                self.assertTrue(entered.wait(timeout=5))
                interrupted = True
                signal.raise_signal(signal.SIGINT)
                release.set()
            return real_wait(*args, **kwargs)

        with (
            patch.object(pipeline, "JevClient", BlockingClient),
            patch.object(pipeline, "wait", side_effect=interrupt_once),
            patch.object(pipeline, "load_dotenv"),
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-only"}, clear=True),
            patch("sys.stderr", new=io.StringIO()),
        ):
            code = pipeline.main(
                [
                    "--provider",
                    "typesafe",
                    "--input",
                    str(self.input_path),
                    "--ledger",
                    str(self.ledger_path),
                    "--output",
                    str(output_path),
                    "--allow-full-run",
                    "--workers",
                    "1",
                    "--model",
                    "fake-jev-1",
                ]
            )
        self.assertEqual(code, 130)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
        self.assertEqual(len(FakeJevClient.instances[-1].requests), 1)
        self.assertFalse(output_path.exists())
        with closing(sqlite3.connect(self.ledger_path)) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM responses").fetchone(), (1,)
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM annotations").fetchone(), (0,)
            )
        with patch.object(pipeline, "JevClient", FakeJevClient):
            resumed = pipeline.annotate(
                **dict(
                    annotation_kwargs(self.input_path, self.ledger_path, resume=True),
                    endpoint=pipeline.DEFAULT_ENDPOINT,
                    header_only_policy="include",
                )
            )
        self.assertEqual(resumed["annotated"], 2)

    def test_keyboard_interrupt_cancels_queued_rows_and_checkpoints_running_stage(self):
        write_jsonl(
            self.input_path, [{"content": "First rule."}, {"content": "Second rule."}]
        )
        entered, release = Event(), Event()
        real_wait = pipeline.wait
        interrupted = False

        class BlockingClient(FakeJevClient):
            def decide(self, payload):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test did not release the running request")
                return super().decide(payload)

        def interrupt_once(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                self.assertTrue(entered.wait(timeout=5))
                interrupted = True
                raise KeyboardInterrupt()
            release.set()
            return real_wait(*args, **kwargs)

        with (
            patch.object(pipeline, "JevClient", BlockingClient),
            patch.object(pipeline, "wait", side_effect=interrupt_once),
        ):
            with self.assertRaises(KeyboardInterrupt):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        requests = FakeJevClient.instances[-1].requests
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["state"]["excerpt"], "First rule.")
        with closing(sqlite3.connect(self.ledger_path)) as db:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM responses").fetchone(), (1,)
            )
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM annotations").fetchone(), (0,)
            )
        result = self._annotate(resume=True)
        self.assertEqual(result["annotated"], 2)
        self.assertEqual(len(FakeJevClient.instances[-1].requests), 3)

    def test_unknown_request_rejection_stops_but_is_not_resent_on_resume(self):
        """An unrecognized 400 may be global, so stop and retain its row outcome."""
        from neurips2026_locus.jev import JevHTTPError

        write_jsonl(self.input_path, [{"content": f"rule {i}"} for i in range(10)])
        original = FakeJevClient.decide
        rejected = []

        def decide(client, payload):
            if payload["state"]["excerpt"] == "rule 3":
                rejected.append(1)
                raise JevHTTPError(400)
            return original(client, payload)

        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", decide),
        ):
            with self.assertRaises(pipeline.InputError):
                pipeline.annotate(**kwargs)
            result = pipeline.annotate(**dict(kwargs, resume=True))
        self.assertEqual(len(rejected), 1)
        self.assertEqual(result["failed"], 1)
        with closing(sqlite3.connect(self.ledger_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT status,COUNT(*) FROM annotations GROUP BY status"
                ).fetchall(),
                [("annotated", 9), ("failed", 1)],
            )

    def test_authentication_failure_does_not_become_a_failed_data_row(self):
        from neurips2026_locus.jev import JevHTTPError

        write_jsonl(self.input_path, [{"content": "A rule."}])
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", side_effect=JevHTTPError(401)),
        ):
            with self.assertRaises(pipeline.InputError):
                pipeline.annotate(**kwargs)
        with closing(sqlite3.connect(self.ledger_path)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone(), (0,)
            )
        self.assertEqual(self._annotate(resume=True)["annotated"], 1)

    def test_changed_request_builders_block_new_calls_but_allow_export(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        self._annotate()
        with (
            patch.object(pipeline, "request_builder_sha256", return_value="changed"),
            patch.object(pipeline, "JevClient") as client,
        ):
            with self.assertRaisesRegex(
                pipeline.InputError, "Request-building code changed"
            ):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path, resume=True)
                )
            client.assert_not_called()
            result = pipeline.export_augmented_jsonl(
                input_path=self.input_path,
                ledger_path=self.ledger_path,
                output_path=self.directory / "export.jsonl",
                include_excluded_header_only=False,
                allow_partial=False,
                overwrite=False,
            )
            self.assertEqual(result["exported"], 1)

    def test_rejected_row_is_durable_and_does_not_block_the_other_39_rows(self):
        """A resume skips rejections; only explicit retry sends the failed row."""
        from neurips2026_locus.jev import JevHTTPError

        write_jsonl(self.input_path, [{"content": f"rule {i}"} for i in range(40)])
        decide = FakeJevClient.decide
        rejected = []

        def reject_one(client, payload):
            if payload["state"]["excerpt"] == "rule 3":
                rejected.append(payload)
                raise JevHTTPError(400, "max_tokens_exceeded")
            return decide(client, payload)

        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", reject_one),
        ):
            first = pipeline.annotate(**kwargs, workers=4)
            resumed = pipeline.annotate(**dict(kwargs, resume=True), workers=4)
        self.assertEqual((first["annotated"], first["failed"]), (39, 1))
        self.assertEqual((resumed["already_complete"], resumed["failed"]), (39, 1))
        self.assertEqual(len(rejected), 1)
        with self.assertRaisesRegex(pipeline.InputError, "failed rows"):
            pipeline.export_augmented_jsonl(
                input_path=self.input_path,
                ledger_path=self.ledger_path,
                output_path=self.directory / "output.jsonl",
                overwrite=False,
                include_excluded_header_only=False,
                allow_partial=False,
            )
        with patch.object(pipeline, "JevClient", FakeJevClient):
            retried = pipeline.annotate(
                **dict(kwargs, resume=True), workers=4, retry_failed=True
            )
        self.assertEqual((retried["annotated"], retried["failed"]), (1, 0))
        with closing(sqlite3.connect(self.ledger_path)) as database:
            self.assertEqual(
                database.execute(
                    "SELECT status, COUNT(*) FROM annotations GROUP BY status"
                ).fetchall(),
                [("annotated", 40)],
            )

    def test_repeated_rejections_stop_scheduling_without_retrying_saved_failures(self):
        """A shared invalid request cannot produce an unbounded rejection loop."""
        from neurips2026_locus.jev import JevHTTPError

        write_jsonl(self.input_path, [{"content": f"rule {i}"} for i in range(40)])
        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(
                FakeJevClient,
                "decide",
                side_effect=JevHTTPError(400, "max_tokens_exceeded"),
            ) as send,
        ):
            with self.assertRaises(pipeline.InputError):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path), workers=4
                )
        self.assertGreaterEqual(send.call_count, pipeline.MAX_CONSECUTIVE_REJECTIONS)
        self.assertLess(send.call_count, 40)

    def setUp(self) -> None:
        output = patch("sys.stdout", new=io.StringIO())
        output.start()
        self.addCleanup(output.stop)
        FakeJevClient.instances = []
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "input.jsonl"
        self.ledger_path = self.directory / "run.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _annotate(
        self, *, resume: bool = False, model: str = "fake-jev-1"
    ) -> dict[str, int]:
        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            return pipeline.annotate(
                **annotation_kwargs(
                    self.input_path, self.ledger_path, resume=resume, model=model
                )
            )

    def test_explicit_env_file_must_exist_but_default_is_optional(self) -> None:
        missing = self.directory / "absent.env"
        with patch("sys.stderr", new=io.StringIO()) as error:
            result = pipeline.main(
                ["--input", "unused.jsonl", "--dry-run", "--env-file", str(missing)]
            )
        self.assertEqual(result, 2)
        self.assertIn("Environment file not found", error.getvalue())
        # The default does not depend on the working directory.
        self.assertEqual(pipeline.resolve_env_file(None), PROJECT_ROOT / ".env")

    def test_provider_options_and_default(self) -> None:
        parser = pipeline.build_parser()
        self.assertIn("--provider {typesafe,openai,google}", parser.format_help())
        self.assertEqual(
            parser.parse_args(["--input", "unused.jsonl"]).provider, "typesafe"
        )
        for provider in ("typesafe", "openai", "google"):
            with self.subTest(provider=provider):
                args = parser.parse_args(
                    ["--provider", provider, "--input", "unused.jsonl"]
                )
                self.assertEqual(args.provider, provider)
        with (
            patch("sys.stderr", new=io.StringIO()),
            self.assertRaises(SystemExit) as caught,
        ):
            parser.parse_args(["--provider", "jev", "--input", "unused.jsonl"])
        self.assertEqual(caught.exception.code, 2)

    def test_all_provider_dry_runs_are_offline(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        for provider, model in (
            ("typesafe", "jev-1.13.0"),
            ("openai", "gpt-6-luna"),
            ("google", "gemini-3.8-flash"),
        ):
            with (
                self.subTest(provider=provider),
                patch("sys.stdout", new=io.StringIO()) as output,
            ):
                with (
                    patch.object(
                        pipeline,
                        "load_dotenv",
                        side_effect=AssertionError("must not load keys"),
                    ),
                    patch.object(
                        pipeline,
                        "annotate",
                        side_effect=AssertionError("must not annotate"),
                    ),
                ):
                    result = pipeline.main(
                        [
                            "--provider",
                            provider,
                            "--input",
                            str(self.input_path),
                            "--model",
                            model,
                            "--dry-run",
                        ]
                    )
                self.assertEqual(result, 0)
                preview = json.loads(output.getvalue())
                self.assertEqual(preview["provider"], provider)
                requested_model = (
                    preview["request"]["model"]
                    if provider == "typesafe"
                    else preview["config"]["model"]
                )
                self.assertEqual(requested_model, model)
                self.assertNotIn("A license is required.", output.getvalue())
        self.assertEqual(list(self.directory.iterdir()), [self.input_path])

    def test_typesafe_cli_uses_typesafe_default_ledger_and_existing_key(self) -> None:
        with (
            patch.object(pipeline, "load_dotenv"),
            patch.object(
                pipeline, "get_api_key", return_value=("test-key", "TYPESAFE_API_KEY")
            ) as key,
            patch.object(pipeline, "annotate", return_value={"failed": 0}) as annotate,
            patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(
                pipeline.main(
                    [
                        "--provider",
                        "typesafe",
                        "--input",
                        str(self.input_path),
                        "--max-records",
                        "1",
                        "--model",
                        "jev-1.13.0",
                    ]
                ),
                0,
            )
        key.assert_called_once_with(None)
        self.assertEqual(
            annotate.call_args.kwargs["ledger_path"],
            PROJECT_ROOT / "runs/locus_typesafe.sqlite3",
        )
        self.assertEqual(annotate.call_args.kwargs["model"], "jev-1.13.0")

    def test_explicit_cli_key_variable_does_not_fall_back_to_a_default_account(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        with (
            patch.dict(os.environ, {"TYPESAFE_API_KEY": "default-account"}, clear=True),
            patch.object(pipeline, "load_dotenv"),
            patch.object(pipeline, "annotate") as annotate,
            patch("sys.stderr", new=io.StringIO()),
        ):
            self.assertEqual(
                pipeline.main(
                    [
                        "--input",
                        str(self.input_path),
                        "--max-records",
                        "1",
                        "--api-key-env",
                        "MISSING_COLLABORATOR_KEY",
                    ]
                ),
                2,
            )
        annotate.assert_not_called()
        self.assertFalse(self.ledger_path.exists())

    def test_structural_row_is_preserved_in_augmented_export(self) -> None:
        original_structural_classification = {
            "is_substantive": 0,
            "primary_function": "Structural",
            "sub_category": None,
            "logic": "Original legacy label.",
        }
        write_jsonl(
            self.input_path,
            [
                {
                    "document_id": "structural-1",
                    "header": "Contents",
                    "content": "STRUCTURAL-FIXTURE: table of contents and source note.",
                    "classification": original_structural_classification,
                },
                {
                    "document_id": "rule-1",
                    "content": "No person may operate without a license.",
                },
            ],
        )

        counts = self._annotate()
        output_path = self.directory / "augmented.jsonl"
        export_counts = pipeline.export_augmented_jsonl(
            input_path=self.input_path,
            ledger_path=self.ledger_path,
            output_path=output_path,
            include_excluded_header_only=False,
            allow_partial=False,
            overwrite=False,
        )

        self.assertEqual(
            counts,
            {
                "scanned": 2,
                "annotated": 2,
                "excluded_header_only": 0,
                "already_complete": 0,
                "failed": 0,
            },
        )
        self.assertEqual(
            export_counts,
            {"exported": 2, "excluded_header_only": 0, "missing": 0, "failed": 0},
        )
        output_rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        structural_row = next(
            row for row in output_rows if row["document_id"] == "structural-1"
        )
        self.assertEqual(
            structural_row["classification"], original_structural_classification
        )
        self.assertEqual(
            structural_row["classification_jev"]["final"]["primary_function"],
            "Structural",
        )
        self.assertEqual(
            structural_row["classification_jev"]["final"]["confidence"], "High"
        )
        self.assertIsNone(structural_row["classification_jev"]["final"]["subcategory"])

    def test_header_only_rows_are_explicitly_excluded_without_a_model_call(
        self,
    ) -> None:
        write_jsonl(
            self.input_path,
            [
                {"document_id": "header-only", "header": "CHAPTER 1", "content": ""},
                {
                    "document_id": "structural-content",
                    "header": "Source record",
                    "content": "STRUCTURAL-FIXTURE: history and source note.",
                },
            ],
        )

        counts = self._annotate()
        output_path = self.directory / "augmented.jsonl"
        export_counts = pipeline.export_augmented_jsonl(
            input_path=self.input_path,
            ledger_path=self.ledger_path,
            output_path=output_path,
            include_excluded_header_only=False,
            allow_partial=False,
            overwrite=False,
        )

        self.assertEqual(
            counts,
            {
                "scanned": 2,
                "annotated": 1,
                "excluded_header_only": 1,
                "already_complete": 0,
                "failed": 0,
            },
        )
        self.assertEqual(len(FakeJevClient.instances), 1)
        self.assertEqual(len(FakeJevClient.instances[0].requests), 2)
        self.assertEqual(
            export_counts,
            {"exported": 1, "excluded_header_only": 1, "missing": 0, "failed": 0},
        )
        output_rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [row["document_id"] for row in output_rows], ["structural-content"]
        )
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            statuses = dict(
                connection.execute(
                    "SELECT status, COUNT(*) FROM annotations GROUP BY status"
                )
            )
        self.assertEqual(statuses, {"annotated": 1, "header_only_excluded": 1})

    def test_resuming_a_compatible_ledger_never_duplicates_rows_or_calls_jev_again(
        self,
    ) -> None:
        write_jsonl(
            self.input_path,
            [
                {"document_id": "one", "content": "A license is required."},
                {"document_id": "two", "content": "STRUCTURAL-FIXTURE: source note."},
            ],
        )

        first_counts = self._annotate()
        resumed_counts = self._annotate(resume=True)

        self.assertEqual(first_counts["annotated"], 2)
        self.assertEqual(
            resumed_counts,
            {
                "scanned": 2,
                "annotated": 0,
                "excluded_header_only": 0,
                "already_complete": 2,
                "failed": 0,
            },
        )
        self.assertEqual(len(FakeJevClient.instances), 2)
        self.assertEqual(len(FakeJevClient.instances[0].requests), 4)
        self.assertEqual(len(FakeJevClient.instances[1].requests), 0)
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            row_count = connection.execute(
                "SELECT COUNT(*) FROM annotations"
            ).fetchone()[0]
            unique_id_count = connection.execute(
                "SELECT COUNT(DISTINCT source_id) FROM annotations"
            ).fetchone()[0]
        self.assertEqual(row_count, 2)
        self.assertEqual(unique_id_count, 2)

    def test_bounded_workers_commit_each_completed_row_once(self) -> None:
        write_jsonl(
            self.input_path,
            [
                {"document_id": "one", "content": "A license is required."},
                {"document_id": "two", "content": "STRUCTURAL-FIXTURE: source note."},
                {"document_id": "three", "content": "No conduct is allowed."},
            ],
        )
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        kwargs["workers"] = 2

        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            counts = pipeline.annotate(**kwargs)

        self.assertEqual(counts["annotated"], 3)
        self.assertEqual(len(FakeJevClient.instances), 1)
        self.assertEqual(len(FakeJevClient.instances[0].requests), 6)
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 3
            )

    def test_bounded_selection_stays_fixed_on_resume_and_never_parses_beyond_it(self):
        self.input_path.write_text('{"content":"First rule."}\nnot valid JSON\n')
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        kwargs["max_records"] = 1
        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            self.assertEqual(pipeline.annotate(**kwargs)["annotated"], 1)
            self.assertEqual(
                pipeline.annotate(**dict(kwargs, resume=True))["already_complete"], 1
            )
            with self.assertRaisesRegex(pipeline.InputError, "metadata does not match"):
                pipeline.annotate(**dict(kwargs, resume=True, max_records=2))
        self.assertEqual(
            sum(len(client.requests) for client in FakeJevClient.instances), 2
        )
        output = self.directory / "bounded.jsonl"
        result = pipeline.export_augmented_jsonl(
            input_path=self.input_path,
            ledger_path=self.ledger_path,
            output_path=output,
            include_excluded_header_only=False,
            allow_partial=False,
            overwrite=False,
        )
        self.assertEqual(
            result,
            {"exported": 1, "excluded_header_only": 0, "missing": 0, "failed": 0},
        )
        self.assertEqual(json.loads(output.read_text())["content"], "First rule.")

    def test_ledger_setup_failures_always_close_the_connection(self):
        for operation in ("execute", "commit"):
            connection = Mock()
            getattr(connection, operation).side_effect = sqlite3.OperationalError(
                "fixture disk failure"
            )
            with (
                self.subTest(operation=operation),
                patch.object(pipeline.sqlite3, "connect", return_value=connection),
            ):
                with self.assertRaises(sqlite3.OperationalError):
                    pipeline.AnnotationLedger(
                        self.ledger_path, run_metadata={}, resume=False
                    )
            connection.close.assert_called_once_with()

    def test_official_alias_pins_the_confidence_stage_to_the_first_response(self):
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        self._annotate(model="jev-latest")
        requests = FakeJevClient.instances[-1].requests
        self.assertEqual(requests[0]["model"], "jev-latest")
        self.assertEqual(requests[1]["model"], "fake-jev-1")

    def test_response_must_resolve_a_requested_alias_to_a_model_version(self):
        class UnresolvedAliasClient(FakeJevClient):
            def decide(self, request):
                response = super().decide(request)
                response.body["model"] = "jev-latest"
                return response

        write_jsonl(self.input_path, [{"content": "A license is required."}])
        with patch("neurips2026_locus.pipeline.JevClient", UnresolvedAliasClient):
            with self.assertRaisesRegex(pipeline.InputError, "resolve its alias"):
                pipeline.annotate(
                    **dict(
                        annotation_kwargs(self.input_path, self.ledger_path),
                        model="jev-latest",
                    )
                )

    def test_audit_id_is_retained_in_source_provenance(self) -> None:
        self.assertEqual(
            pipeline.source_locator({"audit_id": "LOCUS_AUDIT_0408"}, 7),
            {"input_row_index": 7, "audit_id": "LOCUS_AUDIT_0408"},
        )

    def test_document_ids_have_the_declared_collection_namespace(self):
        locator = "va/municode/counties/fixture_county.jsonl"
        expected = "doc_" + sha256(("original|" + locator).encode()).hexdigest()[:24]
        self.assertEqual(opaque_document_id(locator), expected)
        self.assertNotEqual(
            expected, opaque_document_id(locator, collection="ocr_addition")
        )

    def test_public_locator_keeps_the_raw_identity_recipe_and_input_untouched(self):
        record = {
            "header": "Rule",
            "content": "A license is required.",
            "source_file": "/private/source/va/municode/counties/fixture_county.jsonl",
            "document_id": "/Users/example/private/code.pdf",
            "custom_id": "private/row.jsonl:4",
            "source_record_id": "private/raw.csv:4",
            "audit_id": "LOCUS_AUDIT_0001",
        }
        expected_locator = {
            "input_row_index": 7,
            "document_id": record["document_id"],
            "audit_id": record["audit_id"],
            "source_file": record["source_file"],
            "custom_id": record["custom_id"],
        }
        basis = {
            "locator": expected_locator,
            "header": record["header"],
            "content": record["content"],
        }
        digest = sha256(pipeline.canonical_json(basis).encode()).hexdigest()
        before = json.loads(json.dumps(record))
        self.assertEqual(
            pipeline.source_identity(record, 7), ("locus-v1:" + digest, digest)
        )
        public = pipeline.public_source_record(record)
        self.assertEqual(
            public["document_id"],
            opaque_document_id("va/municode/counties/fixture_county.jsonl"),
        )
        self.assertNotIn("source_file", public)
        for key in ["document_id", "custom_id", "source_record_id"]:
            self.assertRegex(public[key], r"^doc_[0-9a-f]{24}$")
        self.assertEqual(public["audit_id"], record["audit_id"])
        self.assertEqual(pipeline.source_locator(record, 7), expected_locator)
        self.assertEqual(record, before)

    def test_public_sanitizer_does_not_rewrite_legal_text_or_classification(self):
        record = {
            "content": "The directory /var/log/ is defined here.",
            "header": "Filename foo.jsonl in a quotation",
            "classification": {
                "logic": "Source text mentions /private/example.",
                "primary_function": "Rules",
            },
            "document_id": "stable-document-1",
            "custom_id": "stable-row-1",
            "metadata": {"source_file": "/private/source/file.jsonl"},
        }
        public = pipeline.public_source_record(record)
        for key in ["content", "header", "classification", "document_id", "custom_id"]:
            self.assertEqual(public[key], record[key])
        self.assertNotIn("source_file", public["metadata"])
        self.assertRegex(public["metadata"]["document_id"], r"^doc_[0-9a-f]{24}$")

    def test_public_sanitizer_handles_nested_locator_aliases_and_keeps_urls(self):
        aliases = [
            "input_file",
            "original_file",
            "source_filename",
            "pdf_file",
            "archive_path",
        ]
        for alias in aliases:
            record = {
                "content": "A license is required.",
                "classification": {"logic": "Unchanged source classification."},
                "metadata": {
                    "entries": [{alias: "/private/collected/foo.jsonl"}],
                    "download_file": "https://example.org/public/code.pdf",
                },
            }
            before = json.loads(json.dumps(record))
            identity = pipeline.source_identity(record, 0)
            public = pipeline.public_source_record(record)
            self.assertNotIn(alias, public["metadata"]["entries"][0])
            self.assertRegex(
                public["metadata"]["entries"][0]["document_id"], r"^doc_[0-9a-f]{24}$"
            )
            self.assertEqual(
                public["metadata"]["download_file"],
                "https://example.org/public/code.pdf",
            )
            self.assertEqual(public["classification"], record["classification"])
            self.assertEqual(public["content"], record["content"])
            self.assertEqual(pipeline.source_identity(record, 0), identity)
            self.assertEqual(record, before)

    def test_public_additional_alias_matches_the_document_inventory_namespace(self):
        locator = "new/mtas/tn/charters/shared.pdf"
        for metadata in [
            {"collection": "additional"},
            {"collection": "additional", "origin": "ocr_addition"},
            {"collection": "unrecognized", "origin": "ocr_addition"},
        ]:
            record = {
                "content": "A charter provision.",
                "source_file": locator,
                **metadata,
            }
            identity = pipeline.source_identity(record, 3)
            expected = opaque_document_id(locator, collection="ocr_addition")
            self.assertEqual(
                pipeline.public_source_record(record)["document_id"], expected
            )
            self.assertEqual(
                pipeline.public_source_locator(record, 3)["document_id"], expected
            )
            self.assertEqual(pipeline.source_identity(record, 3), identity)

    def test_public_sanitizer_recurses_through_plural_and_list_locator_aliases(self):
        first = "/private/collected/va/municode/counties/fixture_county.jsonl"
        second = "/private/collected/another.pdf"
        url = "HTTPS://example.org/public/code.pdf"
        quotation = "The directory /private/example is defined in this rule."
        classification = {"logic": quotation, "primary_function": "Rules"}
        for alias in (
            "source_files",
            "input_files",
            "original_files",
            "source_filenames",
            "pdf_files",
            "archive_paths",
            "file_names",
            "source_file",
            "document_ids",
            "source_record_ids",
            "custom_id",
        ):
            with self.subTest(alias=alias):
                record = {
                    "content": quotation,
                    "source_file": url,
                    "metadata": {
                        alias: [
                            first,
                            {
                                "primary": second,
                                "content": quotation,
                                "classification": classification,
                            },
                            [url, first],
                        ],
                    },
                }
                before = json.loads(json.dumps(record))
                identity = pipeline.source_identity(record, 4)
                public = pipeline.public_source_record(record)
                values = public["metadata"][alias]
                first_id = opaque_document_id(
                    "va/municode/counties/fixture_county.jsonl"
                )
                self.assertEqual(values[0], first_id)
                self.assertEqual(values[1]["primary"], opaque_document_id(second))
                self.assertEqual(values[1]["content"], quotation)
                self.assertEqual(values[1]["classification"], classification)
                self.assertEqual(values[2], [url, first_id])
                self.assertEqual(public["content"], quotation)
                self.assertEqual(public["source_file"], url)
                self.assertEqual(pipeline.source_identity(record, 4), identity)
                self.assertEqual(record, before)

    def test_public_sanitizer_replaces_locator_mapping_keys_without_losing_values(self):
        locator = "va/municode/counties/fixture_county.jsonl"
        url = "https://example.org/public/code.pdf"
        quotation = "The directory /private/example is defined in this rule."
        classification = {locator: {"logic": quotation}}
        record = {
            "content": quotation,
            "classification": classification,
            "metadata": {
                "source_files": {
                    locator: {"pages": [1, 2]},
                    url: {"pages": [3]},
                    "supplement": {"/private/appendix.pdf": {"pages": [4]}},
                    "content": quotation,
                    "classification": classification,
                }
            },
        }
        before = json.loads(json.dumps(record))
        identity = pipeline.source_identity(record, 5)
        public = pipeline.public_source_record(record)
        expected_keys = [
            opaque_document_id(locator),
            url,
            "supplement",
            "content",
            "classification",
        ]
        public_files = public["metadata"]["source_files"]
        self.assertEqual(list(public_files), expected_keys)
        self.assertEqual(public_files[expected_keys[0]], {"pages": [1, 2]})
        self.assertEqual(public_files[url], {"pages": [3]})
        self.assertEqual(
            public_files["supplement"],
            {opaque_document_id("/private/appendix.pdf"): {"pages": [4]}},
        )
        self.assertEqual(public_files["content"], quotation)
        self.assertEqual(public_files["classification"], classification)
        self.assertEqual(public["content"], quotation)
        self.assertEqual(public["classification"], classification)
        self.assertEqual(pipeline.source_identity(record, 5), identity)
        self.assertEqual(record, before)

    def test_public_sanitizer_rejects_colliding_locator_mapping_keys(self):
        locator = "va/municode/counties/fixture_county.jsonl"
        for duplicate in (
            "/private/source/" + locator,
            opaque_document_id(locator),
        ):
            with self.subTest(duplicate=duplicate):
                record = {
                    "content": "A license is required.",
                    "metadata": {
                        "source_files": {
                            locator: {"pages": [1]},
                            duplicate: {"pages": [2]},
                        }
                    },
                }
                before = json.loads(json.dumps(record))
                identity = pipeline.source_identity(record, 6)
                with self.assertRaisesRegex(
                    pipeline.InputError, "colliding document-ID keys"
                ):
                    pipeline.public_source_record(record)
                self.assertEqual(pipeline.source_identity(record, 6), identity)
                self.assertEqual(record, before)

    def test_export_redacts_source_paths_without_mutating_ledger_annotations(self):
        record = {
            "header": "Rule",
            "content": "A license is required.",
            "source_file": "/private/source/va/municode/counties/fixture_county.jsonl",
            "document_id": "/private/files/code.pdf",
            "custom_id": "private/row.jsonl:4",
            "source_record_id": "private/source.csv:4",
            "source_document_slug": "charles_city_county.jsonl",
            "metadata": {
                "source_files": [
                    "/private/appendix.pdf",
                    {"alternate": "/private/appendix.txt"},
                ]
            },
            "classification": {
                "primary_function": "Rules",
                "sub_category": "Other",
                "is_substantive": 1,
            },
        }
        write_jsonl(self.input_path, [record])
        original_bytes = self.input_path.read_bytes()
        identity = pipeline.source_identity(record, 0)
        self._annotate()
        with closing(sqlite3.connect(self.ledger_path)) as db:
            ledger_before = db.execute(
                "SELECT annotation_json FROM annotations WHERE source_id=?",
                (identity[0],),
            ).fetchone()[0]
        output = self.directory / "public.jsonl"
        pipeline.export_augmented_jsonl(
            input_path=self.input_path,
            ledger_path=self.ledger_path,
            output_path=output,
            include_excluded_header_only=False,
            allow_partial=False,
            overwrite=False,
        )
        public = json.loads(output.read_text())
        self.assertNotIn("source_file", public)
        self.assertNotIn("source_document_slug", public)
        self.assertEqual(public["classification"], record["classification"])
        self.assertEqual(public["content"], record["content"])
        self.assertEqual(public["header"], record["header"])
        self.assertEqual(
            public["metadata"]["source_files"],
            [
                opaque_document_id("/private/appendix.pdf"),
                {"alternate": opaque_document_id("/private/appendix.txt")},
            ],
        )
        self.assertEqual(public["classification_jev"]["source"]["id"], identity[0])
        self.assertNotIn(
            "source_file", public["classification_jev"]["source"]["locator"]
        )
        self.assertNotIn("/private", output.read_text())
        self.assertNotIn(".jsonl", output.read_text())
        with closing(sqlite3.connect(self.ledger_path)) as db:
            self.assertEqual(
                db.execute(
                    "SELECT annotation_json FROM annotations WHERE source_id=?",
                    (identity[0],),
                ).fetchone()[0],
                ledger_before,
            )
        self.assertEqual(self.input_path.read_bytes(), original_bytes)

    def test_resume_rejects_configuration_that_does_not_match_the_ledger(self) -> None:
        write_jsonl(
            self.input_path,
            [{"document_id": "one", "content": "A license is required."}],
        )
        self._annotate()

        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            with self.assertRaisesRegex(pipeline.InputError, "metadata does not match"):
                pipeline.annotate(
                    **annotation_kwargs(
                        self.input_path,
                        self.ledger_path,
                        resume=True,
                        model="fake-jev-incompatible-version",
                    )
                )

        # Reject mismatched metadata before even constructing a provider client.
        self.assertEqual(len(FakeJevClient.instances), 1)
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 1
            )

    def test_export_rejects_changed_reference_fields_even_when_legal_text_matches(
        self,
    ) -> None:
        row = {
            "audit_id": "one",
            "content": "A license is required.",
            "human_reference": "before",
        }
        write_jsonl(self.input_path, [row])
        self._annotate()
        row["human_reference"] = "after"
        write_jsonl(self.input_path, [row])
        output = self.directory / "output.jsonl"
        with self.assertRaisesRegex(pipeline.InputError, "input hash"):
            pipeline.export_augmented_jsonl(
                input_path=self.input_path,
                ledger_path=self.ledger_path,
                output_path=output,
                include_excluded_header_only=False,
                allow_partial=True,
                overwrite=False,
            )
        self.assertFalse(output.exists())

    def test_overwrite_flag_cannot_replace_input_or_ledger(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        self._annotate()
        input_before = self.input_path.read_bytes()
        for output in (self.input_path, self.ledger_path):
            with (
                self.subTest(output=output),
                self.assertRaisesRegex(pipeline.InputError, "different files"),
            ):
                pipeline.export_augmented_jsonl(
                    input_path=self.input_path,
                    ledger_path=self.ledger_path,
                    output_path=output,
                    include_excluded_header_only=False,
                    allow_partial=False,
                    overwrite=True,
                )
        self.assertEqual(self.input_path.read_bytes(), input_before)
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 1
            )

    def test_malformed_later_row_does_not_discard_completed_paid_work(self) -> None:
        self.input_path.write_text(
            '{"content": "A license is required."}\ninvalid\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(pipeline.InputError, "Invalid JSON"):
            self._annotate()
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 1
            )

    def test_failed_confidence_resumes_from_checkpointed_first_pass(self) -> None:
        class FailingConfidenceClient(FakeJevClient):
            def decide(self, payload):
                if "classification" in payload["state"]:
                    self.requests.append(payload)
                    raise RuntimeError("confidence fixture failure")
                return super().decide(payload)

        write_jsonl(self.input_path, [{"content": "A license is required."}])
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        with patch("neurips2026_locus.pipeline.JevClient", FailingConfidenceClient):
            with self.assertRaisesRegex(
                pipeline.InputError, "confidence fixture failure"
            ):
                pipeline.annotate(**kwargs)
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM annotations").fetchone()[0], 0
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0], 1
            )
        kwargs["resume"] = True
        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            result = pipeline.annotate(**kwargs)
        self.assertEqual(result["annotated"], 1)
        self.assertEqual(len(FakeJevClient.instances[-1].requests), 1)
        self.assertIn(
            "classification", FakeJevClient.instances[-1].requests[0]["state"]
        )

    def test_export_only_needs_no_api_key(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        self._annotate()
        output = self.directory / "output.jsonl"
        with patch(
            "neurips2026_locus.pipeline.get_api_key",
            side_effect=AssertionError("must not load a key"),
        ):
            with patch("sys.stdout", new=io.StringIO()):
                result = pipeline.main(
                    [
                        "--provider",
                        "typesafe",
                        "--input",
                        str(self.input_path),
                        "--ledger",
                        str(self.ledger_path),
                        "--output",
                        str(output),
                        "--export-only",
                    ]
                )
        self.assertEqual(result, 0)
        self.assertTrue(output.exists())

    def test_export_cannot_relabel_a_different_providers_ledger(self) -> None:
        record = {"content": "A license is required."}
        write_jsonl(self.input_path, [record])
        source_id, source_digest = pipeline.source_identity(record, 0)
        annotation = {
            "final": {
                "primary_function": "Rules",
                "is_substantive": 1,
                "sub_category": "Business",
                "logic": "A license is required.",
            }
        }
        for provider in ("google", "openai", "jev"):
            with self.subTest(provider=provider):
                ledger_path = self.directory / f"{provider}.sqlite3"
                metadata = {
                    "ledger_schema_version": pipeline.LEDGER_SCHEMA_VERSION,
                    "input_sha256": pipeline.sha256_file(self.input_path),
                }
                # Old Jev ledgers have no provider field and must remain readable.
                if provider != "jev":
                    metadata["provider"] = provider
                ledger = pipeline.AnnotationLedger(
                    ledger_path, run_metadata=metadata, resume=False
                )
                try:
                    ledger.insert(
                        source_id=source_id,
                        source_index=0,
                        source_sha256=source_digest,
                        status="annotated",
                        annotation=annotation,
                    )
                finally:
                    ledger.close()
                output = self.directory / f"{provider}.jsonl"
                wrong_provider = "google" if provider == "jev" else "jev"
                with self.assertRaisesRegex(
                    pipeline.InputError, "Export provider does not match"
                ):
                    pipeline.export_augmented_jsonl(
                        input_path=self.input_path,
                        ledger_path=ledger_path,
                        output_path=output,
                        include_excluded_header_only=False,
                        allow_partial=False,
                        overwrite=False,
                        annotation_field=f"classification_{wrong_provider}",
                    )
                self.assertFalse(output.exists())
                pipeline.export_augmented_jsonl(
                    input_path=self.input_path,
                    ledger_path=ledger_path,
                    output_path=output,
                    include_excluded_header_only=False,
                    allow_partial=False,
                    overwrite=False,
                    annotation_field=f"classification_{provider}",
                )
                exported = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(exported[f"classification_{provider}"], annotation)
                self.assertNotIn(f"classification_{wrong_provider}", exported)

    def test_incomplete_export_keeps_existing_output_unchanged(self) -> None:
        write_jsonl(
            self.input_path, [{"content": "First rule."}, {"content": "Second rule."}]
        )
        self._annotate()
        # Model an interrupted unbounded run with one final checkpoint missing.
        with closing(sqlite3.connect(self.ledger_path)) as db, db:
            db.execute("DELETE FROM annotations WHERE source_index=1")
        output = self.directory / "output.jsonl"
        output.write_text("previous complete export\n", encoding="utf-8")
        with self.assertRaisesRegex(pipeline.InputError, "incomplete"):
            pipeline.export_augmented_jsonl(
                input_path=self.input_path,
                ledger_path=self.ledger_path,
                output_path=output,
                include_excluded_header_only=False,
                allow_partial=False,
                overwrite=True,
            )
        self.assertEqual(
            output.read_text(encoding="utf-8"), "previous complete export\n"
        )

    def test_bounded_export_counts_eligible_rows_and_preserves_explicit_exclusions(
        self,
    ):
        write_jsonl(
            self.input_path,
            [
                {"header": "A heading", "content": ""},
                {"content": "First rule."},
                {"content": "Second rule."},
            ],
        )
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        kwargs["max_records"] = 1
        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            pipeline.annotate(**kwargs)
        output = self.directory / "selected.jsonl"
        counts = pipeline.export_augmented_jsonl(
            input_path=self.input_path,
            ledger_path=self.ledger_path,
            output_path=output,
            include_excluded_header_only=True,
            allow_partial=False,
            overwrite=False,
        )
        self.assertEqual(counts["exported"], 2)
        self.assertEqual(
            [json.loads(line)["content"] for line in output.read_text().splitlines()],
            ["", "First rule."],
        )

    def test_source_change_during_export_does_not_publish_or_replace_a_result(self):
        write_jsonl(self.input_path, [{"content": "First rule."}])
        self._annotate()
        output = self.directory / "stable.jsonl"
        output.write_text("previous result\n")
        original = pipeline.iter_jsonl

        def changed_after_read(path):
            yield from original(path)
            with path.open("a") as handle:
                handle.write("\n")

        with patch.object(pipeline, "iter_jsonl", changed_after_read):
            with self.assertRaisesRegex(pipeline.InputError, "changed during export"):
                pipeline.export_augmented_jsonl(
                    input_path=self.input_path,
                    ledger_path=self.ledger_path,
                    output_path=output,
                    include_excluded_header_only=False,
                    allow_partial=False,
                    overwrite=True,
                )
        self.assertEqual(output.read_text(), "previous result\n")
        self.assertEqual(list(self.directory.glob(".stable.jsonl.*.tmp")), [])

    def test_export_revalidates_current_labels_and_source_identity_from_the_ledger(
        self,
    ):
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        self._annotate()
        output = self.directory / "published.jsonl"
        with closing(sqlite3.connect(self.ledger_path)) as db:
            source_id, original = db.execute(
                "SELECT source_id, annotation_json FROM annotations"
            ).fetchone()
            for field in ("final", "source", "contract_version"):
                changed = json.loads(original)
                if field == "final":
                    changed["final"]["subcategory"] = "Housing"
                elif field == "source":
                    changed["source"]["id"] = "different-source"
                else:
                    # A row must not opt out of validation by removing its version.
                    changed.pop("contract_version")
                db.execute(
                    "UPDATE annotations SET annotation_json=? WHERE source_id=?",
                    (json.dumps(changed), source_id),
                )
                db.commit()
                with self.subTest(field=field), self.assertRaises(pipeline.InputError):
                    pipeline.export_augmented_jsonl(
                        input_path=self.input_path,
                        ledger_path=self.ledger_path,
                        output_path=output,
                        include_excluded_header_only=False,
                        allow_partial=False,
                        overwrite=False,
                    )
                self.assertFalse(output.exists())
            db.execute(
                "UPDATE annotations SET annotation_json=? WHERE source_id=?",
                (original, source_id),
            )
            db.commit()

    def test_export_preserves_a_destination_created_during_export(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        self._annotate()
        output = self.directory / "output.jsonl"
        original_link = pipeline.os.link

        def create_destination_then_link(source, destination):
            output.write_text("concurrent output\n", encoding="utf-8")
            return original_link(source, destination)

        with patch(
            "neurips2026_locus.pipeline.os.link",
            side_effect=create_destination_then_link,
        ):
            with self.assertRaises(FileExistsError):
                pipeline.export_augmented_jsonl(
                    input_path=self.input_path,
                    ledger_path=self.ledger_path,
                    output_path=output,
                    include_excluded_header_only=False,
                    allow_partial=False,
                    overwrite=False,
                )
        self.assertEqual(output.read_text(encoding="utf-8"), "concurrent output\n")
        self.assertEqual(list(self.directory.glob(".output.jsonl.*.tmp")), [])

    def test_current_exclusions_are_validated_against_their_recorded_source(self):
        write_jsonl(self.input_path, [{"header": "Chapter heading", "content": ""}])
        self._annotate()
        output = self.directory / "excluded.jsonl"
        with closing(sqlite3.connect(self.ledger_path)) as db:
            source_id, original = db.execute(
                "SELECT source_id, annotation_json FROM annotations"
            ).fetchone()
            for field in ("source", "status", "contract_version"):
                changed = json.loads(original)
                if field == "source":
                    changed[field]["id"] = "different-row"
                else:
                    changed.pop(field)
                db.execute(
                    "UPDATE annotations SET annotation_json=? WHERE source_id=?",
                    (json.dumps(changed), source_id),
                )
                db.commit()
                with self.subTest(field=field), self.assertRaises(pipeline.InputError):
                    pipeline.export_augmented_jsonl(
                        input_path=self.input_path,
                        ledger_path=self.ledger_path,
                        output_path=output,
                        include_excluded_header_only=True,
                        allow_partial=False,
                        overwrite=False,
                    )
                self.assertFalse(output.exists())

    def test_resume_rejects_missing_and_unknown_schema_ledgers(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        with self.assertRaisesRegex(pipeline.InputError, "does not exist"):
            self._annotate(resume=True)
        self._annotate()
        with closing(sqlite3.connect(self.ledger_path)) as connection, connection:
            connection.execute(
                "UPDATE metadata SET value = '999' WHERE key = 'schema_version'"
            )
        with self.assertRaisesRegex(pipeline.InputError, "schema version"):
            self._annotate(resume=True)

    def test_nontext_content_and_existing_annotations_are_rejected_before_calls(
        self,
    ) -> None:
        for row in (
            {"content": {"text": "not a legal text string"}},
            {"content": "rule", "classification_jev": {}},
        ):
            with self.subTest(row=row), self.assertRaises(pipeline.InputError):
                pipeline.source_identity(row, 0)

    def test_request_does_not_expose_human_or_legacy_labels(self) -> None:
        write_jsonl(
            self.input_path,
            [
                {
                    "content": "A license is required.",
                    "human_reference": {"primary_function": "Rules"},
                    "classification": {"primary_function": "Process"},
                    "human_notes": "Private gold annotation",
                }
            ],
        )
        self._annotate()
        self.assertEqual(
            FakeJevClient.instances[-1].requests[0]["state"],
            {"excerpt": "A license is required."},
        )

    def test_dry_run_honors_explicit_truncation(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        with patch("sys.stdout", new=io.StringIO()) as output:
            pipeline.dry_run(
                self.input_path,
                model="fake-jev-1",
                header_only_policy="exclude",
                max_text_chars=5,
            )
        preview = json.loads(output.getvalue())
        self.assertEqual(preview["sent_text_characters"], 5)
        self.assertTrue(preview["text_truncated"])
        self.assertEqual(
            pipeline.limit_text("A license is required.", 5), ("A lic", True)
        )

    def test_unexpected_response_model_is_not_accepted(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        with self.assertRaisesRegex(pipeline.InputError, "but Jev returned"):
            self._annotate(model="different-requested-model")

    def test_stop_event_leaves_unstarted_rows_for_resume(self):
        write_jsonl(
            self.input_path, [{"content": "First rule."}, {"content": "Second rule."}]
        )
        stop = Event()
        stop.set()
        kwargs = annotation_kwargs(self.input_path, self.ledger_path)
        with patch("neurips2026_locus.pipeline.JevClient", FakeJevClient):
            self.assertEqual(
                pipeline.annotate(**kwargs, stop_event=stop)["annotated"], 0
            )
        self.assertEqual(FakeJevClient.instances[-1].requests, [])
        self.assertEqual(self._annotate(resume=True)["annotated"], 2)

    def test_input_changed_after_preflight_stops_before_client_creation(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        with patch("neurips2026_locus.pipeline.JevClient") as client:
            with self.assertRaisesRegex(pipeline.InputError, "changed since"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path),
                    expected_input_sha256="not-the-input-hash",
                )
        client.assert_not_called()

    def test_input_change_while_hashing_stops_before_creating_a_client_or_ledger(self):
        write_jsonl(self.input_path, [{"content": "A rule."}])
        original_hash = pipeline.sha256_file

        def change_after_hash(path):
            digest = original_hash(path)
            path.write_bytes(path.read_bytes() + b"\n")
            return digest

        with (
            patch.object(pipeline, "sha256_file", side_effect=change_after_hash),
            patch.object(pipeline, "JevClient") as client,
        ):
            with self.assertRaisesRegex(pipeline.InputError, "Input changed"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        client.assert_not_called()
        self.assertFalse(self.ledger_path.exists())

    def test_input_change_during_a_call_stops_scheduling_and_keeps_completed_work(self):
        write_jsonl(self.input_path, [{"content": f"A rule {i}."} for i in range(4)])
        original_bytes = self.input_path.read_bytes()
        original_decide = FakeJevClient.decide
        changed = Event()

        def change_during_call(client, payload):
            response = original_decide(client, payload)
            if "confidence_label" in payload["questions"] and not changed.is_set():
                self.input_path.write_bytes(original_bytes + b"\n")
                changed.set()
            return response

        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", change_during_call),
        ):
            with self.assertRaisesRegex(pipeline.InputError, "Input changed"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        with closing(sqlite3.connect(self.ledger_path)) as db:
            retained = db.execute(
                "SELECT COUNT(*) FROM annotations WHERE status='annotated'"
            ).fetchone()[0]
        self.assertGreater(retained, 0)
        self.assertLess(retained, 4)
        self.input_path.write_bytes(original_bytes)
        result = self._annotate(resume=True)
        self.assertEqual(result["already_complete"], retained)
        self.assertEqual(result["annotated"], 4 - retained)

    def test_input_change_after_reader_finishes_is_reported_and_checkpointed(self):
        write_jsonl(self.input_path, [{"content": "One rule."}])
        original_bytes = self.input_path.read_bytes()
        original_decide = FakeJevClient.decide

        def change_during_call(client, payload):
            response = original_decide(client, payload)
            if "confidence_label" in payload["questions"]:
                self.input_path.write_bytes(original_bytes + b"\n")
            return response

        with (
            patch.object(pipeline, "JevClient", FakeJevClient),
            patch.object(FakeJevClient, "decide", change_during_call),
        ):
            with self.assertRaisesRegex(pipeline.InputError, "Input changed"):
                pipeline.annotate(
                    **annotation_kwargs(self.input_path, self.ledger_path)
                )
        self.input_path.write_bytes(original_bytes)
        self.assertEqual(self._annotate(resume=True)["already_complete"], 1)

    def test_overlapping_annotation_run_is_rejected_before_any_paid_call(self) -> None:
        entered = Event()
        released = Event()

        class BlockingClient(FakeJevClient):
            def decide(self, payload):
                entered.set()
                if not released.wait(timeout=5):
                    raise RuntimeError("Test did not release the first request")
                return super().decide(payload)

        write_jsonl(self.input_path, [{"content": "A license is required."}])
        with patch("neurips2026_locus.pipeline.JevClient", BlockingClient):
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(
                    pipeline.annotate,
                    **annotation_kwargs(self.input_path, self.ledger_path),
                )
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    with self.assertRaisesRegex(pipeline.InputError, "already in use"):
                        pipeline.annotate(
                            **annotation_kwargs(
                                self.input_path, self.ledger_path, resume=True
                            )
                        )
                finally:
                    released.set()
                self.assertEqual(first.result(timeout=5)["annotated"], 1)
        self.assertEqual(
            sum(len(client.requests) for client in FakeJevClient.instances), 2
        )
        self.assertEqual(self._annotate(resume=True)["already_complete"], 1)

    def test_process_exit_releases_the_lock_without_deleting_the_lock_file(
        self,
    ) -> None:
        lock_path = Path(str(self.ledger_path.resolve()) + ".lock.sqlite3")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sqlite3, sys; c = sqlite3.connect(sys.argv[1]); "
                "c.execute('BEGIN EXCLUSIVE'); os._exit(0)",
                str(lock_path),
            ],
            check=True,
            timeout=5,
        )
        self.assertTrue(lock_path.exists())
        connection = pipeline.acquire_annotation_lock(self.ledger_path)
        connection.close()

    def test_lock_paths_cannot_alias_source_output_or_ledger(self) -> None:
        write_jsonl(self.input_path, [{"content": "A license is required."}])
        lock_path = Path(str(self.ledger_path.resolve()) + ".lock.sqlite3")
        with self.assertRaisesRegex(pipeline.InputError, "different files"):
            pipeline.distinct_paths(self.input_path, self.ledger_path, lock_path)
        with self.assertRaisesRegex(pipeline.InputError, "sidecar path"):
            pipeline.distinct_paths(
                self.input_path, self.ledger_path, Path(str(lock_path) + "-journal")
            )
        self._annotate()
        alias = self.directory / "aliased-ledger.sqlite3"
        os.link(self.ledger_path, alias)
        with self.assertRaisesRegex(pipeline.InputError, "Hard-linked ledgers"):
            pipeline.acquire_annotation_lock(alias)


if __name__ == "__main__":
    unittest.main()
