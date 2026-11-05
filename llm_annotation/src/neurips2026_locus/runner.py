#!/usr/bin/env python3
"""Annotate human_audit (250 excerpts) or LOCUS (an explicit corpus input).

Select exactly one provider per invocation. Native batches are bounded in flight;
Typesafe uses a bounded worker pool. Repeating the same command resumes its ledger.
"""

from __future__ import annotations

import argparse
from concurrent.futures import CancelledError
from contextlib import closing
import csv
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time

from . import batch, jev, pipeline
from .contract import (
    ContractError,
    compose_input_text,
    contract_sha256,
    example_overlap,
)
from .municipal_law_annotation.municipal_law_jev_adapter import build_jev_request
from .files import atomic_text_output, project_root

# Source checkouts use their own data/runs directories. Installed wheels use
# the working directory, where the caller supplies the external data files.
ROOT = project_root()

PROVIDERS = ("typesafe", "openai", "google")
SDK_MODULES = {"typesafe": "httpx", "openai": "openai", "google": "google.genai"}


def default_human_csv():
    """Use the human annotation release beside this source component."""
    base = ROOT.parent if (ROOT / "pyproject.toml").is_file() else ROOT
    return base / "human_annotation/LOCUS_snorkel_labels.csv"


def parse_args(argv=None):
    """Resolve one explicit dataset and validate local selection/run controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=("human_audit", "LOCUS"),
        help="human_audit: 250 audit excerpts; LOCUS: every row of --input.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="human_audit defaults to human_annotation/LOCUS_snorkel_labels.csv; LOCUS requires JSONL. Explicit human-audit JSONL remains supported.",
    )
    parser.add_argument(
        "--name", help="Run name; defaults to the dataset name. Reuse it to resume."
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        required=True,
        help="Exactly one provider: typesafe (Jev), openai (GPT), or google (Gemini).",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--api-key-env",
        help="Use only this named environment variable for the selected provider's key.",
    )
    parser.add_argument(
        "--model",
        type=pipeline.model_id,
        help=f"Model override. Defaults: typesafe {jev.DEFAULT_MODEL}; openai {batch.DEFAULT_OPENAI_MODEL}; google {batch.DEFAULT_GOOGLE_MODEL}.",
    )
    parser.add_argument(
        "--expect-rows",
        type=pipeline.positive_int,
        help="Optional exact LOCUS input size check; human_audit always requires 250.",
    )
    parser.add_argument(
        "--workers",
        type=pipeline.positive_int,
        default=pipeline.DEFAULT_WORKERS,
        help="Concurrent Typesafe rows.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=pipeline.positive_int,
        help="Optional character cap; truncation is recorded.",
    )
    parser.add_argument(
        "--batch-size", type=pipeline.positive_int, default=pipeline.DEFAULT_BATCH_SIZE
    )
    parser.add_argument(
        "--max-inflight-batches",
        type=pipeline.positive_int,
        default=pipeline.DEFAULT_MAX_INFLIGHT,
        help="Per-provider remote job cap; increase only within your token quota.",
    )
    parser.add_argument(
        "--check-every",
        type=pipeline.positive_float,
        help="Minutes between batch polls (human_audit: 0.5; LOCUS: 10).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry failed batch rows once on restart.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check every input row, selection, policy, and existing ledgers; no writes or API calls.",
    )
    args = parser.parse_args(argv)
    if args.api_key_env is not None and not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env
    ):
        raise pipeline.InputError(
            "--api-key-env must name an environment variable, not contain an API key"
        )
    if args.dataset == "LOCUS" and args.input is None:
        parser.error("LOCUS requires --input /absolute/path/to/original.jsonl")
    if args.dataset == "human_audit" and args.expect_rows not in (None, 250):
        parser.error("human_audit requires exactly 250 rows")
    args.input = (args.input or default_human_csv()).resolve()
    if args.dataset == "LOCUS" and args.input.suffix == ".gz":
        parser.error(
            "LOCUS requires uncompressed JSONL for efficient indexed reads; decompress separately first"
        )
    if args.dataset == "human_audit":
        args.expect_rows = 250
    if args.check_every is None:
        args.check_every = 0.5 if args.dataset == "human_audit" else 10
    if args.name is None:
        args.name = args.dataset
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        parser.error("--name must use only letters, numbers, underscores, or hyphens")
    if args.batch_size > 5000:
        parser.error("--batch-size must not exceed 5000")
    args.env_file = pipeline.resolve_env_file(args.env_file)
    return args


def ledger_path(args):
    """Keep each selected provider's checkpoint under its run name."""
    return ROOT / "runs" / f"{args.name}_{args.provider}.sqlite3"


def output_path(args):
    """Name a provider export beside the other outputs for this checkout."""
    return ROOT / "outputs" / f"{args.name}_{args.provider}.jsonl"


def input_cache_path(args):
    """Keep the model-only CSV projection separate from labels and outputs."""
    return ROOT / "runs" / f"{args.name}_{args.provider}.input.jsonl"


def _human_csv_records(path, stop_event=None):
    """Read only IDs and exact legal text from the human annotation CSV."""
    records = []
    previous_limit = csv.field_size_limit()
    try:
        csv.field_size_limit(sys.maxsize)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = reader.fieldnames
            if (
                fields is None
                or len(fields) != len(set(fields))
                or not {"audit_id", "text"} <= set(fields)
            ):
                raise pipeline.InputError(
                    "human_audit CSV needs unique column names including audit_id and text"
                )
            for row in reader:
                pipeline.check_cancelled(stop_event)
                if None in row or any(value is None for value in row.values()):
                    raise pipeline.InputError(
                        "human_audit CSV has a row whose columns do not match its header"
                    )
                records.append(
                    {"audit_id": row["audit_id"], "header": "", "content": row["text"]}
                )
                if len(records) > 250:
                    raise pipeline.InputError("human_audit CSV exceeds 250 rows")
    except (csv.Error, UnicodeError) as exc:
        raise pipeline.InputError("Invalid human_audit CSV") from exc
    finally:
        csv.field_size_limit(previous_limit)
    return records


def _select_human_input(args, records):
    """Validate a deterministic text-only cache without writing during preflight."""
    data = "".join(pipeline.canonical_json(row) + "\n" for row in records).encode(
        "utf-8"
    )
    digest = sha256(data).hexdigest()
    cache = input_cache_path(args)
    if cache.exists() and cache.read_bytes() != data:
        raise pipeline.InputError(
            "human_audit text or order differs from this run's input cache; use a new --name"
        )
    if ledger_path(args).exists():
        with closing(
            pipeline.AnnotationLedger.open_readonly(ledger_path(args))
        ) as ledger:
            if ledger.run_metadata.get("input_sha256") != digest:
                raise pipeline.InputError(
                    "human_audit CSV does not match this checkpoint's input; use a new --name"
                )
    args.human_csv_input = args.input
    args.input = cache
    args._prepared_input_bytes = data
    return digest


def _materialize_human_input(args, stop_event=None):
    """Publish the validated model-only cache only for an actual annotation run."""
    data = getattr(args, "_prepared_input_bytes", None)
    if data is None:
        return
    pipeline.check_cancelled(stop_event)
    if batch.file_stamp(args.human_csv_input) != args._human_csv_stamp:
        raise pipeline.InputError("human_audit CSV changed after preflight")
    if args.input.exists():
        if args.input.read_bytes() != data:
            raise pipeline.InputError("human_audit input cache changed after preflight")
        return
    try:
        with atomic_text_output(args.input) as handle:
            handle.write(data.decode("utf-8"))
    except FileExistsError:
        if args.input.read_bytes() != data:
            raise pipeline.InputError("Conflicting human_audit input cache") from None


def arguments(args):
    """Use the recovery CLI's semantics for the selected provider's operations."""
    result = [
        "--provider",
        args.provider,
        "--input",
        str(args.input),
        "--ledger",
        str(ledger_path(args)),
        "--allow-full-run",
        "--header-only-policy",
        "include",
        "--model",
        args.model,
    ]
    # The default .env may be absent when keys/model IDs come from the shell.
    if args.env_file.is_file():
        result += ["--env-file", str(args.env_file)]
    if args.api_key_env is not None:
        result += ["--api-key-env", args.api_key_env]
    if args.max_text_chars is not None:
        result += ["--max-text-chars", str(args.max_text_chars)]
    if args.provider == "typesafe":
        result += ["--workers", str(args.workers)]
        if args.retry_failed:
            result += ["--retry-failed"]
    else:
        result += [
            "--batch-size",
            str(args.batch_size),
            "--max-inflight-batches",
            str(args.max_inflight_batches),
        ]
    return result


def preflight(args, stop_event=None):
    """Validate the complete input and selected provider's ledger before paid work."""
    jev.load_dotenv(args.env_file)
    provider = args.provider
    if provider == "typesafe":
        args.model = pipeline.model_id(
            args.model or os.environ.get("TYPESAFE_MODEL_ID") or jev.DEFAULT_MODEL
        )
    elif args.model is None:
        args.model = (
            batch.DEFAULT_OPENAI_MODEL
            if provider == "openai"
            else batch.DEFAULT_GOOGLE_MODEL
        )
    source_input = args.input
    before = batch.file_stamp(source_input)
    # Fail on missing local dependencies/keys before scanning a large corpus.
    # Dry runs intentionally need neither and never contact a provider.
    if not args.dry_run:
        try:
            installed = importlib.util.find_spec(SDK_MODULES[provider]) is not None
        except ModuleNotFoundError:
            installed = False
        if not installed:
            raise pipeline.InputError(
                f"Install the {provider} dependency into this Python environment: pip install -e '.[{provider}]'"
            )
        if provider == "typesafe":
            jev.get_api_key(args.api_key_env)
        else:
            batch.api_key(provider, args.api_key_env)
    csv_records = (
        _human_csv_records(source_input, stop_event)
        if args.dataset == "human_audit" and source_input.suffix.lower() == ".csv"
        else None
    )
    source = (
        enumerate(csv_records)
        if csv_records is not None
        else pipeline.iter_jsonl(source_input)
    )
    rows, first = 0, None
    total_characters, maximum_characters, truncated_rows = 0, 0, 0
    audit_ids = set()
    last_progress = time.monotonic()
    for index, record in source:
        pipeline.check_cancelled(stop_event)
        pipeline.source_identity(record, index)
        text = compose_input_text(record)
        if not text:
            raise pipeline.InputError(f"Source row {index} has no text to annotate")
        if args.dataset == "human_audit":
            audit_id = record.get("audit_id")
            if (
                not isinstance(audit_id, str)
                or not audit_id.strip()
                or audit_id != audit_id.strip()
                or audit_id in audit_ids
            ):
                raise pipeline.InputError(
                    f"human_audit row {index} needs a unique audit_id without surrounding whitespace"
                )
            audit_ids.add(audit_id)
            if example_overlap(text):
                raise pipeline.InputError(
                    f"human_audit row {index} overlaps a supplied prompt example; resolve held-out leakage before running"
                )
        rows += 1
        length = len(text)
        total_characters += length
        maximum_characters = max(maximum_characters, length)
        truncated_rows += (
            args.max_text_chars is not None and length > args.max_text_chars
        )
        now = time.monotonic()
        if now - last_progress >= 15:
            print(
                json.dumps({"phase": "preflight", "validated_rows": rows}), flush=True
            )
            last_progress = now
        if args.expect_rows is not None and rows > args.expect_rows:
            raise pipeline.InputError(
                f"Input exceeds the expected {args.expect_rows:,} rows; no rows were selected or submitted"
            )
        if first is None:
            first = record
    if rows == 0:
        raise pipeline.InputError("Input contains no rows; no API calls made")
    if args.expect_rows is not None and rows != args.expect_rows:
        raise pipeline.InputError(
            f"Expected {args.expect_rows:,} rows, found {rows:,}; no API calls made"
        )
    input_hash = (
        _select_human_input(args, csv_records)
        if csv_records is not None
        else pipeline.sha256_file(source_input)
    )
    pipeline.check_cancelled(stop_event)
    args.input_sha256 = input_hash
    if batch.file_stamp(source_input) != before:
        raise pipeline.InputError("Input changed during preflight")
    if csv_records is not None:
        args._human_csv_stamp = before
    local_args = pipeline.build_parser().parse_args(arguments(args))
    pipeline.distinct_paths(args.input, local_args.ledger, output_path(args))
    if csv_records is not None:
        pipeline.distinct_paths(source_input, local_args.ledger, output_path(args))
    if provider == "typesafe":
        text, _ = pipeline.limit_text(compose_input_text(first), args.max_text_chars)
        build_jev_request(text, args.model)
        config = pipeline.run_metadata(
            input_sha256=input_hash,
            model=args.model,
            endpoint=jev.DEFAULT_ENDPOINT,
            header_only_policy="include",
            max_text_chars=args.max_text_chars,
        )
    else:
        config = batch.configuration(local_args, hash_input=False)
        config["input_sha256"] = input_hash
        batch.request_for(first, "preflight", config)
    if local_args.ledger.exists():
        with closing(
            pipeline.AnnotationLedger.open_readonly(local_args.ledger)
        ) as ledger:
            if not pipeline.run_metadata_matches(
                pipeline.ledger_metadata(ledger.connection), config
            ):
                raise pipeline.InputError(
                    f"{provider} ledger is incompatible; use a new --name"
                )
            pipeline.check_request_builders(ledger.connection)
    elif output_path(args).exists():
        raise pipeline.InputError(
            f"Output exists without its ledger; choose a new --name: {output_path(args)}"
        )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "rows": rows,
                "input": str(source_input),
                "provider": provider,
                "model": args.model,
                "input_sha256": input_hash,
                "max_text_chars": args.max_text_chars,
                "excerpt_characters": total_characters,
                "maximum_excerpt_characters": maximum_characters,
                "truncated_rows": truncated_rows,
                "contract_sha256": contract_sha256(),
                "dry_run": args.dry_run,
                "model_requests_before_retries": rows
                * (2 if provider == "typesafe" else 1),
            }
        ),
        flush=True,
    )


def run_provider(args, stop):
    """Resume only the selected provider through collection and export."""
    provider = args.provider
    shared = arguments(args)
    resume = ["--resume"] if ledger_path(args).exists() else []
    if provider == "typesafe":
        if stop.is_set():
            return "stopped"
        if (
            pipeline.main(
                [*shared, *resume],
                stop_event=stop,
                expected_input_sha256=args.input_sha256,
            )
            != 0
        ):
            return "failed; inspect recorded errors before --retry-failed; successful stages are checkpointed"
    else:
        # Local preparation completes before this invocation can create any job.
        if pipeline.main(
            [*shared, *resume, "--batch-action", "prepare"],
            stop_event=stop,
            expected_input_sha256=args.input_sha256,
        ) not in (0, batch.ROW_ERRORS_EXIT_CODE):
            return "preparation stopped; no new jobs submitted"
        retry = args.retry_failed
        while not stop.is_set():
            collected = pipeline.main(
                [*shared, "--resume", "--batch-action", "collect"],
                stop_event=stop,
                expected_input_sha256=args.input_sha256,
            )
            with closing(
                pipeline.AnnotationLedger.open_readonly(ledger_path(args))
            ) as ledger:
                progress = batch.summary(ledger.connection)
            if stop.is_set():
                return "stopped"
            if collected not in (0, batch.ROW_ERRORS_EXIT_CODE):
                return (
                    "collection failed; inspect provider/ledger state before resuming"
                )
            if progress["errors"] and not retry:
                return "failed; inspect errors before --retry-failed"
            if progress["annotated"] == progress["requests"]:
                break
            action = "retry" if retry else "submit"
            retry = False
            if (
                pipeline.main(
                    [*shared, "--resume", "--batch-action", action],
                    stop_event=stop,
                    expected_input_sha256=args.input_sha256,
                )
                != 0
            ):
                return "stopped; inspect submission state before retrying"
            print(
                f"{provider}: {progress['annotated']:,}/{progress['requests']:,} complete",
                flush=True,
            )
            if stop.wait(args.check_every * 60):
                return "stopped"
        else:
            return "stopped"
    if stop.is_set():
        return "stopped"
    code = pipeline.main(
        [
            *shared,
            "--export-only",
            "--output",
            str(output_path(args)),
            "--overwrite-output",
        ],
        stop_event=stop,
    )
    return f"exported: {output_path(args)}" if code == 0 else "export failed"


def main(argv=None):
    """Run one explicit provider on one explicit dataset, with safe interruption."""
    try:
        args = parse_args(argv)
        with pipeline.cancellation_scope() as stop:
            preflight(args, stop)
            pipeline.check_cancelled(stop)
            if args.dry_run:
                return 0
            _materialize_human_input(args, stop)
            status = run_provider(args, stop)
            pipeline.check_cancelled(stop)
            print(f"{args.provider}: {status}", flush=True)
            return 0 if status.startswith("exported:") else 2
    except (KeyboardInterrupt, CancelledError):
        print(
            "Interrupted; completed work is checkpointed. Resume the same command to continue.",
            file=sys.stderr,
        )
        return 130
    except (
        pipeline.InputError,
        jev.EnvironmentError,
        ContractError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
