"""Build and validate the four LOCUS release subsets without model API calls."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from .records import (
    ReleaseError, PROVIDERS, FUNCTIONS, TOPICS, audit_key, consensus,
    extract_annotation, read_counties, read_human, read_json,
    section_key, section_record,
)
from .provenance import LedgerEvidence, ProvenanceError
from .audit import summarize_human_audit


REPO = Path(__file__).resolve().parents[1]
MODELS = {"google": "gemini-3.8-flash", "openai": "gpt-6-luna", "typesafe": "jev-1.14.0"}
PROMPT = "oct2026_codebook_short"
SOURCE_COLUMNS = [
    "document_id", "section_index", "state", "jurisdiction_name", "jurisdiction_type",
    "jurisdiction_geoid", "represented_county_fips", "represented_county_name", "header", "content",
]
CONSENSUS_COLUMNS = ["function_majority", "function_votes", "topic_majority", "topic_votes"]
MODEL_COLUMNS = [f"{prefix}_{field}" for prefix in ("gemini", "openai", "jev")
                 for field in ("function", "topic", "confidence")] + ["jev_function_probs", "jev_topic_probs"]
SECTION_COLUMNS = SOURCE_COLUMNS + CONSENSUS_COLUMNS + MODEL_COLUMNS
LABEL_COLUMNS = SOURCE_COLUMNS[:2] + CONSENSUS_COLUMNS + MODEL_COLUMNS
HUMAN_COLUMNS = ["audit_id", "text", "human_function", "human_topic", "human_confidence",
                 "human_function_eligible", "human_topic_eligible", "human_confidence_eligible", "human_annotations"]
COUNTY_COLUMNS = ["county_fips", "county_name", "state", "document_id", "jurisdiction_name",
                  "jurisdiction_type", "selection_reason", "section_count"]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stamp(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _resolve(base, value):
    if not isinstance(value, str) or not value:
        raise ReleaseError("Every input path must be explicitly supplied")
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


class Inputs:
    """Fingerprint every input and reject modifications throughout the build."""

    def __init__(self):
        self.stamps = {}
        self.files = {}

    def watch(self, path):
        if not path.is_file():
            raise ReleaseError(f"Missing input: {path}")
        now = _stamp(path)
        if path in self.stamps and self.stamps[path] != now:
            raise ReleaseError(f"Input changed: {path}")
        self.stamps[path] = now

    def lines(self, path, role):
        self.watch(path)
        sha = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for position, raw in enumerate(handle):
                sha.update(raw)
                if not raw.strip():
                    raise ReleaseError(f"Blank JSONL line at {path}:{position + 1}")
                try:
                    row = read_json(raw)
                    if not isinstance(row, dict):
                        raise ReleaseError("Expected a JSON object")
                except ReleaseError as exc:
                    raise ReleaseError(f"{path}:{position + 1}: {exc}") from exc
                count += 1
                yield position, row
        self.files[(path, role)] = {"file": path.name, "role": role, "sha256": sha.hexdigest(),
                                    "bytes": self.stamps[path][2], "rows": count}
        self.watch(path)

    def hash(self, path, role):
        self.watch(path)
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(block)
        self.files[(path, role)] = {"file": path.name, "role": role, "sha256": sha.hexdigest(),
                                    "bytes": self.stamps[path][2]}
        self.watch(path)

    def unchanged(self):
        for path in self.stamps:
            self.watch(path)


def _harness():
    # Reuse the original identity and text construction; never rebuild hashes
    # from the cleaned release columns.
    src = str(REPO / "llm_annotation" / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from neurips2026_locus.pipeline import source_identity
    from neurips2026_locus.contract import compose_input_text
    return source_identity, compose_input_text


def schemas():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ReleaseError("PyArrow is required; see release_builder/requirements.txt") from exc
    string = pa.string()
    required = {"document_id", "section_index", "state", "jurisdiction_name", "jurisdiction_type",
                "represented_county_fips", "represented_county_name", "content", "function_votes", "topic_votes"}
    types = {name: string for name in SECTION_COLUMNS}
    types.update(section_index=pa.int64(), function_votes=pa.int8(), topic_votes=pa.int8(),
                 represented_county_fips=pa.list_(pa.field("element", string, nullable=False)),
                 represented_county_name=pa.list_(pa.field("element", string, nullable=False)),
                 jev_function_probs=pa.struct([pa.field(n, pa.float32(), nullable=False) for n in FUNCTIONS]),
                 jev_topic_probs=pa.struct([pa.field(n, pa.float32(), nullable=False) for n in TOPICS]))
    sections = pa.schema([pa.field(n, types[n], nullable=n not in required) for n in SECTION_COLUMNS])
    human_types = {n: string for n in HUMAN_COLUMNS}
    human_types.update({n: pa.bool_() for n in HUMAN_COLUMNS if n.endswith("_eligible")})
    human_types["human_annotations"] = pa.int64()
    human = pa.schema([pa.field(n, human_types[n], nullable=n in {"human_function", "human_topic", "human_confidence"})
                       for n in HUMAN_COLUMNS] + [sections.field(n) for n in MODEL_COLUMNS])
    counties = pa.schema([pa.field(n, pa.int64() if n == "section_count" else string,
                                   nullable=n in {"document_id", "jurisdiction_name", "jurisdiction_type", "section_count"})
                          for n in COUNTY_COLUMNS])
    return {"sections": sections, "labels": pa.schema([sections.field(n) for n in LABEL_COLUMNS]),
            "human_audit": human, "counties": counties}


def _arrow_version():
    try:
        return importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ReleaseError("PyArrow is required; see release_builder/requirements.txt") from exc


def _check_config(config, base):
    if not isinstance(config, dict) or set(config.get("models", {})) != set(PROVIDERS):
        raise ReleaseError("Configure exactly google, openai, and typesafe")
    if config.get("usage_scope") not in ("omit", "recorded"):
        raise ReleaseError("usage_scope must explicitly be omit or recorded")
    cap = config.get("max_text_chars")
    if type(cap) is not int or cap != 16000:
        raise ReleaseError("This release requires max_text_chars=16000")
    expected = config.get("expected", {})
    for name in ("sections", "documents", "counties", "covered_counties", "human_audit"):
        if type(expected.get(name)) is not int or expected[name] < 1:
            raise ReleaseError(f"Explicit expected {name} count is required")
    sha = expected.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ReleaseError("Explicit expected source sha256 is required")
    paths = [_resolve(base, config.get("inputs", {}).get(n)) for n in ("sections", "counties", "human")]
    for provider, model in config["models"].items():
        if model.get("completed") is not True:
            raise ReleaseError(f"{provider}: run has not been explicitly marked completed")
        exports = model.get("exports")
        if not isinstance(exports, list) or not exports:
            raise ReleaseError(f"{provider}: at least one explicit export is required")
        seen = set()
        for export in exports:
            p = _resolve(base, export.get("path"))
            if p in seen:
                raise ReleaseError(f"{provider}: repeated export path: {p}")
            seen.add(p)
            if export.get("kind") not in ("harness", "individual"):
                raise ReleaseError(f"{provider}: invalid ledger kind")
            paths.extend([p, _resolve(base, export.get("ledger"))])
            if export.get("individual_ledger") is not None:
                if export["kind"] != "harness" or provider not in ("google", "openai"):
                    raise ReleaseError("individual_ledger requires a Google/OpenAI harness export")
                paths.append(_resolve(base, export["individual_ledger"]))
        audit = model.get("audit", {})
        paths.extend(_resolve(base, audit.get(n)) for n in ("path", "ledger", "input"))
        dates = (model.get("run_start_utc"), model.get("run_end_utc"))
        parsed = []
        for value in dates:
            if value is None:
                parsed.append(None)
                continue
            try:
                date = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if date.utcoffset() is None or date.utcoffset().total_seconds() != 0:
                    raise ValueError
                parsed.append(date)
            except (ValueError, TypeError, AttributeError):
                raise ReleaseError(f"{provider}: run dates must be explicit UTC dates or null") from None
        if all(p is not None for p in parsed) and parsed[0] > parsed[1]:
            raise ReleaseError(f"{provider}: run end precedes start")
        if None in dates and not model.get("run_dates_note"):
            raise ReleaseError(f"{provider}: explain unavailable run dates in run_dates_note")
    for path in paths:
        if not path.is_file():
            raise ReleaseError(f"Missing input: {path}")
    return paths


def preflight(config, base_dir):
    """Read-only check; never reads large exports or creates an output directory."""
    from .card import validate_publication
    problems = []
    paths = []
    try:
        paths = _check_config(config, Path(base_dir))
    except ReleaseError as exc:
        problems.append(str(exc))
    try:
        validate_publication(config.get("publication"))
    except ReleaseError as exc:
        problems.append(str(exc))
    if problems:
        return {"ready": False, "reason": "; ".join(problems), "problems": problems}
    return {"ready": True, "input_files": len(set(paths)), "final_build_not_started": True}


def _signature(provider, meta):
    details = meta["run_details"]
    if details.get("requested_model") != MODELS[provider] or meta.get("prompt_version") != PROMPT:
        raise ReleaseError(f"Unexpected model or prompt for {provider}")
    if provider == "openai" and details.get("reasoning_effort") != "low":
        raise ReleaseError("OpenAI reasoning_effort must be low")
    return {"provider": provider, "model_id": details["requested_model"], "prompt_version": meta["prompt_version"],
            "contract_sha256": details["contract_sha256"], "request_builder_sha256": details["request_builder_sha256"],
            "reasoning_effort": details.get("reasoning_effort")}


def _permanent_failures(model, provider, db):
    notes = model.get("permanent_failures", [])
    if not isinstance(notes, list):
        raise ReleaseError(f"{provider}: permanent_failures must be a list")
    result, seen = [], set()
    for note in notes:
        if not isinstance(note, dict):
            raise ReleaseError(f"{provider}: invalid permanent failure note")
        document, index, reason = (note.get(n) for n in ("document_id", "section_index", "reason"))
        if (not isinstance(document, str) or not document.strip() or type(index) is not int
                or not 0 <= index < 2**63 or not isinstance(reason, str) or not reason.strip()):
            raise ReleaseError(f"{provider}: permanent failure needs a section key and reason")
        key = (document, index)
        if key in seen:
            raise ReleaseError(f"{provider}: duplicate permanent failure note")
        seen.add(key)
        if db.execute("SELECT 1 FROM sections WHERE document_id=? AND section_index=?", key).fetchone() is None:
            raise ReleaseError(f"{provider}: permanent failure is absent from source sections: {key}")
        answer = db.execute("SELECT status FROM annotations WHERE provider=? AND document_id=? AND section_index=?",
                            (provider, *key)).fetchone()
        if answer is not None and answer[0] == "ANNOTATED":
            raise ReleaseError(f"{provider}: permanent failure contradicts a valid annotation: {key}")
        result.append({"document_id": document, "section_index": index, "reason": reason})
    return result


def _sample_update(samples, rng, count, item, sample_size):
    if len(samples) < sample_size:
        samples.append(item)
    else:
        offset = rng.randrange(count)
        if offset < sample_size:
            samples[offset] = item


class Parquets:
    """Bounded output batches, explicit Arrow schemas, and approximate byte shards."""

    def __init__(self, root, row_group_size, shard_target_bytes):
        import pyarrow.parquet as pq
        self.pq = pq
        self.root, self.group, self.target = root, row_group_size, shard_target_bytes
        self.schemas = schemas()
        self.writers = {}
        self.counts = Counter()
        self.shard = 0
        self.section_bytes = 0
        for subset in self.schemas:
            (root / subset).mkdir()

    def _writer(self, subset):
        if subset not in self.writers:
            filename = f"part-{self.shard:05d}.parquet" if subset == "sections" else "data.parquet"
            self.writers[subset] = self.pq.ParquetWriter(self.root / subset / filename, self.schemas[subset], compression="zstd")
        return self.writers[subset]

    def write(self, subset, rows):
        if not rows:
            return
        import pyarrow as pa
        table = pa.Table.from_pylist(rows, schema=self.schemas[subset])
        table.validate(full=True)
        self._writer(subset).write_table(table, row_group_size=self.group)
        self.counts[subset] += len(rows)
        if subset == "sections":
            # Read the actual bytes flushed by the writer; footer bytes arrive at close.
            path = self.root / subset / f"part-{self.shard:05d}.parquet"
            self.section_bytes = path.stat().st_size
            if self.section_bytes >= self.target:
                self.writers.pop("sections").close()
                self.shard += 1
                self.section_bytes = 0

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def verify(self):
        for subset, schema in self.schemas.items():
            count = 0
            for path in sorted((self.root / subset).glob("*.parquet")):
                file = self.pq.ParquetFile(path)
                if not file.schema_arrow.equals(schema, check_metadata=False):
                    raise ReleaseError(f"Wrong written schema: {path}")
                count += file.metadata.num_rows
                for index in range(file.num_row_groups):
                    group = file.metadata.row_group(index)
                    if group.num_rows > self.group:
                        raise ReleaseError(f"Oversized row group: {path}")
                    if any(group.column(i).compression != "ZSTD" for i in range(group.num_columns)):
                        raise ReleaseError(f"Wrong compression: {path}")
            if count != self.counts[subset]:
                raise ReleaseError(f"Written count mismatch for {subset}")


def _code_evidence():
    files = [REPO / "build_release.py", *sorted(Path(__file__).parent.glob("*.py"))]
    evidence = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files if p.is_file() and not p.name.startswith("test_")}
    # Include identity/text construction and its frozen policy dependencies.
    for p in sorted((REPO / "llm_annotation" / "src" / "neurips2026_locus").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".txt", ".json"):
            evidence[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True,
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "working_tree_dirty": dirty, "source_sha256": evidence}


def _dataset_card(manifest, publication):
    from .card import render_dataset_card
    return render_dataset_card(manifest, publication)


def build_release(config, output, base_dir, *, row_group_size=100000,
                  shard_target_bytes=200_000_000, sample_size=20, seed=0, temp_dir=None):
    """Validate all sources and optionally publish a new local release atomically.

    output=None runs all joins/ledger checks without producing release files.
    The function refuses to overwrite any existing output. It never calls a
    provider, exports a ledger, edits inputs, or uploads to Hugging Face.
    """
    base = Path(base_dir).resolve()
    paths = _check_config(config, base)
    for name, value in (("row_group_size", row_group_size), ("shard_target_bytes", shard_target_bytes), ("sample_size", sample_size)):
        if type(value) is not int or value < 1:
            raise ReleaseError(f"{name} must be a positive integer")
    output = Path(output).resolve() if output is not None else None
    if output is not None:
        if output.exists():
            raise ReleaseError(f"Refusing to overwrite existing output: {output}")
        if not output.parent.is_dir():
            raise ReleaseError(f"Output parent must already exist: {output.parent}")
        if any(p == output or output in p.parents for p in paths):
            raise ReleaseError("Output must not contain any input")
    publication = None
    if output is not None:
        from .card import validate_publication
        publication = validate_publication(config.get("publication"))
    arrow_version = _arrow_version() if output is not None else None
    source_identity, compose_text = _harness()
    inputs = Inputs()
    for p in paths:
        if p.suffix != ".sqlite3":
            inputs.watch(p)
    expected = config["expected"]
    inputs.hash(_resolve(base, config["inputs"]["counties"]), "county_harmonization")
    inputs.hash(_resolve(base, config["inputs"]["human"]), "human_labels")
    counties, documents = read_counties(_resolve(base, config["inputs"]["counties"]))
    human = read_human(_resolve(base, config["inputs"]["human"]))
    covered = sum(c["document_id"] is not None for c in counties)
    for name, actual in (("counties", len(counties)), ("covered_counties", covered),
                         ("documents", len(documents)), ("human_audit", len(human))):
        if actual != expected[name]:
            raise ReleaseError(f"Expected {expected[name]} {name}, found {actual}")
    rng = random.Random(seed)
    samples, document_counts = [], Counter()
    manifest = {"format_version": 1, "dataset": "LOCUS v1.1", "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "counts": {}, "models": {}, "missing_value": None,
                "build_configuration_sha256": hashlib.sha256(canonical(config).encode("utf-8")).hexdigest(),
                "build_parameters": {"row_group_size": row_group_size, "shard_target_bytes": shard_target_bytes,
                                     "sample_size": sample_size, "seed": seed},
                "environment": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                                "pyarrow": arrow_version},
                "publication": publication,
                "validation": {"sample_seed": seed, "requested_sample_rows": sample_size,
                               "sampled_sections": [], "ledger_checks": Counter()},
                "build_script": _code_evidence()}
    staging = None
    try:
        if output is not None:
            staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
        with tempfile.TemporaryDirectory(prefix="locus-release-", dir=temp_dir) as scratch, ExitStack() as stack:
            db = sqlite3.connect(Path(scratch) / "join.sqlite3")
            stack.callback(db.close)
            db.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-32768;
                PRAGMA temp_store=FILE;
                CREATE TABLE sections(document_id TEXT NOT NULL, section_index INTEGER NOT NULL,
                    state TEXT NOT NULL, source_id TEXT NOT NULL UNIQUE, chars INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(document_id,section_index)) WITHOUT ROWID;
                CREATE TABLE annotations(provider TEXT NOT NULL, document_id TEXT NOT NULL,
                    section_index INTEGER NOT NULL, source_id TEXT NOT NULL, chars INTEGER NOT NULL,
                    sent INTEGER NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                    PRIMARY KEY(provider,document_id,section_index), UNIQUE(provider,source_id)) WITHOUT ROWID;
            """)
            source = _resolve(base, config["inputs"]["sections"])
            count = 0
            for position, raw in inputs.lines(source, "harmonized_sections"):
                row = section_record(raw, documents)
                key = section_key(raw)
                sid, _ = source_identity(raw, position)
                chars = len(compose_text(raw))
                try:
                    db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?)", (*key, row["state"], sid, chars, canonical(row)))
                except sqlite3.IntegrityError:
                    raise ReleaseError(f"Duplicate section key or source identity at source line {position + 1}") from None
                document_counts[key[0]] += 1
                count += 1
                _sample_update(samples, rng, count, {"document_id": key[0], "section_index": key[1]}, sample_size)
                if count % 10000 == 0:
                    db.commit()
            if count != expected["sections"]:
                raise ReleaseError(f"Expected {expected['sections']} sections, found {count}")
            if inputs.files[(source, "harmonized_sections")]["sha256"] != expected["sha256"]:
                raise ReleaseError("Harmonized input SHA-256 mismatch")
            if set(document_counts) != set(documents):
                raise ReleaseError("Sections and county-selected document sets differ")
            sample_keys = {(r["document_id"], r["section_index"]) for r in samples}
            manifest["validation"]["sampled_sections"] = sorted(samples, key=lambda r: (r["document_id"], r["section_index"]))
            ledgers = {}

            def evidence(spec):
                path = _resolve(base, spec["ledger"])
                kind = spec.get("kind", "harness")
                key = (path, kind)
                if key not in ledgers:
                    ledgers[key] = stack.enter_context(LedgerEvidence(path, kind=kind))
                return ledgers[key]

            signatures = {}
            for provider, model in config["models"].items():
                status_counts, submission_counts = Counter(), Counter()
                truncation_count = 0
                for spec in model["exports"]:
                    ledger = evidence(spec)
                    recovery_ledger = None
                    if spec.get("individual_ledger") is not None:
                        recovery_ledger = evidence({"ledger": spec["individual_ledger"], "kind": "individual"})
                    path = _resolve(base, spec["path"])
                    for position, raw in inputs.lines(path, provider + "_sections"):
                        key = section_key(raw)
                        labels, meta = extract_annotation(raw, provider)
                        signature = _signature(provider, meta)
                        previous = signatures.setdefault(provider, signature)
                        if signature != previous:
                            raise ReleaseError(f"Mixed model/prompt/request provenance for {provider}")
                        details = meta["run_details"]
                        chars, sent = details["input_text_characters"], details["sent_text_characters"]
                        if sent != min(chars, config["max_text_chars"]):
                            raise ReleaseError(f"Unexpected text cap for {provider} at {path}:{position + 1}")
                        submission = details["request_submission"]
                        if spec["kind"] == "individual" and submission != "individual":
                            raise ReleaseError("Individual recovery export claims a different request mode")
                        row_ledger = ledger
                        is_recovery = spec["kind"] == "individual"
                        if spec["kind"] == "harness" and provider in ("google", "openai") and submission == "individual":
                            if recovery_ledger is None:
                                raise ReleaseError(f"{provider}: combined export requires individual_ledger for recovery rows")
                            row_ledger = recovery_ledger
                            is_recovery = True
                        try:
                            db.execute("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?)",
                                       (provider, *key, meta["source_id"], chars, sent, canonical(labels), meta["status"]))
                        except sqlite3.IntegrityError:
                            raise ReleaseError(f"Duplicate {provider} annotation key or source identity at {path}:{position + 1}") from None
                        if key in sample_keys or is_recovery:
                            row_ledger.check(meta["source_id"], meta["annotation"])
                            manifest["validation"]["ledger_checks"][provider] += 1
                        status_counts[meta["status"]] += 1
                        submission_counts[submission] += 1
                        truncation_count += bool(details["text_truncated"])
                        if position % 10000 == 0:
                            db.commit()
                # SQL checks cover every export, including rows outside the sample.
                wrong = db.execute("""SELECT a.document_id,a.section_index FROM annotations a
                    LEFT JOIN sections s USING(document_id,section_index)
                    WHERE a.provider=? AND (s.document_id IS NULL OR s.source_id<>a.source_id
                    OR s.chars<>a.chars) LIMIT 1""", (provider,)).fetchone()
                if wrong:
                    raise ReleaseError(f"Extra key, source identity, or text mismatch for {provider}: {wrong}")
                missing = count - status_counts["ANNOTATED"]
                if missing < 0:
                    raise ReleaseError(f"Too many answers for {provider}")
                info = {**signatures.get(provider, {}), "rows_annotated": status_counts["ANNOTATED"],
                        "rows_missing": missing, "rows_blocked": status_counts["BLOCKED"],
                        "rows_without_export": count - sum(status_counts.values()),
                        "rows_by_request_submission": dict(submission_counts),
                        "truncated_export_rows": truncation_count, "max_text_chars": config["max_text_chars"],
                        "run_start_utc": model.get("run_start_utc"), "run_end_utc": model.get("run_end_utc"),
                        "run_dates_note": model.get("run_dates_note"), "declared_completed": True}
                info["permanent_failures"] = _permanent_failures(model, provider, db)
                manifest["models"][provider] = info
                # Audit identity is tied to the actual model input, not a cleaned
                # reconstruction with guessed locator fields.
                audit = model["audit"]
                audit_source = _resolve(base, audit["input"])
                audit_input = {}
                human_by_id = {r["audit_id"]: r for r in human}
                for position, raw in inputs.lines(audit_source, provider + "_audit_input"):
                    identity = audit_key(raw)
                    if identity in audit_input or identity not in human_by_id:
                        raise ReleaseError(f"Duplicate or unknown human audit input ID: {identity}")
                    if raw.get("content") != human_by_id[identity]["text"] or raw.get("header") not in (None, ""):
                        raise ReleaseError(f"Human audit input text mismatch for {identity}")
                    audit_input[identity] = (source_identity(raw, position)[0], len(compose_text(raw)))
                if set(audit_input) != set(human_by_id):
                    raise ReleaseError(f"Incomplete human audit input for {provider}")
                audit_answers = {}
                ledger = evidence(audit)
                for position, raw in inputs.lines(_resolve(base, audit["path"]), provider + "_audit_export"):
                    identity = audit_key(raw)
                    if identity in audit_answers or identity not in audit_input:
                        raise ReleaseError(f"Duplicate or unknown human audit export ID: {identity}")
                    labels, meta = extract_annotation(raw, provider)
                    signature = _signature(provider, meta)
                    previous = signatures.setdefault(provider, signature)
                    if previous != signature:
                        raise ReleaseError(f"Human audit and full run provenance differ for {provider}")
                    sid, chars = audit_input[identity]
                    if meta["source_id"] != sid or meta["run_details"]["input_text_characters"] != chars:
                        raise ReleaseError(f"Human audit source identity mismatch for {identity}")
                    if meta["run_details"]["sent_text_characters"] != min(chars, config["max_text_chars"]):
                        raise ReleaseError(f"Human audit text cap mismatch for {identity}")
                    ledger.check(sid, meta["annotation"])
                    audit_answers[identity] = labels
                # Missing audit answers are permitted with the same null convention.
                for row in human:
                    prefix = PROVIDERS[provider][1]
                    blank = {n: None for n in MODEL_COLUMNS if n.startswith(prefix + "_")}
                    row.update(audit_answers.get(row["audit_id"], blank))
                info["human_audit_rows_missing"] = sum(audit_answers.get(r["audit_id"], {}).get(PROVIDERS[provider][1] + "_function") is None for r in human)
                info["human_audit_export_rows"] = len(audit_answers)
                if not info.get("provider"):
                    if provider not in signatures:
                        raise ReleaseError(f"No model provenance available for {provider}")
                    info.update(signatures[provider])
            db.commit()
            # Verify cross-provider policy and request builder hashes rather than
            # assuming identical text implies identical policy.
            for field in ("contract_sha256", "request_builder_sha256", "prompt_version"):
                if len({info[field] for info in manifest["models"].values()}) != 1:
                    raise ReleaseError(f"Providers have different {field}")
            db.execute("CREATE INDEX sections_sort ON sections(state,document_id,section_index)")
            db.commit()
            for row in counties:
                if row["document_id"] is not None:
                    row["section_count"] = document_counts[row["document_id"]]
            writer = Parquets(staging, row_group_size, shard_target_bytes) if staging else None
            batches, label_batches, output_samples = [], [], []
            votes, missing_counts = Counter(), Counter()
            query = """SELECT s.payload,g.payload,o.payload,j.payload FROM sections s
                LEFT JOIN annotations g ON g.provider='google' AND g.document_id=s.document_id AND g.section_index=s.section_index
                LEFT JOIN annotations o ON o.provider='openai' AND o.document_id=s.document_id AND o.section_index=s.section_index
                LEFT JOIN annotations j ON j.provider='typesafe' AND j.document_id=s.document_id AND j.section_index=s.section_index
                ORDER BY s.state,s.document_id,s.section_index"""
            try:
                for payload, *answers in db.execute(query):
                    row = read_json(payload)
                    labels = {name: None for name in MODEL_COLUMNS}
                    for answer in answers:
                        if answer is not None:
                            labels.update(read_json(answer))
                    row.update(labels)
                    row.update(consensus(labels))
                    if (row["document_id"], row["section_index"]) in sample_keys:
                        output_samples.append(row)
                    for provider, (_, prefix) in PROVIDERS.items():
                        missing_counts[provider] += labels[prefix + "_function"] is None
                    votes[(row["function_votes"], row["topic_votes"])] += 1
                    if writer:
                        batches.append(row)
                        label_batches.append({n: row[n] for n in LABEL_COLUMNS})
                        if len(batches) >= row_group_size:
                            writer.write("sections", batches)
                            writer.write("labels", label_batches)
                            batches, label_batches = [], []
                for provider in PROVIDERS:
                    if missing_counts[provider] != manifest["models"][provider]["rows_missing"]:
                        raise ReleaseError(f"Null count mismatch for {provider}")
                if writer:
                    writer.write("sections", batches)
                    writer.write("labels", label_batches)
                    writer.write("human_audit", human)
                    writer.write("counties", sorted(counties, key=lambda r: r["county_fips"]))
            finally:
                if writer:
                    writer.close()
            if writer:
                writer.verify()
                from .verify import verify_outputs
                manifest["validation"]["parquet_readback"] = verify_outputs(
                    staging, writer.schemas,
                    {"sections": count, "labels": count, "human_audit": len(human), "counties": len(counties),
                     "documents": len(documents), "covered_counties": covered, "uncovered_counties": len(counties) - covered},
                    output_samples,
                )
                verified = manifest["validation"]["parquet_readback"]
                for provider, (_, prefix) in PROVIDERS.items():
                    info = manifest["models"][provider]
                    if (verified["model_null_counts"][prefix] != info["rows_missing"]
                            or verified["human_model_null_counts"][prefix] != info["human_audit_rows_missing"]):
                        raise ReleaseError(f"Written null counts disagree with manifest for {provider}")
            manifest["counts"] = {"sections": count, "labels": count, "human_audit": len(human),
                                  "counties": len(counties), "documents": len(documents),
                                  "covered_counties": covered, "uncovered_counties": len(counties) - covered}
            manifest["human_audit_summary"] = summarize_human_audit(human)
            manifest["validation"]["vote_count_histogram"] = {f"{f},{t}": n for (f, t), n in sorted(votes.items())}
            manifest["validation"]["ledger_checks"] = dict(manifest["validation"]["ledger_checks"])
            manifest["ledgers"] = []
            for (path, kind), ledger in ledgers.items():
                info = {"file": path.name, "kind": kind, "metadata": ledger.snapshot_metadata()}
                if config["usage_scope"] == "recorded":
                    info["recorded_usage"] = ledger.usage_summary()
                ledger.assert_unchanged()
                manifest["ledgers"].append(info)
            inputs.unchanged()
            manifest["inputs"] = list(inputs.files.values())
            manifest["token_accounting"] = "omitted" if config["usage_scope"] == "omit" else "recorded_usage_only"
            if staging:
                (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                (staging / "README.md").write_text(_dataset_card(manifest, publication), encoding="utf-8")
                inputs.unchanged()
                for ledger in ledgers.values():
                    ledger.assert_unchanged()
        # Context exit performs the final ledger mutation checks. Publish only
        # after every evidence context has closed without an error.
        if staging:
            inputs.unchanged()
            if output.exists():
                raise ReleaseError(f"Output appeared during build: {output}")
            staging.rename(output)
            staging = None
        return manifest
    except (ProvenanceError, sqlite3.Error) as exc:
        raise ReleaseError(str(exc)) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Check readiness without reading large files")
    mode.add_argument("--validate-only", action="store_true", help="Run all joins and ledger checks without release files")
    parser.add_argument("--temp-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config.read_bytes())
        if args.preflight:
            result = preflight(config, args.config.resolve().parent)
            print(json.dumps(result, indent=2))
            return 0 if result["ready"] else 1
        if not args.validate_only and args.output is None:
            raise ReleaseError("Supply --output or --validate-only")
        result = build_release(config, None if args.validate_only else args.output,
                               args.config.resolve().parent, temp_dir=args.temp_dir)
        print(json.dumps({"validated": True, "counts": result["counts"],
                          "output": str(args.output) if not args.validate_only else None}, indent=2))
        return 0
    except (ReleaseError, OSError) as exc:
        print(f"Release build stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
