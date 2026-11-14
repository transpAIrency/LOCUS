"""
Sample the laws to be used in the dataset.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT_GLOB = os.path.join(_HERE, "..", "data", "postprocessed", "**", "postprocessed.jsonl")
_DEFAULT_OUTPUT = os.path.join(_HERE, "..", "data", "sampled.jsonl")

_TABLE_RE = re.compile(r"<\s*table\b", re.IGNORECASE)


def compute_law_id(content: str, source: str) -> str:
    """deterministic id derived from a law's content and the file it came from.

    sha256 over source\0content keeps two identical bodies from different files distinct,
    and a null separator prevents boundary-collision between source and content.
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:16]


def _extract_text(obj: dict) -> str:
    """pull the textual body from a postprocessed record."""
    for key in ("text", "body", "content"):
        v = obj.get(key)
        if isinstance(v, str):
            return v
    return ""


def keep(obj: dict) -> bool:
    """drop records that are blank, whitespace-only, or contain an html table."""
    text = _extract_text(obj)
    if not text or not text.strip():
        return False
    if _TABLE_RE.search(text):
        return False
    return True


def reservoir_sample(paths: list[str], k: int, rng: random.Random) -> tuple[list[tuple[str, str]], int, int]:
    """return up to k (line, source_path) pairs drawn uniformly across all files via reservoir sampling.

    also returns (seen, dropped) counts for logging.
    """
    reservoir: list[tuple[str, str]] = []
    n = 0
    seen = 0
    dropped = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                seen += 1
                obj = json.loads(line)
                if not keep(obj):
                    dropped += 1
                    continue
                if len(reservoir) < k:
                    reservoir.append((line, path))
                else:
                    j = rng.randint(0, n)
                    if j < k:
                        reservoir[j] = (line, path)
                n += 1
    return reservoir, seen, dropped


def main():
    parser = argparse.ArgumentParser(description="Uniformly sample k laws across postprocessed.jsonl files.")
    parser.add_argument("--input-glob", default=_DEFAULT_INPUT_GLOB,
                        help="Glob for input jsonl files (default: ../data/postprocessed/**/postprocessed.jsonl)")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT,
                        help="Output jsonl path (default: ../data/sampled.jsonl)")
    parser.add_argument("-k", type=int, default=10_000,
                        help="Total number of items to sample across all files (default: 10000)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob, recursive=True))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    print(f"Found {len(paths)} input files.")
    rng = random.Random(args.seed)
    sampled, seen, dropped = reservoir_sample(paths, args.k, rng)
    print(f"Scanned {seen} lines; dropped {dropped} (blank or html table).")
    print(f"Sampled {len(sampled)} items (requested k={args.k}).")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line, source in sampled:
            # validate it's still parseable jsonl; re-serialize to normalize
            obj = json.loads(line)
            obj["id"] = compute_law_id(_extract_text(obj), source)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
