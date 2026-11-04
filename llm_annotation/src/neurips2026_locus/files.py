"""Atomic text outputs and input/output separation shared by data commands.

Temporary files live beside their destination so publishing is an atomic
filesystem operation. A failed transformation leaves no partial final file.
"""

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Iterator, TextIO


def opaque_document_id(locator: str, collection: str = "original") -> str:
    """Bind a document to its collection without publishing its source locator.

    Callers with a known corpus layout pass the canonical relative key, rather
    than a computer-specific prefix. The digest is an identifier, not a map
    back to a filename, and is independent of annotation-row identities.
    """
    if not isinstance(locator, str) or not locator:
        raise ValueError("A nonempty document locator is required")
    if not isinstance(collection, str) or not collection:
        raise ValueError("A nonempty document collection is required")
    return "doc_" + sha256((collection + "|" + locator).encode()).hexdigest()[:24]


def canonical_document_locator(locator: str) -> str:
    """Remove only a recognized collected-source prefix for public document IDs.

    Unknown layouts are kept as opaque hash inputs rather than guessed. No
    filesystem lookup or path-to-ID bridge is published.
    """
    normalized = locator.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        return locator
    if (
        len(parts) >= 4
        and re.fullmatch(r"[a-zA-Z]{2}", parts[-4])
        and parts[-2] in {"cities", "counties"}
        and parts[-1].endswith(".jsonl")
    ):
        return "/".join(parts[-4:])
    for index, part in enumerate(parts):
        remaining = parts[index:]
        if (
            part in {"new", "fixes"}
            and len(remaining) in {4, 5}
            and re.fullmatch(r"[a-zA-Z]{2}", remaining[2])
            and remaining[-1].endswith(".pdf")
            and (len(remaining) == 4 or remaining[3] == "charters")
        ):
            return "/".join(remaining)
    return locator


def file_stamp(path: Path) -> list[int]:
    """Detect changes to filesystem identity, size, or timestamps during a read."""
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def reject_output_aliases(output: Path, *inputs: Path) -> None:
    """Protect source files even when a destination is a symlink or hard link."""
    for source in inputs:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("The output must not replace an input file")


@contextmanager
def atomic_text_output(path: Path, *, overwrite: bool = False) -> Iterator[TextIO]:
    """Commit UTF-8 text after flush/fsync; refuse a racing overwrite by default.

    The no-overwrite operation uses a hard link on the destination filesystem.
    Callers must protect their own input paths before choosing ``overwrite``.
    The context manager never opens or truncates an existing destination.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
