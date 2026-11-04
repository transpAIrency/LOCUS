#!/usr/bin/env python3
"""Run one selected provider on human_audit or LOCUS from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from neurips2026_locus.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
