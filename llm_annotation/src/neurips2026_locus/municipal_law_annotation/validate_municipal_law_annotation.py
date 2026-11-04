#!/usr/bin/env python3
"""Validate one municipal-law annotation using only the Python standard library.

Usage: python3 validate_municipal_law_annotation.py [FILE|-]
With no FILE, read one JSON object from standard input.
"""

import argparse
import json
import sys


_FIELDS = {"primary_function", "subcategory", "confidence"}
_PRIMARY_FUNCTIONS = ("Context", "Rules", "Process", "Enforcement", "Structural")
_SUBCATEGORIES = ("Buildings", "Business", "Zoning", "Nuisance", "Other")
_CONFIDENCE_LEVELS = ("Low", "Medium", "High")


def validate_annotation(value):
    """Return value unchanged if valid; otherwise raise ValueError.

    Subcategory must be a topic for Rules or Enforcement and None otherwise.
    """
    if not isinstance(value, dict):
        raise ValueError("annotation must be a JSON object")
    if set(value) != _FIELDS:
        raise ValueError(
            "annotation must have exactly primary_function, subcategory, confidence"
        )

    primary = value["primary_function"]
    topic = value["subcategory"]
    confidence = value["confidence"]
    if not isinstance(primary, str) or primary not in _PRIMARY_FUNCTIONS:
        raise ValueError("primary_function must be Context, Rules, Process, Enforcement, or Structural")
    if not isinstance(confidence, str) or confidence not in _CONFIDENCE_LEVELS:
        raise ValueError("confidence must be Low, Medium, or High")

    if primary in ("Rules", "Enforcement"):
        if not isinstance(topic, str) or topic not in _SUBCATEGORIES:
            raise ValueError(
                "subcategory must be Buildings, Business, Zoning, Nuisance, or Other "
                "for Rules or Enforcement"
            )
    elif topic is not None:
        raise ValueError("subcategory must be null for Context, Process, or Structural")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + repr(key))
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("invalid JSON constant: " + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", default="-", help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        options = {"object_pairs_hook": _unique_object, "parse_constant": _invalid_constant}
        if args.file == "-":
            value = json.load(sys.stdin, **options)
        else:
            with open(args.file, encoding="utf-8") as source:
                value = json.load(source, **options)
        validate_annotation(value)
    except (ValueError, OSError, UnicodeError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
