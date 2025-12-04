"""
Utilities for merging structured JSON diffs into source code.

The diff format mirrors Git hunks while remaining JSON-serializable:

{
    "hunks": [
        {
            "original_start": 12,
            "original_length": 3,
            "patched_start": 12,
            "patched_length": 4,
            "lines": [
                {"type": "context", "text": "unchanged line"},
                {"type": "remove", "text": "old line"},
                {"type": "add", "text": "new line"}
            ]
        },
        ...
    ]
}

Line indices are 1-indexed, and `text` entries must not include trailing newlines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Union

DiffInput = Union[str, Mapping[str, Any]]


class DiffMergeError(ValueError):
    """Raised when a diff cannot be parsed or applied deterministically."""


@dataclass(frozen=True)
class _Hunk:
    original_start: int
    original_length: int
    patched_start: int
    patched_length: int
    lines: Sequence[Mapping[str, Any]]


def merge_diff(original_code: str, diff: DiffInput) -> str:
    """
    Apply a JSON hunks diff to the provided source string deterministically.

    Raises DiffMergeError if the diff is malformed or conflicts with the original.
    """
    spec = _parse_diff_spec(diff)
    hunks = spec.get("hunks", [])
    if not hunks:
        return original_code

    parsed_hunks = [_coerce_hunk(h) for h in hunks]
    _validate_hunk_order(parsed_hunks)

    original_lines = original_code.split("\n")
    had_trailing_newline = original_code.endswith("\n")

    result_lines: List[str] = []
    cursor = 0  # current index in original_lines

    for hunk in parsed_hunks:
        hunk_start_idx = hunk.original_start - 1
        if hunk_start_idx < cursor:
            raise DiffMergeError(
                "Overlapping or unsorted hunks detected while applying diff."
            )

        # Copy unchanged region preceding the hunk.
        result_lines.extend(original_lines[cursor:hunk_start_idx])
        cursor = hunk_start_idx

        expected_cursor_after_hunk = hunk_start_idx + hunk.original_length
        patched_line_count = 0

        for line_entry in hunk.lines:
            line_type = line_entry.get("type")
            text = _extract_text(line_entry)

            if line_type == "context":
                _assert_line_matches(original_lines, cursor, text)
                result_lines.append(text)
                cursor += 1
                patched_line_count += 1
            elif line_type == "remove":
                _assert_line_matches(original_lines, cursor, text)
                cursor += 1
            elif line_type == "add":
                result_lines.append(text)
                patched_line_count += 1
            else:
                raise DiffMergeError(
                    f"Unsupported line type '{line_type}'. Expected 'context', 'add', or 'remove'."
                )

        if cursor != expected_cursor_after_hunk:
            raise DiffMergeError(
                "Hunk application consumed an unexpected number of original lines."
            )

        if patched_line_count != hunk.patched_length:
            raise DiffMergeError(
                "Patched line count does not match 'patched_length' declared in diff."
            )

    # Append any trailing text after final hunk.
    result_lines.extend(original_lines[cursor:])

    merged = "\n".join(result_lines)
    if had_trailing_newline and not merged.endswith("\n"):
        merged = merged + "\n"
    return merged


def validate_diff_json(diff: DiffInput) -> bool:
    """
    Validate that the provided diff JSON adheres to the expected schema.

    Returns True when valid; False otherwise.
    """
    try:
        _parse_diff_spec(diff)
    except DiffMergeError:
        return False
    return True


def _parse_diff_spec(diff: DiffInput) -> Mapping[str, Any]:
    if isinstance(diff, str):
        try:
            diff_data = json.loads(diff)
        except json.JSONDecodeError as exc:
            raise DiffMergeError("Diff JSON could not be decoded.") from exc
    elif isinstance(diff, Mapping):
        diff_data = dict(diff)
    else:
        raise DiffMergeError("Diff must be a JSON string or mapping.")

    if not isinstance(diff_data, dict):
        raise DiffMergeError("Diff JSON must be an object or a list of hunks.")

    hunks = diff_data.get("hunks")
    if hunks is None:
        # Allow top-level {"diff": [...]} for robustness.
        hunks = diff_data.get("diff")
        if hunks is not None:
            diff_data = dict(diff_data)
            diff_data["hunks"] = hunks
    if hunks is None:
        raise DiffMergeError("Diff JSON is missing required 'hunks' list.")
    if not isinstance(hunks, list):
        raise DiffMergeError("'hunks' must be a list.")

    for hunk in hunks:
        _coerce_hunk(hunk)  # Validation occurs inside

    return diff_data


def _coerce_hunk(raw_hunk: Mapping[str, Any]) -> _Hunk:
    if not isinstance(raw_hunk, Mapping):
        raise DiffMergeError("Each hunk must be a JSON object.")

    try:
        original_start = int(raw_hunk["original_start"])
        original_length = int(raw_hunk["original_length"])
        patched_start = int(raw_hunk["patched_start"])
        patched_length = int(raw_hunk["patched_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DiffMergeError(
            "Hunk is missing required integer fields: 'original_start', 'original_length', "
            "'patched_start', 'patched_length'."
        ) from exc

    if original_start < 1 or patched_start < 1:
        raise DiffMergeError("Hunk line numbers must be >= 1.")
    if original_length < 0 or patched_length < 0:
        raise DiffMergeError("Hunk lengths must be non-negative.")

    lines = raw_hunk.get("lines")
    if not isinstance(lines, list):
        raise DiffMergeError("Hunk 'lines' must be a list.")
    if not lines and (original_length > 0 or patched_length > 0):
        raise DiffMergeError("Non-empty hunks must provide line changes.")

    for entry in lines:
        if not isinstance(entry, Mapping):
            raise DiffMergeError("Each line entry must be a JSON object.")
        line_type = entry.get("type")
        if line_type not in {"context", "add", "remove"}:
            raise DiffMergeError(
                "Line entry 'type' must be one of 'context', 'add', or 'remove'."
            )
        _extract_text(entry)

    return _Hunk(
        original_start=original_start,
        original_length=original_length,
        patched_start=patched_start,
        patched_length=patched_length,
        lines=lines,
    )


def _validate_hunk_order(hunks: Sequence[_Hunk]) -> None:
    previous_end = 0
    for hunk in hunks:
        if hunk.original_start - 1 < previous_end:
            raise DiffMergeError("Hunks must be sorted and non-overlapping.")
        previous_end = hunk.original_start - 1 + hunk.original_length


def _extract_text(entry: Mapping[str, Any]) -> str:
    try:
        text = entry["text"]
    except KeyError as exc:
        raise DiffMergeError("Line entry missing 'text' field.") from exc
    if not isinstance(text, str):
        raise DiffMergeError("'text' field must be a string.")
    return text


def _assert_line_matches(lines: Sequence[str], index: int, expected: str) -> None:
    if index >= len(lines):
        raise DiffMergeError(
            "Diff references lines beyond the end of the original source."
        )
    if lines[index] != expected:
        raise DiffMergeError(
            "Context or removal line does not match the original source."
        )
