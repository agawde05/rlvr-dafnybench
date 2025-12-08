"""
Utilities to describe, validate, and apply compact annotation patches.

The annotator model is expected to emit a small JSON array of edit
instructions instead of regenerating the full Dafny program. Each edit
specifies a 1-based line number in the original source and whether to
insert annotation text before or after that line. This keeps generations
short and makes reward computation faster because we reconstitute the
annotated code locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence
import json


EditPosition = Literal["insert_before", "insert_after"]


@dataclass(frozen=True)
class AnnotationEdit:
    """
    Single annotation edit to apply to a Dafny source file.

    Attributes
    ----------
    line:
        1-based line number in the original (unannotated) source to anchor
        the insertion. Values <= 0 are rejected.
    position:
        Whether to insert before or after the anchor line.
    content:
        The annotation text to insert. May span multiple lines; callers
        should not include leading indentation unless they want to override
        the detected indentation from the anchor line.
    indent:
        If True, the edit will adopt the indentation of the anchor line for
        every line in `content`. This keeps inserted invariants/contracts
        aligned without requiring the model to predict whitespace.
    """

    line: int
    position: EditPosition
    content: str
    indent: bool = True

    def normalized(self) -> "AnnotationEdit":
        """Return a copy with trimmed content and validated fields."""
        if self.line <= 0:
            raise ValueError("line must be >= 1")
        cleaned = self.content.rstrip("\n")
        if not cleaned.strip():
            raise ValueError("content must be non-empty")
        return AnnotationEdit(
            line=self.line,
            position=self.position,
            content=cleaned,
            indent=self.indent,
        )


def patch_json_schema() -> dict:
    """
    JSON schema describing a list of annotation edits the model must emit.
    """
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["line", "position", "content"],
            "additionalProperties": False,
            "properties": {
                "line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based line number in the original Dafny code.",
                },
                "position": {
                    "type": "string",
                    "enum": ["insert_before", "insert_after"],
                    "description": "Where to insert relative to `line`.",
                },
                "content": {
                    "type": "string",
                    "description": "Annotation text to insert. May be multi-line.",
                },
                "indent": {
                    "type": "boolean",
                    "description": "If true, match indentation of the anchor line.",
                    "default": True,
                },
                "note": {
                    "type": "string",
                    "description": "Optional human-readable rationale; ignored by the applier.",
                },
            },
        },
    }


def try_parse_edits(raw: str) -> Optional[List[AnnotationEdit]]:
    """
    Attempt to parse a JSON patch payload.

    Returns
    -------
    list[AnnotationEdit] if the string is valid JSON matching the expected
    shape, None if the string is not JSON at all, or raises ValueError for
    malformed JSON that looks like an attempted patch.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    edits = _coerce_edits(payload)
    return [edit.normalized() for edit in edits]


def _coerce_edits(payload: object) -> List[AnnotationEdit]:
    if isinstance(payload, dict) and "patch" in payload:
        payload = payload["patch"]

    if not isinstance(payload, list):
        raise ValueError("Patch must be a JSON array of edit objects.")

    edits: List[AnnotationEdit] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Patch entry {idx} must be an object.")

        if "line" not in item or "position" not in item or "content" not in item:
            raise ValueError(
                f"Patch entry {idx} must include line, position, and content."
            )

        line = item.get("line")
        position = item.get("position")
        content = item.get("content")
        indent = item.get("indent", True)

        if not isinstance(line, int):
            raise ValueError(f"Patch entry {idx} line must be an integer.")
        if position not in ("insert_before", "insert_after"):
            raise ValueError(
                f"Patch entry {idx} position must be 'insert_before' or 'insert_after'."
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Patch entry {idx} content must be a non-empty string.")
        if not isinstance(indent, bool):
            raise ValueError(f"Patch entry {idx} indent must be a boolean.")

        edits.append(
            AnnotationEdit(
                line=line,
                position=position,
                content=content,
                indent=indent,
            )
        )

    return edits


def apply_edits(source: str, edits: Sequence[AnnotationEdit]) -> str:
    """
    Apply a sequence of edits to the original Dafny source.

    Insertions are applied in reverse line order to keep downstream indices
    stable. Multi-line content is split and inserted line-by-line.
    """
    if not edits:
        return source

    lines = source.splitlines()
    trailing_newline = source.endswith("\n")

    # Sort descending by line number so earlier insertions do not shift later anchors.
    ordered = sorted((e.normalized() for e in edits), key=lambda e: e.line, reverse=True)

    for edit in ordered:
        anchor_idx = max(0, edit.line - 1)
        insert_at = anchor_idx if edit.position == "insert_before" else anchor_idx + 1
        insert_at = min(insert_at, len(lines))

        indent_prefix = ""
        original_line = ""
        if 0 <= anchor_idx < len(lines):
            original_line = lines[anchor_idx]

        if edit.indent:
            if original_line:
                indent_prefix = original_line[: len(original_line) - len(original_line.lstrip())]
            # If inserting inside a freshly opened block, bump indentation by one level.
            if (
                edit.position == "insert_after"
                and original_line
                and original_line.strip().endswith("{")
            ):
                indent_unit = "\t" if indent_prefix.startswith("\t") else "  "
                indent_prefix = indent_prefix + indent_unit

        new_lines = [
            f"{indent_prefix}{segment}" if segment else indent_prefix
            for segment in edit.content.splitlines()
        ]

        for offset, new_line in enumerate(new_lines):
            lines.insert(insert_at + offset, new_line)

    updated = "\n".join(lines)
    if trailing_newline:
        updated += "\n"
    return updated


def summarise_edits(edits: Iterable[AnnotationEdit]) -> str:
    """
    Human-readable summary string for logging.
    """
    parts = [f"{edit.position}@{edit.line}" for edit in edits]
    return ", ".join(parts)
