"""Utilities for manipulating Dafny source text."""
from __future__ import annotations

from typing import Optional


def _is_identifier_char(ch: str) -> bool:
    # Dafny identifiers can include prime marks (') as suffixes.
    return ch.isalnum() or ch in {"_", "'"}


def _line_indent_for_position(text: str, pos: int) -> str:
    """Return the leading whitespace on the line containing ``pos``."""
    line_start = text.rfind("\n", 0, pos) + 1
    idx = line_start
    while idx < pos and text[idx] in (" ", "\t"):
        idx += 1
    return text[line_start:idx]


def _find_body_start(text: str, start_pos: int) -> Optional[int]:
    """
    Find the opening brace for a method/lemma body starting at ``start_pos``.

    Returns ``None`` if the declaration already ends with a semicolon.
    """
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    method_line_start = text.rfind("\n", 0, start_pos) + 1
    i = start_pos
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = False
            i += 1
            continue

        if in_char:
            if ch == "'" and (i == 0 or text[i - 1] != "\\"):
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            prev_char = text[i - 1] if i > 0 else ""
            if _is_identifier_char(prev_char):
                i += 1
                continue
            in_char = True
            i += 1
            continue

        # Skip attribute blocks that look like "{: ... }".
        if ch == "{" and nxt == ":":
            try:
                attr_end = _find_matching_brace(text, i)
            except ValueError:
                return None
            i = attr_end + 1
            continue

        if ch == "{":
            line_start = text.rfind("\n", 0, i) + 1
            before = text[line_start:i]
            if before.strip() == "" or line_start == method_line_start:
                return i

        i += 1

    return None


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return the index of the closing brace matching ``open_pos``."""
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    depth = 0
    i = open_pos
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = False
            i += 1
            continue

        if in_char:
            if ch == "'" and (i == 0 or text[i - 1] != "\\"):
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            prev_char = text[i - 1] if i > 0 else ""
            if _is_identifier_char(prev_char):
                i += 1
                continue
            in_char = True
            i += 1
            continue

        if not (in_line_comment or in_block_comment or in_string or in_char):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i

        i += 1

    raise ValueError("Unmatched brace while stripping Dafny method body.")


def strip_method_and_lemma_bodies(source: Optional[str]) -> Optional[str]:
    """
    Remove method and lemma implementations, keeping signatures/contracts.

    We treat the first brace on the method/lemma line (or a following line that
    starts with a brace) as the body start, then drop text until the matching
    closing brace. Contract-only declarations ending with ``;`` are left as-is.
    """
    if source is None:
        return None

    n = len(source)
    result: list[str] = []
    last_emit = 0
    i = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == '"' and (i == 0 or source[i - 1] != "\\"):
                in_string = False
            i += 1
            continue

        if in_char:
            if ch == "'" and (i == 0 or source[i - 1] != "\\"):
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            prev_char = source[i - 1] if i > 0 else ""
            if _is_identifier_char(prev_char):
                i += 1
                continue
            in_char = True
            i += 1
            continue

        keyword = None
        if source.startswith("method", i):
            keyword = "method"
        elif source.startswith("lemma", i):
            keyword = "lemma"

        if keyword:
            kw_len = len(keyword)
            prev_char = source[i - 1] if i > 0 else ""
            next_char = source[i + kw_len] if i + kw_len < n else ""
            if not _is_identifier_char(prev_char) and not _is_identifier_char(next_char):
                open_pos = _find_body_start(source, i + kw_len)
                if open_pos is not None:
                    try:
                        close_pos = _find_matching_brace(source, open_pos)
                    except ValueError:
                        # If parsing fails, leave the remainder untouched.
                        break

                    prefix = source[last_emit:open_pos]
                    prefix = prefix.rstrip(" \t")
                    indent = _line_indent_for_position(source, open_pos)

                    if prefix and not prefix.endswith("\n"):
                        prefix += "\n"

                    result.append(prefix)
                    result.append(f"{indent};\n")

                    i = close_pos + 1
                    last_emit = i
                    in_line_comment = in_block_comment = in_string = in_char = False
                    continue

        i += 1

    result.append(source[last_emit:])
    return "".join(result)


__all__ = ["strip_method_and_lemma_bodies"]
