import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation_patch import AnnotationEdit, apply_edits, try_parse_edits  # noqa: E402
from src.verification_task import get_generated_dafny_code  # noqa: E402


def test_try_parse_edits_valid_patch():
    raw = """
    [
        {"line": 3, "position": "insert_before", "content": "invariant i < n"},
        {"line": 5, "position": "insert_after", "content": "ensures arr[..] != null", "indent": false}
    ]
    """
    edits = try_parse_edits(raw)
    assert edits is not None
    assert len(edits) == 2
    assert edits[0].line == 3
    assert edits[0].position == "insert_before"
    assert edits[0].indent is True
    assert edits[1].indent is False


def test_try_parse_edits_non_json_returns_none():
    assert try_parse_edits("not-json") is None


def test_apply_edits_preserves_indentation():
    source = "method Foo()\n{\n  var x := 0;\n}\n"
    edits = [
        AnnotationEdit(
            line=3,
            position="insert_before",
            content="invariant x >= 0",
        ),
        AnnotationEdit(
            line=2,
            position="insert_after",
            content="// loop invariant placeholder",
            indent=False,
        ),
    ]
    updated = apply_edits(source, edits)
    expected = "\n".join(
        [
            "method Foo()",
            "{",
            "// loop invariant placeholder",
            "  invariant x >= 0",
            "  var x := 0;",
            "}",
            "",
        ]
    )
    assert updated == expected


def test_get_generated_dafny_code_applies_patch():
    original = "method Foo()\n{\n  var x := 0;\n}\n"
    patch = '[{"line": 3, "position": "insert_before", "content": "invariant x >= 0"}]'
    dfy_file = get_generated_dafny_code(patch, original)
    assert "invariant x >= 0" in dfy_file.get_code()


def test_try_parse_edits_rejects_missing_fields():
    bad = '[{"line": 2, "content": "invariant x > 0"}]'
    with pytest.raises(ValueError):
        try_parse_edits(bad)


def test_try_parse_edits_accepts_wrapped_patch_key():
    wrapped = '{"patch": [{"line": 1, "position": "insert_after", "content": "ensures true"}]}'
    edits = try_parse_edits(wrapped)
    assert edits is not None
    assert edits[0].line == 1


def test_apply_edits_handles_unsorted_input_and_trailing_newline():
    source = "line1\nline2\n"
    patch = """
    [
      {"line": 2, "position": "insert_after", "content": "ins2"},
      {"line": 1, "position": "insert_before", "content": "ins1"}
    ]
    """
    edits = try_parse_edits(patch)
    updated = apply_edits(source, edits or [])
    # insert_before line 1 then after line2, in reverse order application
    expected = "\n".join(["ins1", "line1", "line2", "ins2", ""])  # preserve trailing newline
    assert updated == expected


def test_apply_edits_respects_indent_false():
    source = "  line1\n"
    patch = '[{"line": 1, "position": "insert_before", "content": "noindent", "indent": false}]'
    edits = try_parse_edits(patch)
    updated = apply_edits(source, edits or [])
    assert updated.startswith("noindent\n  line1")


def test_patch_reconstructs_annotated_dafny():
    original = "\n".join(
        [
            "method Sum(n: int) returns (r: int)",
            "{",
            "  var i := 0;",
            "  while i < n {",
            "    i := i + 1;",
            "  }",
            "  return i;",
            "}",
            "",
        ]
    )
    patch = json.dumps(
        [
            {"line": 1, "position": "insert_after", "content": "ensures r == n"},
            {
                "line": 4,
                "position": "insert_after",
                "content": "invariant i <= n\ninvariant i >= 0\nvariant n - i",
            },
        ]
    )
    edits = try_parse_edits(patch)
    assert edits is not None

    annotated = apply_edits(original, edits)
    expected = "\n".join(
        [
            "method Sum(n: int) returns (r: int)",
            "ensures r == n",
            "{",
            "  var i := 0;",
            "  while i < n {",
            "    invariant i <= n",
            "    invariant i >= 0",
            "    variant n - i",
            "    i := i + 1;",
            "  }",
            "  return i;",
            "}",
            "",
        ]
    )
    assert annotated == expected


def test_get_generated_dafny_code_rebuilds_full_annotation():
    original = "\n".join(
        [
            "method Decr(n: int) returns (r: int)",
            "{",
            "  var k := n;",
            "  while k > 0 {",
            "    k := k - 1;",
            "  }",
            "  return k;",
            "}",
            "",
        ]
    )
    patch = json.dumps(
        [
            {"line": 1, "position": "insert_after", "content": "ensures r == 0"},
            {
                "line": 4,
                "position": "insert_after",
                "content": "invariant k >= 0\nvariant k",
            },
        ]
    )
    dfy_file = get_generated_dafny_code(patch, original)
    annotated = dfy_file.get_code()
    assert "ensures r == 0" in annotated
    assert "invariant k >= 0" in annotated
    assert "variant k" in annotated
    assert "while k > 0 {" in annotated
