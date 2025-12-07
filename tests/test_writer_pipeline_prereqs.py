"""
Tests covering the helpers that prepare data for the unimplemented-to-
unannotated code generation pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.dafny_utils import strip_method_and_lemma_bodies  # noqa: E402
from scripts.get_data import _ensure_unimplemented_column  # noqa: E402
from scripts.train_model import _prepare_writer_pairs  # noqa: E402


def test_strip_method_removes_body_and_preserves_signature():
    source = """method Foo()
    {
        var x := 0;
        assert x == 0;
    }
    """
    stripped = strip_method_and_lemma_bodies(source)

    assert "var x := 0" not in stripped
    assert "assert x == 0" not in stripped
    assert "method Foo()" in stripped
    assert ";\n" in stripped  # Body replaced with stub terminator.


def test_strip_lemma_with_attributes():
    source = """lemma {:axiom} LemmaX(x: int)
    requires x >= 0
    {
        assert x >= 0;
    }
    """
    stripped = strip_method_and_lemma_bodies(source)

    assert "{:axiom}" in stripped
    assert "assert x >= 0" not in stripped
    assert "LemmaX" in stripped


def test_ensure_unimplemented_column_uses_annotated_and_body():
    annotated = """method Annotated()
    ensures true
    {
        assert true;
    }
    """
    body = """method Body()
    {
        var y := 1;
    }
    """
    df = pl.DataFrame(
        {
            "annotated_body": [annotated, None],
            "body": [None, body],
        }
    )

    result = _ensure_unimplemented_column(df)

    assert "unimplemented_body" in result.columns
    first_stub = result.get_column("unimplemented_body")[0]
    second_stub = result.get_column("unimplemented_body")[1]

    assert "assert true" not in first_stub
    assert "var y := 1" not in second_stub
    assert "method Annotated()" in first_stub
    assert "method Body()" in second_stub


def test_prepare_writer_pairs_filters_invalid_rows():
    df = pl.DataFrame(
        {
            "unimplemented_body": [
                "  method Foo();  ",
                None,
                "   ",
            ],
            "body": [
                "method Foo() { var x := 1; }",
                "method Bar() { }",
                None,
            ],
        }
    )

    pairs = _prepare_writer_pairs(df)

    assert pairs == [
        {
            "unimplemented_body": "method Foo();",
            "body": "method Foo() { var x := 1; }",
        }
    ]


def test_prepare_writer_pairs_missing_column():
    df = pl.DataFrame({"body": ["method Foo() { }"]})
    with pytest.raises(ValueError):
        _prepare_writer_pairs(df)
