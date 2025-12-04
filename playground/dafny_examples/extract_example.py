"""
Extract a DafnyBench example from the cached Parquet and write annotated,
unannotated, and unimplemented files into this playground directory.

Usage:
  uv run playground/dafny_examples/extract_example.py --index 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
try:
    from tqdm.auto import tqdm
    tqdm_write = tqdm.write  # type: ignore[attr-defined]
except ImportError:
    class _TqdmStub:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, n: int = 1) -> None:
            return None

        def write(self, msg: str) -> None:
            print(msg)

    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return _TqdmStub(iterable=iterable)

    def tqdm_write(msg: str) -> None:
        print(msg)

ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "data" / "DafnyBench" / "dafnybench.parquet"
OUT_DIR = ROOT / "playground" / "dafny_examples"

sys.path.insert(0, str(ROOT))
from scripts.dafny_utils import strip_method_and_lemma_bodies  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index to extract from the Parquet file (0-based).",
    )
    args = parser.parse_args()

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet not found at {PARQUET_PATH}. Run scripts/get_data.py first."
        )

    df = pl.read_parquet(PARQUET_PATH)
    if args.index < 0 or args.index >= len(df):
        raise IndexError(f"index {args.index} out of range (0..{len(df)-1})")

    row = df.row(args.index, named=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    example_dir = OUT_DIR / f"example_{row['id']}"
    example_dir.mkdir(parents=True, exist_ok=True)

    annotated_body = row.get("annotated_body")
    unannotated_body = row.get("body")
    if annotated_body is None or unannotated_body is None:
        raise ValueError("Missing annotated/body text for the requested row.")

    unimplemented_body = row.get("unimplemented_body")
    if unimplemented_body is None:
        source_for_stub = annotated_body or unannotated_body
        unimplemented_body = strip_method_and_lemma_bodies(source_for_stub)

    annotated_path = example_dir / f"annotated_{row['id']}.dfy"
    unannotated_path = example_dir / f"unannotated_{row['id']}.dfy"
    unimplemented_path = example_dir / f"unimplemented_{row['id']}.dfy"

    writes = [
        ("annotated", annotated_path, annotated_body),
        ("unannotated", unannotated_path, unannotated_body),
        ("unimplemented", unimplemented_path, unimplemented_body),
    ]

    with tqdm(total=len(writes), desc=f"Example {row['id']}", unit="file") as bar:
        for label, path, content in writes:
            path.write_text(content)
            bar.update(1)
            bar.write(f"{label:>13} -> {path}")

    if row.get("header"):
        tqdm_write(f"header: {row['header']}")
    if row.get("spec"):
        tqdm_write(f"spec: {row['spec']}")


if __name__ == "__main__":
    main()
