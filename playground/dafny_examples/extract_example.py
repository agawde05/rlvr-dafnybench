"""
Extract a DafnyBench example from the cached Parquet and write annotated/unannotated
files into this playground directory.

Usage:
  uv run playground/dafny_examples/extract_example.py --index 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "data" / "DafnyBench" / "dafnybench.parquet"
OUT_DIR = ROOT / "playground" / "dafny_examples"


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

    annotated_path = OUT_DIR / f"annotated_{row['id']}.dfy"
    unannotated_path = OUT_DIR / f"unannotated_{row['id']}.dfy"
    annotated_path.write_text(row["annotated_body"])
    unannotated_path.write_text(row["body"])

    print(
        f"Wrote:\n  annotated -> {annotated_path}\n  unannotated -> {unannotated_path}"
    )
    if row.get("header"):
        print(f"header: {row['header']}")
    if row.get("spec"):
        print(f"spec: {row['spec']}")


if __name__ == "__main__":
    main()
