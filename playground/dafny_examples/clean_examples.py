"""
Remove generated example folders (example_*) under playground/dafny_examples.

Usage:
  uv run playground/dafny_examples/clean_examples.py
  uv run playground/dafny_examples/clean_examples.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / "playground" / "dafny_examples"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List folders that would be deleted without removing them.",
    )
    args = parser.parse_args()

    if not EXAMPLES_DIR.exists():
        print(f"No playground directory at {EXAMPLES_DIR}")
        return

    targets = sorted(p for p in EXAMPLES_DIR.glob("example_*") if p.is_dir())
    if not targets:
        print("No example_* folders to remove.")
        return

    for folder in targets:
        if args.dry_run:
            print(f"[dry-run] would remove {folder}")
            continue
        shutil.rmtree(folder)
        print(f"Removed {folder}")


if __name__ == "__main__":
    main()
