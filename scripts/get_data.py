from pathlib import Path

import datasets
import polars as pl

DEFAULT_SAVE_PATH = Path(__file__).resolve().parents[1] / "data" / "DafnyBench" / "dafnybench.parquet"


def load_dafnybench_data(split: str = "test") -> pl.DataFrame:
    """Loads the DafnyBench dataset into a polars DataFrame."""
    ds = datasets.load_dataset("wendy-sun/DafnyBench")
    table = ds[split].with_format("polars")[:]
    df = pl.DataFrame(table)

    rename_map = {
        "test_ID": "id",
        "test_file": "header",
        "hints_removed": "body",
        "ground_truth": "annotated_body",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    if "id" not in df.columns:
        df = df.with_row_index("id")

    if "header" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("header"))

    if "body" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("body"))

    if "annotated_body" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("annotated_body"))

    if "spec" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("spec"))

    return df


def save_dafnybench_data(data: pl.DataFrame, save_path: Path = DEFAULT_SAVE_PATH) -> None:
    """Saves the DafnyBench dataset to a Parquet file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(save_path, compression="zstd", compression_level=9)


def load_saved_dafnybench_data(load_path: Path = DEFAULT_SAVE_PATH) -> pl.DataFrame:
    """Loads the DafnyBench dataset from a Parquet file."""
    return pl.read_parquet(load_path)


if __name__ == "__main__":
    df = load_dafnybench_data()
    save_dafnybench_data(df)
    print(f"Saved DafnyBench to {DEFAULT_SAVE_PATH} ({len(df)} rows)")
