import datasets
import polars as pl
from pathlib import Path

def load_dafnybench_data() -> pl.DataFrame:
    """Loads the DafnyBench dataset into a polars DataFrame."""
    ds = datasets.load_dataset("wendy-sun/DafnyBench")
    return ds.with_format("polars")['test'][:] # type: ignore

def save_dafnybench_data(data: pl.DataFrame, save_path: Path) -> None:
    """Saves the DafnyBench dataset to a Parquet file."""
    data.write_parquet(save_path, compression="zstd", compression_level=9)

def load_saved_dafnybench_data(load_path: Path) -> pl.DataFrame:
    """Loads the DafnyBench dataset from a Parquet file."""
    return pl.read_parquet(load_path)
