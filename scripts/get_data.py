# scripts/get_data.py
import time
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("data/DafnyBench")


def main():
    print("[data] Preparing mock DafnyBench dataset directory...")
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    for _ in tqdm(range(5), desc="[data] Initializing", ncols=80):
        time.sleep(0.2)
    print(f"[data] Done. Directory ready at {DATA_PATH}/")


if __name__ == "__main__":
    main()
