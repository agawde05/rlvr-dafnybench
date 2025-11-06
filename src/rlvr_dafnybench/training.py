# src/rlvr_dafnybench/training.py
import time
from tqdm import tqdm


def main():
    print("[train] Starting mock training...")
    for ep in range(1, 2):
        for _ in tqdm(range(10), desc=f"[train] Epoch {ep}", ncols=80):
            time.sleep(0.05)
        print(f"[train] Finished epoch {ep}")
    print("[train] Done. (mock checkpoint saved)")
