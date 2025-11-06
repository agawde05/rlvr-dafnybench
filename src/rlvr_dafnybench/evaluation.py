# src/rlvr_dafnybench/evaluation.py
import time
import random
from tqdm import tqdm


def main():
    print("[eval] Running mock evaluation...")
    total, ok = 10, 0
    for _ in tqdm(range(total), desc="[eval] Evaluating", ncols=80):
        ok += 1 if random.random() > 0.1 else 0
        time.sleep(0.05)
    print(f"[eval] Accuracy: {ok / total * 100:.2f}%")
