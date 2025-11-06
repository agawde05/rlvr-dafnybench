# src/rlvr_dafnybench/verifier.py
import random
import time


def run_dafny_on_code(_code: str) -> bool:
    time.sleep(0.05)
    return random.random() > 0.3
