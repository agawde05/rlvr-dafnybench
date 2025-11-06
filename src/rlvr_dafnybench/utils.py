# src/rlvr_dafnybench/utils.py
def load_config(_path: str = None):  # type: ignore
    print("[utils] Using mock config.")
    return {"epochs": 3, "batches": 10}
