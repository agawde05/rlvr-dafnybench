# src/rlvr_dafnybench/utils.py


def load_config(_path: str = None):  # type: ignore
    print("[utils] Using mock config.")
    return {"epochs": 3, "batches": 10}

def tensor_info(name, t):
    if t is None:
        print(f"{name}: None")
        return
    numel = t.numel()
    mem = numel * t.element_size() / 1024**2
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, size={mem:.2f} MB")