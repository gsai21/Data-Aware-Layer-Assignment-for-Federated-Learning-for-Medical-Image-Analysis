import os, json, time, random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_out_dir(base: str | None = None):
    base = base or "artifacts"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(base, f"run-{stamp}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    return out

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def bytes_for_params(state_dict: dict, dtype_bytes: int = 4) -> int:
    return sum(int(t.numel()) * dtype_bytes for t in state_dict.values())
