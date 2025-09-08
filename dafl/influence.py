import numpy as np
import torch
import torch.nn as nn
from typing import List

def make_logical_groups(state_dict_keys: list[str]) -> list[str]:
    prefixes = []
    for k in state_dict_keys:
        parts = k.split(".")
        if parts[0] in ("conv1","bn1","fc"):
            pref = parts[0]
        else:
            pref = ".".join(parts[:3]) if len(parts) >= 3 else parts[0]
        if pref not in prefixes:
            prefixes.append(pref)
    return prefixes

def keys_for_prefix(prefix: str, keys: list[str]) -> list[str]:
    return [k for k in keys if (k == prefix or k.startswith(prefix + "."))]

def compute_group_influence(model: nn.Module, test_loader, groups: List[List[str]],
                            device, max_batches: int) -> list[float]:
    model.to(device); model.train()
    crit = nn.CrossEntropyLoss()
    gi = np.zeros(len(groups), dtype=float)
    batch_count = 0
    for i, (xb, yb) in enumerate(test_loader):
        if i >= max_batches: break
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        loss = crit(model(xb), yb); loss.backward()
        named = dict(model.named_parameters())
        for g_idx, group_keys in enumerate(groups):
            s = 0.0
            for pname, p in named.items():
                if any(pname == k or pname.startswith(k) for k in group_keys):
                    if p.grad is not None:
                        s += float(torch.norm(p.grad).item())
            gi[g_idx] += s
        batch_count += 1
    if batch_count == 0:
        return list(gi.tolist())
    return list((gi / float(batch_count)).tolist())