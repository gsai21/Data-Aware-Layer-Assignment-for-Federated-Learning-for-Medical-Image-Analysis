import copy
import torch
import torch.nn as nn
import torch.optim as optim
from .model import get_resnet18
from torch.utils.data import DataLoader

def train_local_full(state_dict: dict, train_loader: DataLoader,
                     epochs: int, lr: float, device) -> tuple[dict, float]:
    """Return (new_state_dict, avg_train_loss)."""
    model = get_resnet18(num_classes=state_dict["fc.weight"].size(0)).to(device)
    model.load_state_dict(state_dict)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    total_loss, total_n = 0.0, 0

    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * yb.size(0)
            total_n += yb.size(0)

    return copy.deepcopy(model.state_dict()), (total_loss / max(total_n,1))

def eval_loss_acc(state_dict: dict, data_loader: DataLoader, device):
    """Return (avg_loss, accuracy)."""
    model = get_resnet18(num_classes=state_dict["fc.weight"].size(0)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss, tot_n, correct = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            tot_loss += float(loss.item()) * yb.size(0)
            tot_n += yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    return (tot_loss / max(tot_n,1)), (0.0 if tot_n==0 else correct/tot_n)