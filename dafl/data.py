import os, random
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def default_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def load_imagefolder_splits(dataset_path: str, num_workers: int):
    train_tf, eval_tf = default_transforms()
    train_folder = os.path.join(dataset_path, "train")
    test_folder  = os.path.join(dataset_path, "test")
    assert os.path.isdir(train_folder) and os.path.isdir(test_folder), \
        f"Expected {train_folder} and {test_folder}"
    full_train = datasets.ImageFolder(train_folder, transform=train_tf)
    test_set   = datasets.ImageFolder(test_folder, transform=eval_tf)
    return full_train, test_set

def uneven_split_lengths(total_len: int, num_clients: int,
                         min_frac=0.02, max_frac=0.15, seed=42) -> List[int]:
    rng = np.random.default_rng(seed)
    min_size = max(1, int(min_frac * total_len))
    max_size = max(1, int(max_frac * total_len))
    lengths, remaining, remaining_clients = [], total_len, num_clients
    for _ in range(num_clients-1):
        max_allowed = min(max_size, remaining - (remaining_clients - 1) * min_size)
        min_allowed = max(min_size, remaining - (remaining_clients - 1) * max_allowed)
        chosen = int(rng.integers(min_allowed, max_allowed+1))
        lengths.append(chosen); remaining -= chosen; remaining_clients -= 1
    lengths.append(int(remaining))
    random.shuffle(lengths)
    return lengths

def make_client_loaders(full_train, num_clients: int, batch_size: int,
                        num_workers: int, seed: int=42):
    total_len = len(full_train)
    lengths = uneven_split_lengths(total_len, num_clients, seed=seed)
    all_idx = list(range(total_len)); random.shuffle(all_idx)
    shards = []
    cursor = 0
    for L in lengths:
        shards.append(all_idx[cursor: cursor + L]); cursor += L

    # validation uses eval transforms (no augmentation)
    eval_tf = default_transforms()[1]
    val_full = datasets.ImageFolder(full_train.root, transform=eval_tf)

    client_train, client_val, counts = [], [], []
    for shard in shards:
        n = len(shard); counts.append(n)
        if n == 0:
            client_train.append(None); client_val.append(None); continue
        val_n = max(1, int(0.2*n)); train_n = n - val_n
        tr_idx, va_idx = shard[:train_n], shard[train_n:]
        train_subset = Subset(full_train, tr_idx)
        val_subset   = Subset(val_full,   va_idx)
        tr_loader = DataLoader(train_subset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
        va_loader = DataLoader(val_subset,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
        client_train.append(tr_loader); client_val.append(va_loader)
    return client_train, client_val, counts

def make_test_loader(test_set, batch_size: int, num_workers: int):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
