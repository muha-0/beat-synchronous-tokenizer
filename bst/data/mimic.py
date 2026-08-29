import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset

IN_CHANNELS = 12
BEAT_LENGTH = 300
MAX_BEAT_LEN = 600


def sanitize(x, max_missing_frac=0.05):
    x = np.where(np.isinf(x), np.nan, x)

    if np.isnan(x).mean() > max_missing_frac:
        return None

    lead_missing_frac = np.isnan(x).mean(axis=1)
    if (lead_missing_frac > max_missing_frac).any():
        return None

    if np.isnan(x).all(axis=1).any():
        return None

    for c in range(x.shape[0]):
        lead = x[c]
        if np.isnan(lead).any():
            idx = np.arange(len(lead))
            mask = np.isfinite(lead)
            if mask.sum() < 2:
                return None
            x[c] = np.interp(idx, idx[mask], lead[mask])

    if not np.isfinite(x).all():
        return None

    return x


def zscore_per_lead(x, clip_value=5.0):
    x = np.clip(x, -clip_value, clip_value)

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    x = (x - mean) / np.clip(std, 1e-4, None)

    if not np.isfinite(x).all():
        return None

    return x.astype(np.float32, copy=False)


def record_to_cache_name(rp: str) -> str:
    h = hashlib.md5(rp.encode("utf-8")).hexdigest()
    return f"{h}.npy"


def cache_name_npz(npy_path: str) -> str:
    h = hashlib.md5(npy_path.encode("utf-8")).hexdigest()
    return f"{h}.npz"


class PrecomputedECGDataset(Dataset):
    def __init__(self, npy_paths):
        self.npy_paths = npy_paths

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        x = np.load(npy_path, mmap_mode=None)   # shape (12, 5000), float32

        if x.shape != (12, 5000):
            raise ValueError(f"Bad cached shape for {npy_path}: {x.shape}")

        return {
            "x": torch.from_numpy(x),
            "record_path": npy_path,
        }


class PrecomputedBeatDataset(Dataset):
    def __init__(self, beat_paths):
        self.beat_paths = beat_paths

    def __len__(self):
        return len(self.beat_paths)

    def __getitem__(self, idx):
        try:
            beats = np.load(self.beat_paths[idx], mmap_mode=None)  # (N, 12, 300)
            if beats.ndim != 3 or beats.shape[1] != 12 or beats.shape[2] != BEAT_LENGTH:
                raise ValueError(f"Bad shape: {beats.shape}")
            return {
                "beats": torch.from_numpy(beats),
                "num_beats": beats.shape[0],
            }
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def beat_collate_fn(batch):
    num_beats = [item["num_beats"] for item in batch]
    max_beats = max(num_beats)
    B = len(batch)

    padded_beats = torch.zeros(B, max_beats, IN_CHANNELS, BEAT_LENGTH)
    padding_mask = torch.ones(B, max_beats, dtype=torch.bool)   # True = padding

    for i, item in enumerate(batch):
        n = item["num_beats"]
        padded_beats[i, :n] = item["beats"]
        padding_mask[i, :n] = False

    return {
        "beats": padded_beats,                 # (B, max_N, 12, 300)
        "padding_mask": padding_mask,          # (B, max_N)
        "num_beats": torch.tensor(num_beats),  # (B,)
    }


class PrecomputedRawBeatDataset(Dataset):
    def __init__(self, beat_paths):
        self.beat_paths = beat_paths

    def __len__(self):
        return len(self.beat_paths)

    def __getitem__(self, idx):
        try:
            data = np.load(self.beat_paths[idx])
            beats = data["beats"]    # (N, 12, max_beat_len_in_record)
            lengths = data["lengths"]  # (N,)
            if beats.ndim != 3 or beats.shape[1] != 12:
                raise ValueError(f"Bad shape: {beats.shape}")
            return {
                "beats": torch.from_numpy(beats),
                "lengths": torch.from_numpy(lengths),
                "num_beats": beats.shape[0],
            }
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def raw_beat_collate_fn(batch):
    num_beats = [item["num_beats"] for item in batch]
    max_n = max(num_beats)
    max_t = min(max(item["beats"].shape[2] for item in batch), MAX_BEAT_LEN)
    B = len(batch)

    padded_beats = torch.zeros(B, max_n, IN_CHANNELS, max_t)
    padding_mask = torch.ones(B, max_n, dtype=torch.bool)
    beat_lengths = torch.zeros(B, max_n, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_beats"]
        t = min(item["beats"].shape[2], MAX_BEAT_LEN)
        padded_beats[i, :n, :, :t] = item["beats"][:, :, :t]
        padding_mask[i, :n] = False
        beat_lengths[i, :n] = item["lengths"]

    return {
        "beats": padded_beats,                 # (B, max_N, 12, max_T)
        "padding_mask": padding_mask,          # (B, max_N)
        "beat_lengths": beat_lengths,          # (B, max_N)
        "num_beats": torch.tensor(num_beats),  # (B,)
    }


class PrecomputedBeatRRDataset(Dataset):
    def __init__(self, beat_paths):
        self.beat_paths = beat_paths

    def __len__(self):
        return len(self.beat_paths)

    def __getitem__(self, idx):
        try:
            data = np.load(self.beat_paths[idx])
            beats = data["beats"]         # (N, 12, 300)
            rr_intervals = data["rr_intervals"]  # (N,)
            if beats.ndim != 3 or beats.shape[1] != 12 or beats.shape[2] != BEAT_LENGTH:
                raise ValueError(f"Bad shape: {beats.shape}")
            return {
                "beats": torch.from_numpy(beats),
                "rr_intervals": torch.from_numpy(rr_intervals),
                "num_beats": beats.shape[0],
            }
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def beat_rr_collate_fn(batch):
    num_beats = [item["num_beats"] for item in batch]
    max_beats = max(num_beats)
    B = len(batch)

    padded_beats = torch.zeros(B, max_beats, IN_CHANNELS, BEAT_LENGTH)
    padded_rr = torch.zeros(B, max_beats)
    padding_mask = torch.ones(B, max_beats, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["num_beats"]
        padded_beats[i, :n] = item["beats"]
        padded_rr[i, :n] = item["rr_intervals"]
        padding_mask[i, :n] = False

    return {
        "beats": padded_beats,                 # (B, max_N, 12, 300)
        "rr_intervals": padded_rr,             # (B, max_N)
        "padding_mask": padding_mask,          # (B, max_N)
        "num_beats": torch.tensor(num_beats),  # (B,)
    }
