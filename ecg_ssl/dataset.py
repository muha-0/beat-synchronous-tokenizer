from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wfdb

from . import config
from .tokenizer import extract_beat_tokens, extract_beat_tokens_raw


# ---------------------------------------------------------------
# Original fixed-patch dataset (unchanged)
# ---------------------------------------------------------------

class MIMICECGDataset(Dataset):
    def __init__(self, record_paths, clip_value=5.0, max_retries=50):
        self.record_paths = record_paths
        self.clip_value = clip_value
        self.max_retries = max_retries

    def __len__(self):
        return len(self.record_paths)

    def _sanitize(self, x):
        x = np.where(np.isinf(x), np.nan, x)

        if np.isnan(x).all(axis=1).any():
            return None

        for c in range(x.shape[0]):
            lead = x[c]
            if np.isnan(lead).any():
                med = np.nanmedian(lead)
                if np.isnan(med):
                    return None
                x[c] = np.where(np.isnan(lead), med, lead)

        if not np.isfinite(x).all():
            return None

        return x

    def _zscore_per_lead(self, x):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)

        if (std < 1e-4).any():
            return None

        x = (x - mean) / np.clip(std, 1e-4, None)
        x = np.clip(x, -self.clip_value, self.clip_value)

        if not np.isfinite(x).all():
            return None

        return x

    def _load_one(self, rp):
        try:
            rec = wfdb.rdrecord(rp)
        except (FileNotFoundError, OSError, ValueError):
            return None

        x = rec.p_signal.astype(np.float32).T   # (12, 5000)

        if x.shape != (12, 5000):
            return None

        x = self._sanitize(x)
        if x is None:
            return None

        x = self._zscore_per_lead(x)
        if x is None:
            return None

        return x

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            rp = self.record_paths[idx]
            x = self._load_one(rp)

            if x is not None:
                return {
                    "x": torch.from_numpy(x),   # (12, 5000)
                    "record_path": rp,
                }

            idx = np.random.randint(0, len(self.record_paths))

        raise RuntimeError(f"Failed to load a valid ECG after {self.max_retries} retries.")


# ---------------------------------------------------------------
# Shared sanitize mixin for beat datasets
# ---------------------------------------------------------------

class _ECGSanitizeMixin:
    """Shared data cleaning logic for beat datasets."""

    def _sanitize(self, x):
        x = np.where(np.isinf(x), np.nan, x)

        if np.isnan(x).all(axis=1).any():
            return None

        for c in range(x.shape[0]):
            lead = x[c]
            if np.isnan(lead).any():
                med = np.nanmedian(lead)
                if np.isnan(med):
                    return None
                x[c] = np.where(np.isnan(lead), med, lead)

        if not np.isfinite(x).all():
            return None

        return x

    def _zscore_per_lead(self, x, clip_value=5.0):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)

        if (std < 1e-4).any():
            return None

        x = (x - mean) / np.clip(std, 1e-4, None)
        x = np.clip(x, -clip_value, clip_value)

        if not np.isfinite(x).all():
            return None

        return x

    def _load_and_clean(self, rp, clip_value=5.0):
        try:
            rec = wfdb.rdrecord(rp)
        except (FileNotFoundError, OSError, ValueError):
            return None

        x = rec.p_signal.astype(np.float32).T   # (12, 5000)

        if x.shape != (12, 5000):
            return None

        x = self._sanitize(x)
        if x is None:
            return None

        x = self._zscore_per_lead(x, clip_value=clip_value)
        return x


# ---------------------------------------------------------------
# Beat-synchronous dataset for Tokenizer 1 (resampled beats)
# ---------------------------------------------------------------

class BeatECGDataset(_ECGSanitizeMixin, Dataset):
    """
    Dataset for Tokenizer 1 (ResampleCNN).
    Returns resampled beat segments of fixed length.
    """

    def __init__(self, record_paths, beat_length=256, clip_value=5.0,
                 sampling_rate=500, min_beats=3, max_retries=50):
        self.record_paths = record_paths
        self.beat_length = beat_length
        self.clip_value = clip_value
        self.sampling_rate = sampling_rate
        self.min_beats = min_beats
        self.max_retries = max_retries

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            rp = self.record_paths[idx]
            x = self._load_and_clean(rp, self.clip_value)

            if x is not None:
                beat_array = extract_beat_tokens(
                    x,
                    target_length=self.beat_length,
                    sampling_rate=self.sampling_rate,
                )

                if beat_array is not None and beat_array.shape[0] >= self.min_beats:
                    return {
                        "beats": torch.from_numpy(beat_array),  # (N, 12, beat_length)
                        "num_beats": beat_array.shape[0],
                        "record_path": rp,
                    }

            idx = np.random.randint(0, len(self.record_paths))

        raise RuntimeError(f"Failed to load a valid ECG after {self.max_retries} retries.")


# ---------------------------------------------------------------
# Beat-synchronous dataset for Tokenizer 2 (raw variable-length beats)
# ---------------------------------------------------------------

class RawBeatECGDataset(_ECGSanitizeMixin, Dataset):
    """
    Dataset for Tokenizer 2 (AdaptivePoolingCNN).
    Returns raw variable-length beat segments with NO resampling.

    Each beat is padded to the max beat length within that single ECG.
    Cross-ECG padding (different number of beats, different max beat lengths)
    is handled by the collate function.
    """

    def __init__(self, record_paths, clip_value=5.0,
                 sampling_rate=500, min_beats=3, max_retries=50):
        self.record_paths = record_paths
        self.clip_value = clip_value
        self.sampling_rate = sampling_rate
        self.min_beats = min_beats
        self.max_retries = max_retries

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            rp = self.record_paths[idx]
            x = self._load_and_clean(rp, self.clip_value)

            if x is not None:
                beat_array, beat_lengths = extract_beat_tokens_raw(
                    x,
                    sampling_rate=self.sampling_rate,
                )

                if (beat_array is not None and
                        beat_lengths is not None and
                        beat_array.shape[0] >= self.min_beats):
                    return {
                        "beats": torch.from_numpy(beat_array),           # (N, 12, max_beat_len)
                        "beat_lengths": torch.from_numpy(beat_lengths),  # (N,)
                        "num_beats": beat_array.shape[0],
                        "max_beat_len": beat_array.shape[2],
                        "record_path": rp,
                    }

            idx = np.random.randint(0, len(self.record_paths))

        raise RuntimeError(f"Failed to load a valid ECG after {self.max_retries} retries.")


# ---------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------

def beat_collate_fn(batch):
    """
    Collate for Tokenizer 1 (BeatECGDataset).
    Pads across number of beats only (beat length is fixed via resampling).
    """
    num_beats = [item["num_beats"] for item in batch]
    max_beats = max(num_beats)

    B = len(batch)
    C = batch[0]["beats"].shape[1]       # 12 leads
    T = batch[0]["beats"].shape[2]       # beat_length (fixed = 256)

    padded_beats = torch.zeros(B, max_beats, C, T)
    padding_mask = torch.ones(B, max_beats, dtype=torch.bool)  # True = padding

    for i, item in enumerate(batch):
        n = item["num_beats"]
        padded_beats[i, :n] = item["beats"]
        padding_mask[i, :n] = False

    return {
        "beats": padded_beats,               # (B, max_N, 12, beat_length)
        "padding_mask": padding_mask,         # (B, max_N)
        "num_beats": torch.tensor(num_beats), # (B,)
    }


def raw_beat_collate_fn(batch):
    """
    Collate for Tokenizer 2 (RawBeatECGDataset).
    Pads on TWO dimensions:
      1. Number of beats (different ECGs have different numbers of heartbeats)
      2. Beat length (different beats have different lengths, even within one ECG)

    Returns:
        beats: (B, max_N, 12, global_max_T) — double-padded beat tensors
        padding_mask: (B, max_N) — True where beat position is padding
        num_beats: (B,) — actual number of beats per ECG
        beat_lengths: (B, max_N) — actual sample length of each beat (0 for padding beats)
        global_max_beat_len: int — the max beat length across the entire batch
    """
    num_beats = [item["num_beats"] for item in batch]
    max_beats = max(num_beats)

    # Find the global max beat length across ALL beats in ALL ECGs in this batch
    global_max_beat_len = max(item["max_beat_len"] for item in batch)

    B = len(batch)
    C = batch[0]["beats"].shape[1]  # 12 leads

    # Double-padded tensor: pad both beat count and beat length
    padded_beats = torch.zeros(B, max_beats, C, global_max_beat_len)
    padding_mask = torch.ones(B, max_beats, dtype=torch.bool)  # True = padding
    all_beat_lengths = torch.zeros(B, max_beats, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item["num_beats"]
        t = item["beats"].shape[2]  # this ECG's max beat length

        # Copy beats into the padded tensor (may need additional length padding)
        padded_beats[i, :n, :, :t] = item["beats"]
        padding_mask[i, :n] = False
        all_beat_lengths[i, :n] = item["beat_lengths"]

    return {
        "beats": padded_beats,                          # (B, max_N, 12, global_max_T)
        "padding_mask": padding_mask,                   # (B, max_N)
        "num_beats": torch.tensor(num_beats),           # (B,)
        "beat_lengths": all_beat_lengths,                # (B, max_N)
        "global_max_beat_len": global_max_beat_len,     # int
    }


# ---------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------

def build_dataloaders(record_paths):
    """Original fixed-patch dataloaders."""
    dataset = MIMICECGDataset(record_paths, max_retries=config.MAX_RETRIES)

    n_total = len(dataset)
    n_train = int(config.TRAIN_FRAC * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader


def build_beat_dataloaders(record_paths):
    """Tokenizer 1 dataloaders (resampled beats)."""
    dataset = BeatECGDataset(
        record_paths,
        beat_length=config.BEAT_LENGTH,
        max_retries=config.MAX_RETRIES,
    )

    n_total = len(dataset)
    n_train = int(config.TRAIN_FRAC * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        collate_fn=beat_collate_fn,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader


def build_raw_beat_dataloaders(record_paths):
    """Tokenizer 2 dataloaders (raw variable-length beats)."""
    dataset = RawBeatECGDataset(
        record_paths,
        max_retries=config.MAX_RETRIES,
    )

    n_total = len(dataset)
    n_train = int(config.TRAIN_FRAC * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.SEED),
    )

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        collate_fn=raw_beat_collate_fn,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader
