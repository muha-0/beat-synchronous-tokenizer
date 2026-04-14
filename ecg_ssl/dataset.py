from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wfdb

from . import config
from .tokenizer import extract_beat_tokens


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

        # reject records with an all-NaN lead
        if np.isnan(x).all(axis=1).any():
            return None

        # repair partial NaNs with per-lead median
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

        # reject flat / near-flat leads
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
# Beat-synchronous dataset (new)
# ---------------------------------------------------------------

class BeatECGDataset(Dataset):
    """
    Dataset that loads ECG recordings and preprocesses them into
    beat-synchronous segments using R-peak detection.

    Each item returns a tensor of resampled beat segments instead of
    the raw signal. R-peak detection happens here (on CPU during data
    loading) rather than inside the model's forward pass.
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

        # Extract beat segments using R-peak detection
        beat_array = extract_beat_tokens(
            x,
            target_length=self.beat_length,
            sampling_rate=self.sampling_rate,
        )

        # Reject if too few beats detected
        if beat_array is None or beat_array.shape[0] < self.min_beats:
            return None

        return beat_array

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            rp = self.record_paths[idx]
            beat_array = self._load_one(rp)

            if beat_array is not None:
                return {
                    "beats": torch.from_numpy(beat_array),  # (N, 12, beat_length)
                    "num_beats": beat_array.shape[0],
                    "record_path": rp,
                }

            idx = np.random.randint(0, len(self.record_paths))

        raise RuntimeError(f"Failed to load a valid ECG after {self.max_retries} retries.")


def beat_collate_fn(batch):
    """
    Custom collate function that pads beat sequences to the same length
    within a batch.

    Different ECGs have different numbers of beats, so we pad shorter
    sequences to match the longest one in the batch. We also create
    a padding mask so the Transformer knows which tokens are real
    and which are padding.

    Args:
        batch: list of dicts from BeatECGDataset.__getitem__

    Returns:
        dict with:
            beats: (B, max_N, 12, beat_length) — padded beat tensors
            padding_mask: (B, max_N) — True where token is padding (to be ignored)
            num_beats: (B,) — actual number of beats per ECG
    """
    num_beats = [item["num_beats"] for item in batch]
    max_beats = max(num_beats)

    B = len(batch)
    C = batch[0]["beats"].shape[1]       # 12 leads
    T = batch[0]["beats"].shape[2]       # beat_length

    # Create padded tensor and padding mask
    padded_beats = torch.zeros(B, max_beats, C, T)
    padding_mask = torch.ones(B, max_beats, dtype=torch.bool)  # True = padding

    for i, item in enumerate(batch):
        n = item["num_beats"]
        padded_beats[i, :n] = item["beats"]
        padding_mask[i, :n] = False  # False = real token, not padding

    return {
        "beats": padded_beats,               # (B, max_N, 12, beat_length)
        "padding_mask": padding_mask,         # (B, max_N)
        "num_beats": torch.tensor(num_beats), # (B,)
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
    """Beat-synchronous dataloaders with padding collation."""
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
