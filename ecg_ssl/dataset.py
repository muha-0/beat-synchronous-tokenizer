from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import wfdb

from . import config


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


def build_dataloaders(record_paths):
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
