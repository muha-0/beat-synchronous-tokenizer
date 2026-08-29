import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from .beats import (
    extract_and_resample_beats,
    extract_beats_with_rr,
    extract_raw_beats_padded,
)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

IN_CHANNELS = 12
BEAT_LENGTH = 300
MAX_BEAT_LEN = 600


def load_ptbxl_splits(ptbxl_base):
    """Official PTB-XL stratified splits: folds 1-8 train, 9 val, 10 test."""
    ptbxl_base = Path(ptbxl_base)
    label_df = pd.read_csv(ptbxl_base / "ptbxl_database.csv")
    scp_df = pd.read_csv(ptbxl_base / "scp_statements.csv", index_col=0)

    label_df["scp_codes"] = label_df["scp_codes"].apply(ast.literal_eval)

    scp_to_superclass = {}
    for scp_code, row in scp_df.iterrows():
        if row["diagnostic_class"] in SUPERCLASSES:
            scp_to_superclass[scp_code] = row["diagnostic_class"]

    def get_superclass_labels(scp_codes_dict):
        labels = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for scp_code in scp_codes_dict.keys():
            if scp_code in scp_to_superclass:
                superclass = scp_to_superclass[scp_code]
                idx = SUPERCLASSES.index(superclass)
                labels[idx] = 1.0
        return labels

    label_df["target"] = label_df["scp_codes"].apply(get_superclass_labels)
    label_df = label_df[label_df["target"].apply(lambda x: x.sum() > 0)].copy()

    label_df["ecg_path"] = label_df.apply(lambda row: ptbxl_base / row["filename_hr"], axis=1)

    def files_exist(row):
        base = Path(row["ecg_path"])
        return Path(str(base) + ".hea").exists() and Path(str(base) + ".dat").exists()

    label_df["file_exists"] = label_df.apply(files_exist, axis=1)
    label_df = label_df[label_df["file_exists"]].copy()

    train_df = label_df[label_df["strat_fold"] <= 8].copy()
    val_df = label_df[label_df["strat_fold"] == 9].copy()
    test_df = label_df[label_df["strat_fold"] == 10].copy()

    return train_df, val_df, test_df


def _load_and_normalize(record_path):
    record = wfdb.rdrecord(record_path)
    x = record.p_signal.astype(np.float32).T   # (12, 5000)

    x = np.clip(x, -5, 5)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x = (x - mean) / np.clip(std, 1e-4, None)
    return x


class PTBXLDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_and_normalize(str(row["ecg_path"]))

        x = torch.from_numpy(x).float()
        y = torch.tensor(row["target"], dtype=torch.float32)   # (5,)
        return x, y


class PTBXLBeatDataset(Dataset):
    def __init__(self, df, beat_length=300):
        self.df = df.reset_index(drop=True)
        self.beat_length = beat_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_and_normalize(str(row["ecg_path"]))

        beats = extract_and_resample_beats(x, beat_length=self.beat_length)
        if beats is None:
            return self.__getitem__((idx + 1) % len(self))

        y = torch.tensor(row["target"], dtype=torch.float32)   # (5,)
        return torch.from_numpy(beats), y


def beat_collate_fn(batch):
    beats_list, labels = zip(*batch)
    num_beats = [b.shape[0] for b in beats_list]
    max_n = max(num_beats)
    B = len(beats_list)

    padded_beats = torch.zeros(B, max_n, IN_CHANNELS, BEAT_LENGTH)
    padding_mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, beats in enumerate(beats_list):
        n = beats.shape[0]
        padded_beats[i, :n] = beats
        padding_mask[i, :n] = False

    return padded_beats, padding_mask, torch.stack(labels)


class PTBXLRawBeatDataset(Dataset):
    def __init__(self, df, max_beat_len=600):
        self.df = df.reset_index(drop=True)
        self.max_beat_len = max_beat_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_and_normalize(str(row["ecg_path"]))

        beats = extract_raw_beats_padded(x, max_beat_len=self.max_beat_len)
        if beats is None:
            return self.__getitem__((idx + 1) % len(self))

        y = torch.tensor(row["target"], dtype=torch.float32)   # (5,)
        return torch.from_numpy(beats), y


def raw_beat_collate_fn(batch):
    beats_list, labels = zip(*batch)
    num_beats = [b.shape[0] for b in beats_list]
    max_n = max(num_beats)
    B = len(beats_list)

    padded_beats = torch.zeros(B, max_n, IN_CHANNELS, MAX_BEAT_LEN)
    padding_mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, beats in enumerate(beats_list):
        n = beats.shape[0]
        padded_beats[i, :n] = beats
        padding_mask[i, :n] = False

    return padded_beats, padding_mask, torch.stack(labels)


class PTBXLBeatHRDataset(Dataset):
    def __init__(self, df, beat_length=300):
        self.df = df.reset_index(drop=True)
        self.beat_length = beat_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_and_normalize(str(row["ecg_path"]))

        beats, rr_intervals = extract_beats_with_rr(x, beat_length=self.beat_length)
        if beats is None:
            return self.__getitem__((idx + 1) % len(self))

        y = torch.tensor(row["target"], dtype=torch.float32)   # (5,)
        return torch.from_numpy(beats), torch.from_numpy(rr_intervals), y


def beat_hr_collate_fn(batch):
    beats_list, rr_list, labels = zip(*batch)
    num_beats = [b.shape[0] for b in beats_list]
    max_n = max(num_beats)
    B = len(beats_list)

    padded_beats = torch.zeros(B, max_n, IN_CHANNELS, BEAT_LENGTH)
    padded_rr = torch.zeros(B, max_n)
    padding_mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, (beats, rr) in enumerate(zip(beats_list, rr_list)):
        n = beats.shape[0]
        padded_beats[i, :n] = beats
        padded_rr[i, :n] = rr
        padding_mask[i, :n] = False

    return padded_beats, padded_rr, padding_mask, torch.stack(labels)
