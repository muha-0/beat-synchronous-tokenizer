import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import wfdb
from scipy.signal import resample
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm

FS = 250
WINDOW_SEC = 60
WINDOW_SAMPLES = FS * WINDOW_SEC      # 15000
PURITY_THRESH = 0.90
BEAT_LENGTH = 160                     # matches contrastive patch_len
MAX_BEATS = 93                        # covers 95.4% of valid 60s windows
VALID_BEAT_SYMBOLS = {"N", "S", "V"}
LABEL2ID = {"N": 0, "AFIB": 1, "AFL": 1}    # binary: N=0, AF=1

RHY_MAP = {
    "N": "N",
    "AFIB": "AFIB",
    "AFL": "AFL",
    "AFLUT": "AFL",
    "AFLUTTER": "AFL",
}

_b, _a = scipy.signal.butter(
    4, [0.5 / (0.5 * FS), 40.0 / (0.5 * FS)], btype="bandpass"
)


def list_patients(root: Path):
    # patient folders like p00/p00000
    patients = []
    for sub in sorted(Path(root).glob("p*/p*")):
        if sub.is_dir() and sub.name.startswith("p"):
            patients.append(sub)
    return patients


def list_record_bases(patient_dir: Path):
    recs = []
    for h in sorted(patient_dir.glob("*.hea")):
        base = h.with_suffix("")
        if base.with_suffix(".dat").exists() and base.with_suffix(".atr").exists():
            recs.append(base)
    return recs


def normalize_rhythm_token(tok: str):
    tok = tok.strip().upper().replace("(", "").replace(")", "").strip()
    return RHY_MAP.get(tok, None)


def build_rhythm_intervals_from_ann(ann, sig_len: int):
    """Parse WFDB rhythm annotations into non-overlapping labeled intervals.

    Returns: list of (start, end, label) in samples, with end > start.
    """
    intervals = []
    cur_label = None
    cur_start = None

    def close_at(end_s):
        nonlocal cur_label, cur_start, intervals
        if cur_start is not None and cur_label is not None and end_s is not None and end_s > cur_start:
            intervals.append((int(cur_start), int(end_s), cur_label))
        cur_label = None
        cur_start = None

    for s, note in zip(ann.sample, ann.aux_note):
        if note is None:
            continue
        note = str(note).strip()
        if note == "" or note.upper() == "NONE":
            continue
        s = int(s)
        if note.startswith("("):
            if cur_start is not None:
                close_at(s)
            lab = normalize_rhythm_token(note)
            if lab is not None:
                cur_label = lab
                cur_start = s
            else:
                cur_label = None
                cur_start = None
        elif note.startswith(")"):
            close_at(s)

    if cur_start is not None and cur_label is not None:
        close_at(sig_len)

    return intervals


def label_window_by_occupancy(intervals, w_start, w_end, thresh=0.6):
    """SSL window labeling: intervals (start, end, label) in samples, window [w_start, w_end)."""
    dur = w_end - w_start
    if dur <= 0:
        return None

    occ = {"AFIB": 0, "AFL": 0}  # N is default remainder
    for a, b, lab in intervals:
        if b <= w_start or a >= w_end:
            continue
        overlap = max(0, min(b, w_end) - max(a, w_start))
        if lab in occ:
            occ[lab] += overlap

    if occ["AFIB"] / dur >= thresh:
        return "AFIB"
    if occ["AFL"] / dur >= thresh:
        return "AFL"
    return "N"


def window_purity_label(intervals, w_start, w_end, purity_thresh=0.90):
    """Downstream window labeling with coverage + dominant rhythm purity threshold."""
    dur = w_end - w_start
    occ = {"AFIB": 0, "AFL": 0, "N": 0}
    for a, b, lab in intervals:
        if b <= w_start or a >= w_end:
            continue
        overlap = max(0, min(b, w_end) - max(a, w_start))
        if lab in occ:
            occ[lab] += overlap
    covered = sum(occ.values())
    if covered / dur < purity_thresh:
        return None
    dominant = max(occ, key=occ.get)
    if occ[dominant] / dur < purity_thresh:
        return None
    return dominant   # "N", "AFIB", or "AFL"


def split_patients(patient_dirs, ssl_frac=0.8, train_frac=0.1, val_frac=0.05, seed=42):
    rng = np.random.default_rng(seed)
    patient_dirs = list(patient_dirs)
    rng.shuffle(patient_dirs)

    n = len(patient_dirs)
    n_ssl = int(n * ssl_frac)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    ssl = patient_dirs[:n_ssl]
    train = patient_dirs[n_ssl: n_ssl + n_train]
    val = patient_dirs[n_ssl + n_train: n_ssl + n_train + n_val]
    test = patient_dirs[n_ssl + n_train + n_val:]

    return ssl, train, val, test


def load_signal(rec_base_str: str):
    sig, _ = wfdb.rdsamp(rec_base_str)
    sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
    return scipy.signal.filtfilt(_b, _a, sig).astype(np.float32)


def preprocess_to_memmap(patient_dirs, save_root, root, fs=250):
    """Bandpass-filter every record once and cache as .npy (run once before SSL)."""
    save_root = Path(save_root)
    root = Path(root)
    os.makedirs(save_root, exist_ok=True)

    lowcut, highcut = 0.5, 40.0
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [lowcut / nyq, highcut / nyq], btype='bandpass')

    print(f"Preprocessing {len(patient_dirs)} patients...")
    for pd in tqdm(patient_dirs):
        rel_path = pd.relative_to(root)
        (save_root / rel_path).mkdir(parents=True, exist_ok=True)

        recs = list_record_bases(pd)
        for rec in recs:
            save_path = save_root / rel_path / f"{rec.name}_filtered.npy"
            if save_path.exists():
                continue
            try:
                sig, _ = wfdb.rdsamp(str(rec))
                sig = sig[:, 0].astype(np.float32) if sig.ndim == 2 else sig.squeeze().astype(np.float32)
                sig = scipy.signal.filtfilt(b, a, sig).astype(np.float32)
                np.save(str(save_path), sig)
            except Exception as e:
                print(f"Error processing {rec}: {e}")


class ECGAugment:
    """Realistic ECG augmentations for contrastive pretraining.
    Designed to avoid shortcut cues (e.g., hard zeros) and preserve morphology."""

    def __init__(
        self,
        fs: int = 250,
        p_scale: float = 1.0,
        scale_min: float = 0.90,
        scale_max: float = 1.10,
        p_white_noise: float = 1.0,
        white_noise_std: float = 0.005,
        p_baseline: float = 0.7,
        baseline_amp: float = 0.10,
        baseline_f_lo: float = 0.05,
        baseline_f_hi: float = 0.50,
        p_emg: float = 0.5,
        emg_std: float = 0.01,
        p_mask: float = 0.5,
        n_masks_min: int = 0,
        n_masks_max: int = 1,
        mask_len_min_sec: float = 0.10,
        mask_len_max_sec: float = 0.50,
        mask_depth_min: float = 0.7,
        mask_depth_max: float = 1.0,
        seed=None,
    ):
        self.fs = fs

        self.p_scale = p_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.p_white_noise = p_white_noise
        self.white_noise_std = white_noise_std

        self.p_baseline = p_baseline
        self.baseline_amp = baseline_amp
        self.baseline_f_lo = baseline_f_lo
        self.baseline_f_hi = baseline_f_hi

        self.p_emg = p_emg
        self.emg_std = emg_std

        self.p_mask = p_mask
        self.n_masks_min = n_masks_min
        self.n_masks_max = n_masks_max
        self.mask_len_min = int(mask_len_min_sec * fs)
        self.mask_len_max = int(mask_len_max_sec * fs)
        self.mask_depth_min = mask_depth_min
        self.mask_depth_max = mask_depth_max

        self.rng = np.random.default_rng(seed)

    def _as_2d(self, x):
        if x.ndim == 1:
            return x[:, None], True
        return x, False

    def _smooth_mask(self, T, start, m, depth):
        # raised-cosine multiplicative mask on [start, start+m)
        mask = np.ones((T,), dtype=np.float32)
        end = min(T, start + m)
        m_eff = end - start
        if m_eff <= 1:
            return mask

        t = np.linspace(0, np.pi, m_eff, dtype=np.float32)
        w = 0.5 - 0.5 * np.cos(t)  # 0..1
        mask[start:end] = 1.0 - depth * w
        return mask

    def _preprocess(self, x):
        x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-6)
        return x.astype(np.float32)

    def __call__(self, x):
        y, was_1d = self._as_2d(x.astype(np.float32, copy=True))
        y = self._preprocess(y)
        T, C = y.shape

        sig_std = float(np.std(y)) + 1e-6

        # gain scaling
        if self.rng.random() < self.p_scale:
            s = self.rng.uniform(self.scale_min, self.scale_max)
            y *= np.float32(s)

        # baseline wander: sum of a few low-freq sinusoids
        if self.rng.random() < self.p_baseline:
            t = np.arange(T, dtype=np.float32) / np.float32(self.fs)
            n_comp = int(self.rng.integers(1, 4))
            drift = np.zeros((T,), dtype=np.float32)
            for _ in range(n_comp):
                f = self.rng.uniform(self.baseline_f_lo, self.baseline_f_hi)
                phase = self.rng.uniform(0, 2 * np.pi)
                drift += np.sin(2 * np.pi * f * t + phase).astype(np.float32)

            drift /= (np.max(np.abs(drift)) + 1e-6)
            amp = np.float32(self.baseline_amp * sig_std)
            y += drift[:, None] * amp

        # EMG-like noise via differenced white noise
        if self.rng.random() < self.p_emg:
            n = self.rng.normal(0.0, 1.0, size=(T, C)).astype(np.float32)
            n = np.concatenate([n[:1], np.diff(n, axis=0)], axis=0)
            y += n * np.float32(self.emg_std * sig_std)

        # small white noise
        if self.rng.random() < self.p_white_noise:
            y += self.rng.normal(0.0, 1.0, size=y.shape).astype(np.float32) * np.float32(self.white_noise_std * sig_std)

        # structured smooth masking (no hard zeros)
        if self.rng.random() < self.p_mask:
            n_masks = int(self.rng.integers(self.n_masks_min, self.n_masks_max + 1))
            for _ in range(n_masks):
                m = int(self.rng.integers(max(2, self.mask_len_min), max(3, self.mask_len_max + 1)))
                start = int(self.rng.integers(0, max(1, T - m)))
                depth = float(self.rng.uniform(self.mask_depth_min, self.mask_depth_max))
                mult = self._smooth_mask(T, start, m, depth)[:, None]
                y *= mult

        return y[:, 0] if was_1d else y


class IcentiaPatientPairDataset(Dataset):
    """Positive pairs: two 60s windows from different records of the same patient."""

    def __init__(self, patient_dirs, window_samples, processed_root, root,
                 fs=250, augment=None, seed=0, cache_size=64):
        self.patient_dirs = list(patient_dirs)
        self.window_samples = window_samples
        self.processed_root = Path(processed_root)
        self.root = Path(root)
        self.fs = fs
        self.augment = augment
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]

        self._cache_size = int(cache_size)
        self._init_record_cache()

    def __len__(self):
        return len(self.patient_dirs)

    def _init_record_cache(self):

        @lru_cache(maxsize=self._cache_size)
        def _cached_load(patient_folder_name, rec_name):
            parent_folder = patient_folder_name[:3]
            npy_path = self.processed_root / parent_folder / patient_folder_name / f"{rec_name}_filtered.npy"
            rec_base_path = self.root / parent_folder / patient_folder_name / rec_name

            sig = np.load(str(npy_path), mmap_mode='r')
            ann = wfdb.rdann(str(rec_base_path), extension="atr")
            intervals = build_rhythm_intervals_from_ann(ann, len(sig))

            return sig, intervals

        self._load_record_cached = _cached_load

    def _sample_valid_window(self, sig, intervals, max_tries=50):
        sig_len = len(sig)
        if sig_len <= self.window_samples:
            return None, None, None

        for _ in range(max_tries):
            w_start = int(self.rng.integers(0, sig_len - self.window_samples))
            w_end = w_start + self.window_samples

            chunk = sig[w_start:w_end]

            # QC: reject windows with NaNs or flat-lines
            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            lab = label_window_by_occupancy(intervals, w_start, w_end, thresh=0.6)
            if lab is not None:
                return w_start, w_end, lab

        return None, None, None

    def __getitem__(self, idx):
        patient_path = self.patient_dirs[idx]
        p_folder = patient_path.name  # e.g. "p00001"
        recs = self.patient_records[idx]

        if len(recs) >= 2:
            r1, r2 = self.rng.choice(len(recs), size=2, replace=False)
            rec_path1 = recs[int(r1)]
            rec_path2 = recs[int(r2)]
        else:
            rec_path1 = recs[0]
            rec_path2 = recs[0]

        sig1, intervals1 = self._load_record_cached(p_folder, rec_path1.name)
        res1 = self._sample_valid_window(sig1, intervals1)
        if res1[0] is None:
            return self.__getitem__(self.rng.integers(0, len(self)))

        s1, e1, y1 = res1
        x_raw1 = np.array(sig1[s1:e1])

        sig2, intervals2 = self._load_record_cached(p_folder, rec_path2.name)
        res2 = self._sample_valid_window(sig2, intervals2)

        if res2[0] is None:
            x_raw2 = x_raw1.copy()
            y2 = y1
        else:
            s2, e2, y2 = res2
            x_raw2 = np.array(sig2[s2:e2])

        if self.augment is not None:
            x1 = self.augment(x_raw1)
            x2 = self.augment(x_raw2)
        else:
            x1, x2 = x_raw1.copy(), x_raw2.copy()

        return {
            "patient_id": p_folder,
            "rec1": rec_path1.name,
            "rec2": rec_path2.name,
            "x1": torch.from_numpy(x1),
            "x2": torch.from_numpy(x2),
            "y": y1,
        }


def collate_patient_pairs(batch):
    xs = []
    pair_ids = []
    patient_ids = []
    rec1_names = []
    rec2_names = []
    ys = []

    for i, item in enumerate(batch):
        xs.append(item["x1"])
        xs.append(item["x2"])
        pair_ids.extend([i, i])
        patient_ids.append(item["patient_id"])
        rec1_names.append(item.get("rec1", item.get("rec", "NA")))
        rec2_names.append(item.get("rec2", item.get("rec", "NA")))
        ys.append(item["y"])

    x = torch.stack(xs, dim=0)
    pair_ids = torch.tensor(pair_ids, dtype=torch.long)
    return {
        "x": x,
        "pair_ids": pair_ids,
        "patient_ids": patient_ids,
        "rec1": rec1_names,
        "rec2": rec2_names,
        "y": ys,
    }


def seed_worker(worker_id: int):
    info = get_worker_info()
    ds = info.dataset

    seed = (torch.initial_seed() + worker_id) % (2**32)

    ds.rng = np.random.default_rng(seed)

    if getattr(ds, "augment", None) is not None and hasattr(ds.augment, "rng"):
        ds.augment.rng = np.random.default_rng(seed + 12345)


def extract_beats_and_rr_from_ann(sig_window, ann, w_start, w_end,
                                  beat_length=160, min_beats=3):
    r_peaks = sorted([
        int(s) - w_start
        for s, sym in zip(ann.sample, ann.symbol)
        if w_start <= int(s) < w_end and sym in VALID_BEAT_SYMBOLS
    ])

    if len(r_peaks) < 2:
        return None, None

    beats = []
    rr_intervals = []

    for i in range(len(r_peaks) - 1):
        start, end = r_peaks[i], r_peaks[i + 1]
        beat = sig_window[start:end]
        if len(beat) < 10:
            continue
        beats.append(resample(beat, beat_length).astype(np.float32))
        rr_intervals.append((end - start) / FS)   # in seconds

    if len(beats) < min_beats:
        return None, None

    beat_arr = np.stack(beats, axis=0)[:, np.newaxis, :]   # [N, 1, beat_length]
    rr_arr = np.array(rr_intervals, dtype=np.float32)      # [N]
    return beat_arr, rr_arr


def build_manifest(patient_dirs, purity_thresh=0.90, min_beats=3,
                   windows_per_patient=2, max_tries=200, seed=42):
    """Frozen val/test window manifests shared by all tokenizers."""
    rng = np.random.default_rng(seed)
    manifest = []

    for i, pd in enumerate(patient_dirs):
        recs = list_record_bases(pd)
        if not recs:
            continue

        collected = []
        tries = 0

        while len(collected) < windows_per_patient and tries < max_tries:
            tries += 1

            rec = recs[int(rng.integers(0, len(recs)))]
            try:
                sig = load_signal(str(rec))
                ann = wfdb.rdann(str(rec), extension="atr")
                ivs = build_rhythm_intervals_from_ann(ann, len(sig))
            except Exception:
                continue

            sig_len = len(sig)
            if sig_len <= WINDOW_SAMPLES:
                continue

            w_start = int(rng.integers(0, sig_len - WINDOW_SAMPLES))
            w_end = w_start + WINDOW_SAMPLES

            lab = window_purity_label(ivs, w_start, w_end, purity_thresh)
            if lab is None:
                continue

            chunk = sig[w_start:w_end]
            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            x = (chunk - chunk.mean()) / (chunk.std() + 1e-6)
            beats, rr = extract_beats_and_rr_from_ann(
                x, ann, w_start, w_end, BEAT_LENGTH, min_beats
            )
            if beats is None:
                continue

            collected.append({
                "rec": str(rec),
                "w_start": int(w_start),
                "label": int(LABEL2ID[lab]),
            })

        manifest.extend(collected)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(patient_dirs)} patients | {len(manifest)} windows so far")

    return manifest


class IcentiaManifestDataset(Dataset):
    """Fixed deterministic dataset for val/test.
    All tokenizers index into the same manifest entries.
    mode: 'fixed' | 'beat' | 'beat_hr'"""

    def __init__(self, manifest, mode="fixed", cache_size=128):
        self.manifest = manifest
        self.mode = mode
        self._init_cache(cache_size)

    def __len__(self):
        return len(self.manifest)

    def _init_cache(self, cache_size):
        @lru_cache(maxsize=cache_size)
        def _load(rec_str):
            sig = load_signal(rec_str)
            ann = wfdb.rdann(rec_str, extension="atr")
            return sig, ann
        self._load = _load

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        sig, ann = self._load(entry["rec"])
        w_start = entry["w_start"]
        w_end = w_start + WINDOW_SAMPLES

        chunk = sig[w_start:w_end]
        x = (chunk - chunk.mean()) / (chunk.std() + 1e-6)
        y = torch.tensor(entry["label"], dtype=torch.long)

        if self.mode == "fixed":
            return torch.from_numpy(x.astype(np.float32)), y

        beats, rr = extract_beats_and_rr_from_ann(
            x, ann, w_start, w_end, BEAT_LENGTH
        )

        if self.mode == "beat":
            return torch.from_numpy(beats), y

        return torch.from_numpy(beats), torch.from_numpy(rr), y


class IcentiaTrainDataset(Dataset):
    """Training dataset with per-epoch deterministic sampling.
    Each patient gets one window per epoch, determined by:
        rng = seed + patient_idx * 1000003 + epoch * 999983
    Different epochs -> different windows. Same epoch -> same windows.
    p_af controls class balancing.
    mode: 'fixed' | 'beat' | 'beat_hr'"""

    def __init__(self, patient_dirs, mode="fixed", p_af=0.5,
                 max_tries=80, seed=0, cache_size=64):
        self.patient_dirs = list(patient_dirs)
        self.patient_records = [list_record_bases(pd) for pd in self.patient_dirs]
        self.mode = mode
        self.p_af = p_af
        self.max_tries = max_tries
        self.seed = seed
        self.epoch = 0
        self._init_cache(cache_size)

    def __len__(self):
        return len(self.patient_dirs)

    def _init_cache(self, cache_size):
        @lru_cache(maxsize=cache_size)
        def _load(rec_str):
            sig = load_signal(rec_str)
            ann = wfdb.rdann(rec_str, extension="atr")
            ivs = build_rhythm_intervals_from_ann(ann, len(sig))
            return sig, ann, ivs
        self._load = _load

    def _sample_window(self, sig, ann, ivs, rng, target_af):
        sig_len = len(sig)
        if sig_len <= WINDOW_SAMPLES:
            return None

        for _ in range(self.max_tries):
            w_start = int(rng.integers(0, sig_len - WINDOW_SAMPLES))
            w_end = w_start + WINDOW_SAMPLES
            chunk = sig[w_start:w_end]

            if np.isnan(chunk).any() or np.std(chunk) < 1e-4:
                continue

            lab = window_purity_label(ivs, w_start, w_end, PURITY_THRESH)
            if lab is None:
                continue

            is_af = lab in ("AFIB", "AFL")
            if target_af is not None and is_af != target_af:
                continue

            x = (chunk - chunk.mean()) / (chunk.std() + 1e-6)

            beats, rr = extract_beats_and_rr_from_ann(
                x, ann, w_start, w_end, BEAT_LENGTH
            )
            if beats is None:
                continue

            return x, beats, rr, LABEL2ID[lab]

        return None

    def __getitem__(self, idx):
        rng = np.random.default_rng(
            self.seed + idx * 1000003 + self.epoch * 999983
        )
        target_af = (rng.random() < self.p_af) if self.p_af is not None else None

        recs = self.patient_records[idx]
        if not recs:
            return self.__getitem__((idx + 1) % len(self))

        for _ in range(10):
            rec = recs[int(rng.integers(0, len(recs)))]
            sig, ann, ivs = self._load(str(rec))
            result = self._sample_window(sig, ann, ivs, rng, target_af)
            if result is not None:
                x, beats, rr, y = result
                y = torch.tensor(y, dtype=torch.long)
                if self.mode == "fixed":
                    return torch.from_numpy(x.astype(np.float32)), y
                if self.mode == "beat":
                    return torch.from_numpy(beats), y
                return torch.from_numpy(beats), torch.from_numpy(rr), y

        return self.__getitem__((idx + 1) % len(self))


def fixed_collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def beat_collate(batch):
    beats_list, ys = zip(*batch)
    B = len(beats_list)
    max_n = min(max(b.shape[0] for b in beats_list), MAX_BEATS)

    padded = torch.zeros(B, max_n, 1, BEAT_LENGTH)
    mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, beats in enumerate(beats_list):
        n = min(beats.shape[0], MAX_BEATS)
        padded[i, :n] = beats[:n]
        mask[i, :n] = False

    return padded, mask, torch.stack(ys)


def beat_hr_collate(batch):
    beats_list, rr_list, ys = zip(*batch)
    B = len(beats_list)
    max_n = min(max(b.shape[0] for b in beats_list), MAX_BEATS)

    padded = torch.zeros(B, max_n, 1, BEAT_LENGTH)
    padded_rr = torch.zeros(B, max_n)
    mask = torch.ones(B, max_n, dtype=torch.bool)

    for i, (beats, rr) in enumerate(zip(beats_list, rr_list)):
        n = min(beats.shape[0], MAX_BEATS)
        padded[i, :n] = beats[:n]
        padded_rr[i, :n] = rr[:n]
        mask[i, :n] = False

    return padded, padded_rr, mask, torch.stack(ys)
