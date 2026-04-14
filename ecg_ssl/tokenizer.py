"""
Beat-Synchronous Tokenizers for ECG Transformers.

This module implements tokenizers that segment ECG signals at heartbeat
boundaries (R-peaks) rather than at arbitrary fixed-time windows.

Tokenizer 1: ResampleCNNTokenizer
    - Detect R-peaks on Lead II
    - Slice all 12 leads between consecutive R-peaks
    - Resample each beat to a fixed number of samples
    - Encode each beat with a 1D CNN to produce a token embedding

Tokenizer 2: AdaptivePoolingCNNTokenizer
    - Detect R-peaks on Lead II
    - Slice all 12 leads between consecutive R-peaks
    - NO resampling — feed raw variable-length beats into CNN
    - Use AdaptiveAvgPool1d to collapse to fixed-size output

Tokenizer 3: ResampleCNNWithHRTokenizer
    - Same as Tokenizer 1 (resample + CNN)
    - PLUS: encode the R-R interval duration as a heart rate feature
    - Add the heart rate feature to the beat embedding
"""

import numpy as np
import torch
import torch.nn as nn
import neurokit2 as nk
from scipy.signal import resample


# ---------------------------------------------------------------
# R-peak detection and beat segmentation (shared across all tokenizers)
# ---------------------------------------------------------------

def detect_r_peaks(signal_lead_ii, sampling_rate=500):
    """
    Detect R-peak locations in a single-lead ECG signal.

    Args:
        signal_lead_ii: 1D numpy array of voltage values from Lead II
        sampling_rate: sampling rate in Hz (default 500 for our datasets)

    Returns:
        r_peaks: 1D numpy array of sample indices where R-peaks were detected
    """
    try:
        _, info = nk.ecg_peaks(signal_lead_ii, sampling_rate=sampling_rate)
        r_peaks = info["ECG_R_Peaks"]
        return np.array(r_peaks)
    except Exception:
        # If detection fails (noisy signal, flat lead, etc.), return empty
        return np.array([])


def segment_beats(signal_12lead, r_peaks):
    """
    Slice a 12-lead ECG signal into individual beat segments using R-peak indices.

    Args:
        signal_12lead: numpy array of shape (12, T) — the full 12-lead signal
        r_peaks: 1D array of R-peak sample indices

    Returns:
        beats: list of numpy arrays, each of shape (12, beat_length)
               where beat_length varies per beat
    """
    beats = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        beat = signal_12lead[:, start:end]  # (12, beat_length)
        beats.append(beat)
    return beats


def resample_beat(beat, target_length=256):
    """
    Resample a single beat segment to a fixed number of samples.

    Args:
        beat: numpy array of shape (12, variable_length)
        target_length: number of samples to resample to

    Returns:
        resampled: numpy array of shape (12, target_length)
    """
    num_leads = beat.shape[0]
    resampled = np.zeros((num_leads, target_length), dtype=np.float32)
    for lead_idx in range(num_leads):
        resampled[lead_idx] = resample(beat[lead_idx], target_length)
    return resampled


# ---------------------------------------------------------------
# Extraction pipelines
# ---------------------------------------------------------------

def extract_beat_tokens(signal_12lead_np, target_length=256, sampling_rate=500):
    """
    Full pipeline for Tokenizer 1: take a raw 12-lead ECG and return
    resampled beat segments.

    Args:
        signal_12lead_np: numpy array of shape (12, 5000)
        target_length: fixed length to resample each beat to
        sampling_rate: sampling rate in Hz

    Returns:
        beat_array: numpy array of shape (num_beats, 12, target_length)
                    or None if R-peak detection fails
    """
    lead_ii = signal_12lead_np[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None

    beats = segment_beats(signal_12lead_np, r_peaks)

    if len(beats) == 0:
        return None

    resampled_beats = []
    for beat in beats:
        resampled = resample_beat(beat, target_length=target_length)
        resampled_beats.append(resampled)

    beat_array = np.stack(resampled_beats, axis=0)
    return beat_array


def extract_beat_tokens_with_rr(signal_12lead_np, target_length=256, sampling_rate=500):
    """
    Full pipeline for Tokenizer 3: take a raw 12-lead ECG and return
    resampled beat segments PLUS R-R interval durations.

    The R-R interval for each beat is the number of samples between
    the two R-peaks that define it, normalized by the sampling rate
    to give duration in seconds.

    Args:
        signal_12lead_np: numpy array of shape (12, 5000)
        target_length: fixed length to resample each beat to
        sampling_rate: sampling rate in Hz

    Returns:
        beat_array: numpy array of shape (num_beats, 12, target_length)
        rr_intervals: numpy array of shape (num_beats,) — R-R interval
                      in seconds for each beat
        Returns None, None if R-peak detection fails
    """
    lead_ii = signal_12lead_np[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None, None

    beats = segment_beats(signal_12lead_np, r_peaks)

    if len(beats) == 0:
        return None, None

    resampled_beats = []
    rr_intervals = []

    for i in range(len(beats)):
        resampled = resample_beat(beats[i], target_length=target_length)
        resampled_beats.append(resampled)

        # R-R interval = distance between consecutive R-peaks in seconds
        rr_samples = r_peaks[i + 1] - r_peaks[i]
        rr_seconds = rr_samples / sampling_rate
        rr_intervals.append(rr_seconds)

    beat_array = np.stack(resampled_beats, axis=0)
    rr_intervals = np.array(rr_intervals, dtype=np.float32)

    return beat_array, rr_intervals


def extract_beat_tokens_raw(signal_12lead_np, sampling_rate=500):
    """
    Full pipeline for Tokenizer 2: take a raw 12-lead ECG and return
    variable-length beat segments padded to the max beat length.

    No resampling — beats are kept at their original length and
    zero-padded to the length of the longest beat in this recording.

    Args:
        signal_12lead_np: numpy array of shape (12, 5000)
        sampling_rate: sampling rate in Hz

    Returns:
        beat_array: numpy array of shape (num_beats, 12, max_beat_length)
                    where shorter beats are zero-padded
        beat_lengths: numpy array of shape (num_beats,) with the actual
                      length of each beat before padding
        Returns None, None if R-peak detection fails
    """
    lead_ii = signal_12lead_np[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None, None

    beats = segment_beats(signal_12lead_np, r_peaks)

    if len(beats) == 0:
        return None, None

    beat_lengths = np.array([beat.shape[1] for beat in beats])
    max_beat_len = beat_lengths.max()
    num_leads = signal_12lead_np.shape[0]

    beat_array = np.zeros((len(beats), num_leads, max_beat_len), dtype=np.float32)
    for i, beat in enumerate(beats):
        beat_len = beat.shape[1]
        beat_array[i, :, :beat_len] = beat

    return beat_array, beat_lengths


# ---------------------------------------------------------------
# Tokenizer 1: Resample + CNN
# ---------------------------------------------------------------

class ResampleCNNTokenizer(nn.Module):
    """
    Beat-synchronous tokenizer that:
    1. Detects R-peaks and segments ECG into individual beats
    2. Resamples each beat to a fixed length
    3. Encodes each beat with a 1D CNN to produce a token embedding
    """

    def __init__(self, in_channels=12, d_model=256, beat_length=256):
        super().__init__()
        self.beat_length = beat_length
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, beat_tokens):
        """
        Args:
            beat_tokens: (B, N, 12, beat_length)
        Returns:
            embeddings: (B, N, d_model)
        """
        B, N, C, T = beat_tokens.shape
        x = beat_tokens.view(B * N, C, T)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        embeddings = x.view(B, N, self.d_model)
        return embeddings


# ---------------------------------------------------------------
# Tokenizer 2: Adaptive Pooling CNN (no resampling)
# ---------------------------------------------------------------

class AdaptivePoolingCNNTokenizer(nn.Module):
    """
    Beat-synchronous tokenizer that:
    1. Detects R-peaks and segments ECG into individual beats
    2. Does NOT resample — beats stay at original variable length
    3. Encodes each beat with Conv1d layers
    4. Uses AdaptiveAvgPool1d(1) to produce fixed-size embedding
    """

    def __init__(self, in_channels=12, d_model=256):
        super().__init__()
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, beat_tokens):
        """
        Args:
            beat_tokens: (B, N, 12, T_padded)
        Returns:
            embeddings: (B, N, d_model)
        """
        B, N, C, T = beat_tokens.shape
        x = beat_tokens.view(B * N, C, T)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        embeddings = x.view(B, N, self.d_model)
        return embeddings


# ---------------------------------------------------------------
# Tokenizer 3: Resample + CNN + Heart Rate Feature
# ---------------------------------------------------------------

class ResampleCNNWithHRTokenizer(nn.Module):
    """
    Beat-synchronous tokenizer that:
    1. Detects R-peaks and segments ECG into individual beats
    2. Resamples each beat to a fixed length
    3. Encodes each beat with a 1D CNN (same as Tokenizer 1)
    4. ADDITIONALLY encodes the R-R interval (heart rate) as a feature
    5. Adds the heart rate feature to the beat embedding

    This preserves heart rate information that is lost during resampling.
    A short R-R interval means fast heart rate (tachycardia),
    a long R-R interval means slow heart rate (bradycardia).
    Both are diagnostically meaningful.
    """

    def __init__(self, in_channels=12, d_model=256, beat_length=256):
        super().__init__()
        self.beat_length = beat_length
        self.d_model = d_model

        # Same CNN encoder as Tokenizer 1
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Heart rate encoder: takes a single scalar (R-R interval in seconds)
        # and projects it to d_model dimensions so it can be added to the
        # beat embedding
        self.hr_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

    def forward(self, beat_tokens, rr_intervals):
        """
        Args:
            beat_tokens: (B, N, 12, beat_length) — resampled beats
            rr_intervals: (B, N) — R-R interval in seconds for each beat

        Returns:
            embeddings: (B, N, d_model) — beat embedding + heart rate feature
        """
        B, N, C, T = beat_tokens.shape

        # Encode beats with CNN (same as Tokenizer 1)
        x = beat_tokens.view(B * N, C, T)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        beat_embeddings = x.view(B, N, self.d_model)  # (B, N, d_model)

        # Encode R-R intervals
        # rr_intervals: (B, N) -> (B, N, 1) for the linear layer
        rr_input = rr_intervals.unsqueeze(-1).float()  # (B, N, 1)
        hr_features = self.hr_encoder(rr_input)         # (B, N, d_model)

        # Add heart rate features to beat embeddings
        embeddings = beat_embeddings + hr_features

        return embeddings
