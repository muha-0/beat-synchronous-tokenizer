"""
Beat-Synchronous Tokenizers for ECG Transformers.

This module implements tokenizers that segment ECG signals at heartbeat
boundaries (R-peaks) rather than at arbitrary fixed-time windows.

Tokenizer 1: ResampleCNNTokenizer
    - Detect R-peaks on Lead II
    - Slice all 12 leads between consecutive R-peaks
    - Resample each beat to a fixed number of samples
    - Encode each beat with a 1D CNN to produce a token embedding
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


def extract_beat_tokens(signal_12lead_np, target_length=256, sampling_rate=500):
    """
    Full pipeline: take a raw 12-lead ECG and return resampled beat segments.

    Args:
        signal_12lead_np: numpy array of shape (12, 5000)
        target_length: fixed length to resample each beat to
        sampling_rate: sampling rate in Hz

    Returns:
        beat_array: numpy array of shape (num_beats, 12, target_length)
                    or None if R-peak detection fails
    """
    # Step 1: Detect R-peaks on Lead II (index 1)
    lead_ii = signal_12lead_np[1]  # Lead II is index 1
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    # Need at least 2 R-peaks to get 1 beat
    if len(r_peaks) < 2:
        return None

    # Step 2: Segment between R-peaks (applies to all 12 leads)
    beats = segment_beats(signal_12lead_np, r_peaks)

    if len(beats) == 0:
        return None

    # Step 3: Resample each beat to fixed length
    resampled_beats = []
    for beat in beats:
        resampled = resample_beat(beat, target_length=target_length)
        resampled_beats.append(resampled)

    # Stack into (num_beats, 12, target_length)
    beat_array = np.stack(resampled_beats, axis=0)
    return beat_array


# ---------------------------------------------------------------
# Tokenizer 1: Resample + CNN
# ---------------------------------------------------------------

class ResampleCNNTokenizer(nn.Module):
    """
    Beat-synchronous tokenizer that:
    1. Detects R-peaks and segments ECG into individual beats
    2. Resamples each beat to a fixed length
    3. Encodes each beat with a 1D CNN to produce a token embedding

    The CNN architecture mirrors the existing FixedCNNTokenizer but operates
    on individual resampled beats rather than fixed-width patches.
    """

    def __init__(self, in_channels=12, d_model=256, beat_length=256):
        super().__init__()
        self.beat_length = beat_length
        self.d_model = d_model

        # CNN encoder: takes a (12, beat_length) beat and produces a d_model-dim embedding
        # We use multiple conv layers to progressively compress the beat
        self.encoder = nn.Sequential(
            # Layer 1: (12, 256) -> (64, 128)
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),

            # Layer 2: (64, 128) -> (128, 64)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),

            # Layer 3: (128, 64) -> (256, 32)
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        # Adaptive pooling to collapse whatever spatial dim remains into 1
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, beat_tokens):
        """
        Args:
            beat_tokens: tensor of shape (B, N, 12, beat_length)
                         B = batch size, N = number of beats per ECG
                         Already segmented, resampled, and batched.

        Returns:
            embeddings: tensor of shape (B, N, d_model)
        """
        B, N, C, T = beat_tokens.shape

        # Reshape to process all beats at once: (B*N, 12, beat_length)
        x = beat_tokens.view(B * N, C, T)

        # Run through CNN encoder
        x = self.encoder(x)    # (B*N, d_model, reduced_length)
        x = self.pool(x)       # (B*N, d_model, 1)
        x = x.squeeze(-1)      # (B*N, d_model)

        # Reshape back to (B, N, d_model)
        embeddings = x.view(B, N, self.d_model)
        return embeddings
