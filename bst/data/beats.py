import warnings

import neurokit2 as nk
import numpy as np
from scipy.signal import resample


def detect_r_peaks(lead_ii, sampling_rate=500):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ecg_peaks(lead_ii, sampling_rate=sampling_rate)
        return np.array(info["ECG_R_Peaks"])
    except Exception:
        return np.array([])


def extract_and_resample_beats(x, beat_length=300, sampling_rate=500, min_beats=3):
    """
    x: (12, 5000) float32, already z-scored
    returns: (N, 12, beat_length) float32 or None
    """
    lead_ii = x[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None

    beats = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        beat = x[:, start:end]   # (12, variable_length)

        if beat.shape[1] < 10:    # reject tiny segments
            continue

        resampled = np.zeros((12, beat_length), dtype=np.float32)
        for c in range(12):
            resampled[c] = resample(beat[c], beat_length)
        beats.append(resampled)

    if len(beats) < min_beats:
        return None

    return np.stack(beats, axis=0)


def extract_raw_beats(x, sampling_rate=500, min_beats=3):
    """
    x: (12, 5000) float32, already z-scored
    returns:
        beat_array: (N, 12, max_beat_len_in_record) zero-padded
        lengths:    (N,) actual length of each beat
    or None, None if detection fails
    """
    lead_ii = x[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None, None

    beats = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        beat = x[:, start:end]
        if beat.shape[1] < 10:
            continue
        beats.append(beat)

    if len(beats) < min_beats:
        return None, None

    lengths = np.array([b.shape[1] for b in beats], dtype=np.int32)
    max_beat_len = lengths.max()

    beat_array = np.zeros((len(beats), 12, max_beat_len), dtype=np.float32)
    for i, beat in enumerate(beats):
        beat_array[i, :, :beat.shape[1]] = beat

    return beat_array, lengths


def extract_raw_beats_padded(x, max_beat_len=600, sampling_rate=500, min_beats=3):
    """
    x: (12, 5000) z-scored
    returns: (N, 12, max_beat_len) zero-padded, truncated to max_beat_len, or None
    """
    lead_ii = x[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None

    beats = []
    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        beat = x[:, start:end]
        if beat.shape[1] < 10:
            continue
        beats.append(beat)

    if len(beats) < min_beats:
        return None

    beat_array = np.zeros((len(beats), 12, max_beat_len), dtype=np.float32)
    for i, beat in enumerate(beats):
        t = min(beat.shape[1], max_beat_len)
        beat_array[i, :, :t] = beat[:, :t]

    return beat_array


def extract_beats_with_rr(x, beat_length=300, sampling_rate=500, min_beats=3):
    """
    x: (12, 5000) float32, already z-scored
    returns:
        beat_array: (N, 12, beat_length) resampled beats
        rr_seconds: (N,) R-R interval in seconds for each beat
    or None, None if detection fails
    """
    lead_ii = x[1]
    r_peaks = detect_r_peaks(lead_ii, sampling_rate=sampling_rate)

    if len(r_peaks) < 2:
        return None, None

    beats = []
    rr_seconds = []

    for i in range(len(r_peaks) - 1):
        start = r_peaks[i]
        end = r_peaks[i + 1]
        beat = x[:, start:end]

        if beat.shape[1] < 10:
            continue

        # R-R interval in seconds — before resampling
        rr = (end - start) / sampling_rate
        rr_seconds.append(rr)

        resampled = np.zeros((12, beat_length), dtype=np.float32)
        for c in range(12):
            resampled[c] = resample(beat[c], beat_length)
        beats.append(resampled)

    if len(beats) < min_beats:
        return None, None

    return np.stack(beats, axis=0), np.array(rr_seconds, dtype=np.float32)
