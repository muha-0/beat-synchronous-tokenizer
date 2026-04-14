"""
Quick test script for the beat-synchronous tokenizer.
Run from the repo root: python test_tokenizer.py
"""

import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt

from ecg_ssl.tokenizer import (
    detect_r_peaks,
    segment_beats,
    resample_beat,
    extract_beat_tokens,
    ResampleCNNTokenizer,
)

# ---------------------------------------------------------------
# Test 1: Load a real ECG and detect R-peaks
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: R-peak detection on a real ECG")
print("=" * 60)

rec = wfdb.rdrecord("data/ptb-xl/records500/00000/00001_hr")
signal = rec.p_signal.astype(np.float32).T  # (12, 5000)
print(f"Loaded ECG shape: {signal.shape}")

lead_ii = signal[1]  # Lead II
r_peaks = detect_r_peaks(lead_ii, sampling_rate=500)
print(f"R-peaks found: {len(r_peaks)}")
print(f"R-peak indices: {r_peaks}")

# Sanity check: for a 10-second recording, expect roughly 7-15 beats
if len(r_peaks) < 2:
    print("WARNING: Too few R-peaks detected!")
elif len(r_peaks) > 20:
    print("WARNING: Unusually many R-peaks detected!")
else:
    print("R-peak count looks reasonable.")

# ---------------------------------------------------------------
# Test 2: Segment into beats
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Beat segmentation")
print("=" * 60)

beats = segment_beats(signal, r_peaks)
print(f"Number of beats: {len(beats)}")
for i, beat in enumerate(beats):
    print(f"  Beat {i}: shape {beat.shape} ({beat.shape[1]} samples = {beat.shape[1]/500:.3f}s)")

# ---------------------------------------------------------------
# Test 3: Resample beats to fixed length
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: Resample beats to fixed length")
print("=" * 60)

target_length = 256
for i, beat in enumerate(beats[:3]):  # just show first 3
    resampled = resample_beat(beat, target_length=target_length)
    print(f"  Beat {i}: {beat.shape} -> {resampled.shape}")

# ---------------------------------------------------------------
# Test 4: Full pipeline (extract_beat_tokens)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Full extraction pipeline")
print("=" * 60)

beat_array = extract_beat_tokens(signal, target_length=256, sampling_rate=500)
if beat_array is not None:
    print(f"Beat array shape: {beat_array.shape}")
    print(f"  (num_beats={beat_array.shape[0]}, leads={beat_array.shape[1]}, samples={beat_array.shape[2]})")
else:
    print("ERROR: extract_beat_tokens returned None")

# ---------------------------------------------------------------
# Test 5: CNN tokenizer forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: ResampleCNNTokenizer forward pass")
print("=" * 60)

if beat_array is not None:
    tokenizer = ResampleCNNTokenizer(in_channels=12, d_model=256, beat_length=256)

    # Simulate a batch of 2 by duplicating
    beat_tensor = torch.from_numpy(beat_array).unsqueeze(0)  # (1, N, 12, 256)
    beat_tensor = beat_tensor.repeat(2, 1, 1, 1)              # (2, N, 12, 256)
    print(f"Input tensor shape: {beat_tensor.shape}")

    with torch.no_grad():
        embeddings = tokenizer(beat_tensor)
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"  Expected: (2, {beat_array.shape[0]}, 256)")

    if embeddings.shape == (2, beat_array.shape[0], 256):
        print("PASS: Output shape is correct!")
    else:
        print("FAIL: Output shape mismatch!")

# ---------------------------------------------------------------
# Test 6: Visualize R-peaks on the signal
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 6: Saving visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot Lead II with R-peaks marked
time = np.arange(len(lead_ii)) / 500  # convert to seconds
axes[0].plot(time, lead_ii, color="steelblue", linewidth=0.8)
axes[0].scatter(r_peaks / 500, lead_ii[r_peaks], color="red", marker="v", s=100, zorder=5, label="R-peaks")
axes[0].set_title("Lead II with Detected R-peaks")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Voltage")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot individual resampled beats overlaid
if beat_array is not None:
    for i in range(beat_array.shape[0]):
        axes[1].plot(beat_array[i, 1, :], alpha=0.6, label=f"Beat {i}")  # Lead II of each beat
    axes[1].set_title(f"Resampled Beats (Lead II, {target_length} samples each)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Voltage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tokenizer_test_output.png", dpi=150)
print("Saved visualization to tokenizer_test_output.png")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
