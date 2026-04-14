"""
Test script for Tokenizer 2: Adaptive Pooling CNN (no resampling).
Run from the repo root: python test_tokenizer2.py
"""

import numpy as np
import torch
import wfdb
import matplotlib.pyplot as plt
from pathlib import Path

from ecg_ssl.tokenizer import (
    detect_r_peaks,
    segment_beats,
    extract_beat_tokens_raw,
    AdaptivePoolingCNNTokenizer,
)

# ---------------------------------------------------------------
# Test 1: Extract raw (non-resampled) beats
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: Extract raw beat segments (no resampling)")
print("=" * 60)

rec = wfdb.rdrecord("data/ptb-xl/records500/00000/00001_hr")
signal = rec.p_signal.astype(np.float32).T  # (12, 5000)
print(f"Loaded ECG shape: {signal.shape}")

beat_array, beat_lengths = extract_beat_tokens_raw(signal, sampling_rate=500)

if beat_array is not None:
    print(f"Beat array shape: {beat_array.shape}")
    print(f"  (num_beats={beat_array.shape[0]}, leads={beat_array.shape[1]}, "
          f"max_beat_length={beat_array.shape[2]})")
    print(f"Individual beat lengths: {beat_lengths}")
    print(f"  Min: {beat_lengths.min()}, Max: {beat_lengths.max()}, "
          f"Mean: {beat_lengths.mean():.1f}")
else:
    print("ERROR: extract_beat_tokens_raw returned None")

# ---------------------------------------------------------------
# Test 2: Verify padding is correct
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Verify zero-padding")
print("=" * 60)

if beat_array is not None:
    for i in range(min(3, len(beat_lengths))):
        actual_len = beat_lengths[i]
        max_len = beat_array.shape[2]
        # Check that values after actual_len are zero
        padding_region = beat_array[i, :, actual_len:]
        is_zero = np.allclose(padding_region, 0.0)
        # Check that values before actual_len are NOT all zero
        real_region = beat_array[i, :, :actual_len]
        has_signal = not np.allclose(real_region, 0.0)
        print(f"  Beat {i}: length={actual_len}, padded_to={max_len}, "
              f"padding_is_zero={is_zero}, has_real_signal={has_signal}")

# ---------------------------------------------------------------
# Test 3: AdaptivePoolingCNNTokenizer forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: AdaptivePoolingCNNTokenizer forward pass")
print("=" * 60)

if beat_array is not None:
    tokenizer = AdaptivePoolingCNNTokenizer(in_channels=12, d_model=256)

    # Simulate a batch of 2
    beat_tensor = torch.from_numpy(beat_array).unsqueeze(0)  # (1, N, 12, max_len)
    beat_tensor = beat_tensor.repeat(2, 1, 1, 1)              # (2, N, 12, max_len)
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
# Test 4: Compare Tokenizer 1 vs Tokenizer 2 outputs
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Compare Tokenizer 1 vs Tokenizer 2")
print("=" * 60)

from ecg_ssl.tokenizer import extract_beat_tokens, ResampleCNNTokenizer

# Tokenizer 1: resampled
beat_array_resampled = extract_beat_tokens(signal, target_length=256, sampling_rate=500)
tok1 = ResampleCNNTokenizer(in_channels=12, d_model=256, beat_length=256)

# Tokenizer 2: raw
tok2 = AdaptivePoolingCNNTokenizer(in_channels=12, d_model=256)

if beat_array_resampled is not None and beat_array is not None:
    t1_input = torch.from_numpy(beat_array_resampled).unsqueeze(0)  # (1, N, 12, 256)
    t2_input = torch.from_numpy(beat_array).unsqueeze(0)             # (1, N, 12, max_len)

    with torch.no_grad():
        t1_out = tok1(t1_input)
        t2_out = tok2(t2_input)

    print(f"Tokenizer 1 input shape:  {t1_input.shape}")
    print(f"Tokenizer 1 output shape: {t1_out.shape}")
    print(f"Tokenizer 2 input shape:  {t2_input.shape}")
    print(f"Tokenizer 2 output shape: {t2_out.shape}")

    # Both should produce same number of tokens with same embedding dim
    assert t1_out.shape[1] == t2_out.shape[1], "Different number of beats!"
    assert t1_out.shape[2] == t2_out.shape[2], "Different embedding dim!"
    print("PASS: Both tokenizers produce same-shaped output!")

    # But the actual values should be different since the encoding is different
    diff = (t1_out - t2_out).abs().mean().item()
    print(f"Mean absolute difference between embeddings: {diff:.4f}")
    print("(Expected to be non-zero since they process the signal differently)")

# ---------------------------------------------------------------
# Test 5: Variable-length batch simulation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: Variable-length batch (different max beat lengths)")
print("=" * 60)

# Load two different records that may have different max beat lengths
data_dir = Path("data/ptb-xl")
hea_files = sorted(data_dir.rglob("*.hea"))
record_paths = [str(p.with_suffix("")) for p in hea_files[:5]]

results = []
for rp in record_paths:
    rec = wfdb.rdrecord(rp)
    sig = rec.p_signal.astype(np.float32).T
    ba, bl = extract_beat_tokens_raw(sig, sampling_rate=500)
    if ba is not None:
        results.append((ba, bl))
        print(f"  {rp}: {ba.shape[0]} beats, max_len={ba.shape[2]}, "
              f"lengths={bl.tolist()}")

if len(results) >= 2:
    # These will have different max_beat_lengths — we need to pad to the
    # global max across the batch for the collate function
    global_max_beats = max(r[0].shape[0] for r in results)
    global_max_len = max(r[0].shape[2] for r in results)
    print(f"\nGlobal max beats: {global_max_beats}")
    print(f"Global max beat length: {global_max_len}")

    B = len(results)
    padded = torch.zeros(B, global_max_beats, 12, global_max_len)
    for i, (ba, bl) in enumerate(results):
        n_beats = ba.shape[0]
        max_len = ba.shape[2]
        padded[i, :n_beats, :, :max_len] = torch.from_numpy(ba)

    print(f"Padded batch shape: {padded.shape}")

    tokenizer = AdaptivePoolingCNNTokenizer(in_channels=12, d_model=256)
    with torch.no_grad():
        out = tokenizer(padded)
    print(f"Output shape: {out.shape}")
    print("PASS: Tokenizer 2 handles variable-length batches!")

# ---------------------------------------------------------------
# Test 6: Visualize raw vs resampled beats
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 6: Saving comparison visualization")
print("=" * 60)

if beat_array is not None and beat_array_resampled is not None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot raw beats (different lengths, zero-padded)
    for i in range(min(5, beat_array.shape[0])):
        actual_len = beat_lengths[i]
        axes[0].plot(beat_array[i, 1, :actual_len], alpha=0.7, label=f"Beat {i} ({actual_len} samples)")
    axes[0].set_title("Tokenizer 2: Raw Beats (original length, no resampling)")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Voltage")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot resampled beats (all same length)
    for i in range(min(5, beat_array_resampled.shape[0])):
        axes[1].plot(beat_array_resampled[i, 1, :], alpha=0.7, label=f"Beat {i} (256 samples)")
    axes[1].set_title("Tokenizer 1: Resampled Beats (all 256 samples)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Voltage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tokenizer2_test_output.png", dpi=150)
    print("Saved visualization to tokenizer2_test_output.png")

print("\n" + "=" * 60)
print("ALL TOKENIZER 2 TESTS COMPLETE")
print("=" * 60)
