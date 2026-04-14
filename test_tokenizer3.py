"""
Test script for Tokenizer 3: Resample + CNN + Heart Rate Feature.
Tests standalone tokenizer, dataset, collation, model, loss, backward, and optimizer step.

Run from repo root: python test_tokenizer3.py
"""

import numpy as np
import torch
import wfdb
from pathlib import Path

from ecg_ssl.tokenizer import (
    detect_r_peaks,
    extract_beat_tokens_with_rr,
    ResampleCNNWithHRTokenizer,
    ResampleCNNTokenizer,
)
from ecg_ssl.loss import masked_patch_mse_loss

# ---------------------------------------------------------------
# Test 1: Extract beats with R-R intervals
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: Extract beats with R-R intervals")
print("=" * 60)

rec = wfdb.rdrecord("data/ptb-xl/records500/00000/00001_hr")
signal = rec.p_signal.astype(np.float32).T  # (12, 5000)
print(f"Loaded ECG shape: {signal.shape}")

beat_array, rr_intervals = extract_beat_tokens_with_rr(signal, target_length=256, sampling_rate=500)

if beat_array is not None:
    print(f"Beat array shape: {beat_array.shape}")
    print(f"R-R intervals (seconds): {rr_intervals}")
    print(f"  Min: {rr_intervals.min():.3f}s, Max: {rr_intervals.max():.3f}s, "
          f"Mean: {rr_intervals.mean():.3f}s")

    # Convert to heart rate (BPM) for sanity check
    hr_bpm = 60.0 / rr_intervals
    print(f"Heart rates (BPM): {np.round(hr_bpm, 1)}")
    print(f"  Min: {hr_bpm.min():.1f}, Max: {hr_bpm.max():.1f}, Mean: {hr_bpm.mean():.1f}")

    # Sanity check: normal resting HR is 50-100 BPM
    if hr_bpm.mean() > 40 and hr_bpm.mean() < 150:
        print("Heart rate range looks reasonable.")
    else:
        print("WARNING: Heart rate range seems unusual!")
else:
    print("ERROR: extraction failed")
    exit()

# ---------------------------------------------------------------
# Test 2: Tokenizer 3 forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: ResampleCNNWithHRTokenizer forward pass")
print("=" * 60)

tokenizer = ResampleCNNWithHRTokenizer(in_channels=12, d_model=256, beat_length=256)

# Simulate batch of 2
beat_tensor = torch.from_numpy(beat_array).unsqueeze(0).repeat(2, 1, 1, 1)  # (2, N, 12, 256)
rr_tensor = torch.from_numpy(rr_intervals).unsqueeze(0).repeat(2, 1)         # (2, N)

print(f"Input beats shape: {beat_tensor.shape}")
print(f"Input rr_intervals shape: {rr_tensor.shape}")

with torch.no_grad():
    embeddings = tokenizer(beat_tensor, rr_tensor)
print(f"Output embeddings shape: {embeddings.shape}")

if embeddings.shape == (2, beat_array.shape[0], 256):
    print("PASS: Output shape is correct!")
else:
    print("FAIL: Output shape mismatch!")

# ---------------------------------------------------------------
# Test 3: Compare Tokenizer 1 vs Tokenizer 3
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: Compare Tokenizer 1 (no HR) vs Tokenizer 3 (with HR)")
print("=" * 60)

tok1 = ResampleCNNTokenizer(in_channels=12, d_model=256, beat_length=256)
tok3 = ResampleCNNWithHRTokenizer(in_channels=12, d_model=256, beat_length=256)

# Copy Tokenizer 1's CNN weights into Tokenizer 3 so the only difference is HR
tok3.encoder.load_state_dict(tok1.encoder.state_dict())
tok3.pool.load_state_dict(tok1.pool.state_dict())

single_beat = beat_tensor[:1]  # (1, N, 12, 256)
single_rr = rr_tensor[:1]     # (1, N)

with torch.no_grad():
    out1 = tok1(single_beat)
    out3 = tok3(single_beat, single_rr)

diff = (out1 - out3).abs().mean().item()
print(f"Mean absolute difference: {diff:.4f}")
print("(Should be non-zero — Tokenizer 3 adds HR features that Tokenizer 1 doesn't have)")

if diff > 0.0:
    print("PASS: HR encoding is changing the embeddings!")
else:
    print("FAIL: Embeddings are identical — HR encoding isn't working")

# ---------------------------------------------------------------
# Test 4: Dataset simulation with R-R intervals
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Batch collation with R-R intervals")
print("=" * 60)

# Load multiple records and simulate batching
data_dir = Path("data/ptb-xl")
hea_files = sorted(data_dir.rglob("*.hea"))
record_paths = [str(p.with_suffix("")) for p in hea_files[:5]]

items = []
for rp in record_paths:
    rec = wfdb.rdrecord(rp)
    sig = rec.p_signal.astype(np.float32).T
    ba, rr = extract_beat_tokens_with_rr(sig, target_length=256, sampling_rate=500)
    if ba is not None:
        items.append({"beats": ba, "rr_intervals": rr, "num_beats": ba.shape[0]})
        print(f"  {rp}: {ba.shape[0]} beats, rr_mean={rr.mean():.3f}s")

# Pad to max beats
max_beats = max(item["num_beats"] for item in items)
B = len(items)

padded_beats = torch.zeros(B, max_beats, 12, 256)
padded_rr = torch.zeros(B, max_beats)
padding_mask = torch.ones(B, max_beats, dtype=torch.bool)

for i, item in enumerate(items):
    n = item["num_beats"]
    padded_beats[i, :n] = torch.from_numpy(item["beats"])
    padded_rr[i, :n] = torch.from_numpy(item["rr_intervals"])
    padding_mask[i, :n] = False

print(f"\nPadded batch:")
print(f"  beats: {padded_beats.shape}")
print(f"  rr_intervals: {padded_rr.shape}")
print(f"  padding_mask: {padding_mask.shape}")

# ---------------------------------------------------------------
# Test 5: Full model forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: ECGMaskedSSLBeatHR full forward pass")
print("=" * 60)

from ecg_ssl.model_beat_hr import ECGMaskedSSLBeatHR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ECGMaskedSSLBeatHR(
    in_channels=12,
    d_model=256,
    beat_length=256,
    num_heads=8,
    num_layers=4,
    mlp_ratio=4,
    dropout=0.1,
).to(device)

beats_gpu = padded_beats.to(device)
rr_gpu = padded_rr.to(device)
mask_gpu = padding_mask.to(device)

with torch.no_grad():
    output = model(beats_gpu, rr_gpu, padding_mask=mask_gpu, mask_ratio=0.20, span_len=1)

beat_dim = 12 * 256
print(f"pred_patches: {output['pred_patches'].shape}")
print(f"target_patches: {output['target_patches'].shape}")
print(f"mask: {output['mask'].shape}")
print(f"encoded: {output['encoded'].shape}")
print(f"pooled: {output['pooled'].shape}")

assert output["pred_patches"].shape == (B, max_beats, beat_dim)
assert output["target_patches"].shape == (B, max_beats, beat_dim)
print("All shape checks PASSED!")

# ---------------------------------------------------------------
# Test 6: Loss + backward + optimizer step
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 6: Loss, backward, optimizer step")
print("=" * 60)

model.train()
output = model(beats_gpu, rr_gpu, padding_mask=mask_gpu, mask_ratio=0.20, span_len=1)
loss = masked_patch_mse_loss(output["pred_patches"], output["target_patches"], output["mask"])
print(f"Loss: {loss.item():.4f}")
print(f"Loss is finite: {torch.isfinite(loss).item()}")

loss.backward()

total_params = sum(1 for _ in model.parameters())
grads = sum(1 for p in model.parameters() if p.grad is not None)
print(f"Parameters with gradients: {grads}/{total_params}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
optimizer.zero_grad()

output = model(beats_gpu, rr_gpu, padding_mask=mask_gpu, mask_ratio=0.20, span_len=1)
loss = masked_patch_mse_loss(output["pred_patches"], output["target_patches"], output["mask"])
print(f"Loss before step: {loss.item():.4f}")
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print("Optimizer step completed!")

with torch.no_grad():
    output2 = model(beats_gpu, rr_gpu, padding_mask=mask_gpu, mask_ratio=0.20, span_len=1)
    loss2 = masked_patch_mse_loss(output2["pred_patches"], output2["target_patches"], output2["mask"])
print(f"Loss after step: {loss2.item():.4f}")

print("\n" + "=" * 60)
print("ALL TOKENIZER 3 TESTS COMPLETE")
print("=" * 60)
