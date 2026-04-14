"""
Integration test for the beat-synchronous pipeline.
Tests: dataset loading -> collation -> model forward pass -> loss computation -> backward pass

Run from repo root: python test_integration.py
"""

import numpy as np
import torch
import wfdb
from pathlib import Path

from ecg_ssl.tokenizer import extract_beat_tokens, ResampleCNNTokenizer
from ecg_ssl.dataset import BeatECGDataset, beat_collate_fn
from ecg_ssl.model_beat import ECGMaskedSSLBeat
from ecg_ssl.loss import masked_patch_mse_loss


# ---------------------------------------------------------------
# Test 1: BeatECGDataset loads and returns correct format
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: BeatECGDataset loading")
print("=" * 60)

# Find all available records
data_dir = Path("data/ptb-xl")
hea_files = sorted(data_dir.rglob("*.hea"))
record_paths = [str(p.with_suffix("")) for p in hea_files]
print(f"Found {len(record_paths)} records")

dataset = BeatECGDataset(
    record_paths[:10],  # just use first 10 for testing
    beat_length=256,
    sampling_rate=500,
    min_beats=3,
    max_retries=10,
)

item = dataset[0]
print(f"beats shape: {item['beats'].shape}")
print(f"num_beats: {item['num_beats']}")
print(f"record_path: {item['record_path']}")

# ---------------------------------------------------------------
# Test 2: Collate function handles variable-length batching
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: Collate function (variable-length batching)")
print("=" * 60)

# Load a few items
batch_items = []
for i in range(min(4, len(record_paths))):
    try:
        item = dataset[i]
        batch_items.append(item)
        print(f"  Record {i}: {item['num_beats']} beats")
    except RuntimeError:
        print(f"  Record {i}: failed to load, skipping")

if len(batch_items) < 2:
    print("ERROR: Need at least 2 records for batch test")
else:
    batch = beat_collate_fn(batch_items)
    print(f"\nCollated batch:")
    print(f"  beats shape: {batch['beats'].shape}")
    print(f"  padding_mask shape: {batch['padding_mask'].shape}")
    print(f"  num_beats: {batch['num_beats']}")
    print(f"  padding_mask example (first 2 rows):")
    for i in range(min(2, len(batch_items))):
        print(f"    Record {i}: {batch['padding_mask'][i].tolist()}")

# ---------------------------------------------------------------
# Test 3: Model forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: ECGMaskedSSLBeat forward pass")
print("=" * 60)

model = ECGMaskedSSLBeat(
    in_channels=12,
    d_model=256,
    beat_length=256,
    num_heads=8,
    num_layers=4,
    mlp_ratio=4,
    dropout=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

beats = batch["beats"].to(device)
padding_mask = batch["padding_mask"].to(device)

print(f"Input beats shape: {beats.shape}")
print(f"Input padding_mask shape: {padding_mask.shape}")

with torch.no_grad():
    output = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)

print(f"\nOutput shapes:")
print(f"  pred_patches: {output['pred_patches'].shape}")
print(f"  target_patches: {output['target_patches'].shape}")
print(f"  mask: {output['mask'].shape}")
print(f"  encoded: {output['encoded'].shape}")
print(f"  pooled: {output['pooled'].shape}")

# Verify shapes are consistent
B, N = batch["beats"].shape[:2]
beat_dim = 12 * 256
assert output["pred_patches"].shape == (B, N, beat_dim), "pred_patches shape mismatch!"
assert output["target_patches"].shape == (B, N, beat_dim), "target_patches shape mismatch!"
assert output["mask"].shape == (B, N), "mask shape mismatch!"
assert output["encoded"].shape == (B, N, 256), "encoded shape mismatch!"
assert output["pooled"].shape == (B, 256), "pooled shape mismatch!"
print("\nAll shape checks PASSED!")

# ---------------------------------------------------------------
# Test 4: Loss computation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 4: Loss computation")
print("=" * 60)

loss = masked_patch_mse_loss(
    output["pred_patches"],
    output["target_patches"],
    output["mask"],
)
print(f"Loss value: {loss.item():.4f}")
print(f"Loss is finite: {torch.isfinite(loss).item()}")
print(f"Masked tokens: {output['mask'].sum().item()} / {output['mask'].numel()}")

# ---------------------------------------------------------------
# Test 5: Backward pass (can we compute gradients?)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: Backward pass")
print("=" * 60)

# Need to re-run forward pass with gradients enabled
model.train()
output = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)
loss = masked_patch_mse_loss(
    output["pred_patches"],
    output["target_patches"],
    output["mask"],
)

loss.backward()

# Check that gradients exist
total_params = 0
params_with_grad = 0
for name, param in model.named_parameters():
    total_params += 1
    if param.grad is not None:
        params_with_grad += 1

print(f"Parameters with gradients: {params_with_grad}/{total_params}")
if params_with_grad == total_params:
    print("PASS: All parameters received gradients!")
else:
    print("WARNING: Some parameters did not receive gradients")

# ---------------------------------------------------------------
# Test 6: Single optimizer step
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 6: Single optimizer step")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
optimizer.zero_grad()

output = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)
loss = masked_patch_mse_loss(
    output["pred_patches"],
    output["target_patches"],
    output["mask"],
)

print(f"Loss before step: {loss.item():.4f}")
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print("Optimizer step completed successfully!")

# Run forward again to check loss changed
optimizer.zero_grad()
with torch.no_grad():
    output2 = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)
    loss2 = masked_patch_mse_loss(
        output2["pred_patches"],
        output2["target_patches"],
        output2["mask"],
    )
print(f"Loss after step: {loss2.item():.4f}")

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS COMPLETE")
print("=" * 60)
