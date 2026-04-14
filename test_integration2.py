"""
Integration test for Tokenizer 2 (Adaptive Pooling CNN) full pipeline.
Tests: dataset -> collation -> model forward -> loss -> backward -> optimizer step

Run from repo root: python test_integration2.py
"""

import numpy as np
import torch
from pathlib import Path

from ecg_ssl.tokenizer import AdaptivePoolingCNNTokenizer
from ecg_ssl.dataset import RawBeatECGDataset, raw_beat_collate_fn
from ecg_ssl.model_beat_adaptive import ECGMaskedSSLBeatAdaptive
from ecg_ssl.loss import masked_patch_mse_loss

# ---------------------------------------------------------------
# Test 1: RawBeatECGDataset loading
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: RawBeatECGDataset loading")
print("=" * 60)

data_dir = Path("data/ptb-xl")
hea_files = sorted(data_dir.rglob("*.hea"))
record_paths = [str(p.with_suffix("")) for p in hea_files]
print(f"Found {len(record_paths)} records")

dataset = RawBeatECGDataset(
    record_paths[:10],
    sampling_rate=500,
    min_beats=3,
    max_retries=10,
)

item = dataset[0]
print(f"beats shape: {item['beats'].shape}")
print(f"beat_lengths: {item['beat_lengths']}")
print(f"num_beats: {item['num_beats']}")
print(f"max_beat_len: {item['max_beat_len']}")

# ---------------------------------------------------------------
# Test 2: Raw beat collate function
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 2: raw_beat_collate_fn (double padding)")
print("=" * 60)

batch_items = []
for i in range(min(4, len(record_paths))):
    try:
        item = dataset[i]
        batch_items.append(item)
        print(f"  Record {i}: {item['num_beats']} beats, "
              f"max_beat_len={item['max_beat_len']}, "
              f"beat_lengths={item['beat_lengths'].tolist()}")
    except RuntimeError:
        print(f"  Record {i}: failed to load, skipping")

if len(batch_items) < 2:
    print("ERROR: Need at least 2 records for batch test")
    exit()

batch = raw_beat_collate_fn(batch_items)
print(f"\nCollated batch:")
print(f"  beats shape: {batch['beats'].shape}")
print(f"  padding_mask shape: {batch['padding_mask'].shape}")
print(f"  num_beats: {batch['num_beats']}")
print(f"  beat_lengths shape: {batch['beat_lengths'].shape}")
print(f"  global_max_beat_len: {batch['global_max_beat_len']}")

# Verify: beats tensor time dimension matches global_max_beat_len
assert batch["beats"].shape[3] == batch["global_max_beat_len"], \
    "Beat time dimension doesn't match global max!"
print("PASS: Double padding is correct!")

# ---------------------------------------------------------------
# Test 3: Model forward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 3: ECGMaskedSSLBeatAdaptive forward pass")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ECGMaskedSSLBeatAdaptive(
    in_channels=12,
    d_model=256,
    num_heads=8,
    num_layers=4,
    mlp_ratio=4,
    dropout=0.1,
    max_target_dim=12 * 700,  # 12 leads * ~700 max samples (generous upper bound)
).to(device)

beats = batch["beats"].to(device)
padding_mask = batch["padding_mask"].to(device)

print(f"Input beats shape: {beats.shape}")
print(f"Input padding_mask shape: {padding_mask.shape}")

with torch.no_grad():
    output = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)

B, N = batch["beats"].shape[:2]
T = batch["global_max_beat_len"]
target_dim = 12 * T

print(f"\nOutput shapes:")
print(f"  pred_patches: {output['pred_patches'].shape}")
print(f"  target_patches: {output['target_patches'].shape}")
print(f"  mask: {output['mask'].shape}")
print(f"  encoded: {output['encoded'].shape}")
print(f"  pooled: {output['pooled'].shape}")

assert output["pred_patches"].shape == (B, N, target_dim), \
    f"pred_patches shape mismatch! Expected {(B, N, target_dim)}"
assert output["target_patches"].shape == (B, N, target_dim), \
    f"target_patches shape mismatch! Expected {(B, N, target_dim)}"
assert output["mask"].shape == (B, N), "mask shape mismatch!"
assert output["encoded"].shape == (B, N, 256), "encoded shape mismatch!"
assert output["pooled"].shape == (B, 256), "pooled shape mismatch!"
print("All shape checks PASSED!")

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
# Test 5: Backward pass
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST 5: Backward pass")
print("=" * 60)

model.train()
output = model(beats, padding_mask=padding_mask, mask_ratio=0.20, span_len=1)
loss = masked_patch_mse_loss(
    output["pred_patches"],
    output["target_patches"],
    output["mask"],
)
loss.backward()

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
print("ALL TOKENIZER 2 INTEGRATION TESTS COMPLETE")
print("=" * 60)
