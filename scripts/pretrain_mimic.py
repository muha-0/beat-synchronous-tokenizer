"""Masked-reconstruction SSL pretraining on MIMIC-IV-ECG.

fixed -> fixed temporal patches (--patch-size 50|100|250|500)
tok1  -> resampled beat tokens
tok2  -> adaptive pooled beat tokens
tok3  -> resampled beat tokens + R-R interval embedding
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.mimic import (
    PrecomputedBeatDataset,
    PrecomputedBeatRRDataset,
    PrecomputedECGDataset,
    PrecomputedRawBeatDataset,
    beat_collate_fn,
    beat_rr_collate_fn,
    raw_beat_collate_fn,
)
from bst.models.masked_ssl import (
    ECGMaskedSSL,
    ECGMaskedSSLBeat,
    ECGMaskedSSLBeatAdaptive,
    ECGMaskedSSLBeatHR,
)
from bst.training.masked import run_epoch_beat, run_epoch_beat_hr, run_epoch_fixed

IN_CHANNELS = 12
BEAT_LENGTH = 300
MAX_BEATS = 35
MAX_BEAT_LEN = 600
MAX_TARGET_DIM = IN_CHANNELS * MAX_BEAT_LEN
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
MLP_RATIO = 4
DROPOUT = 0.1

# span lengths used for the paper runs (fixed patches: coarser patches use shorter spans)
FIXED_SPAN_LEN = {50: 5, 100: 3, 250: 1, 500: 1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", choices=["fixed", "tok1", "tok2", "tok3"], required=True)
    p.add_argument("--patch-size", type=int, default=50, choices=[50, 100, 250, 500],
                   help="Fixed tokenizer only")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Signal cache (fixed) or beat cache (tok1/2/3) directory")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.50)
    p.add_argument("--span-len", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.95)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print("Device:", device)

    name = f"fixed{args.patch_size}" if args.tokenizer == "fixed" else args.tokenizer
    ckpt_dir = args.ckpt_dir or Path(f"checkpoints_{name}_ssl")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.span_len is None:
        args.span_len = FIXED_SPAN_LEN[args.patch_size] if args.tokenizer == "fixed" else 1

    # -------------------------
    # Data
    # -------------------------
    manifest_path = args.cache_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]

    if args.tokenizer == "fixed":
        paths = [row["npy_path"] for row in rows]
        dataset = PrecomputedECGDataset(paths)
        collate_fn = None
    else:
        paths = [row["beat_path"] for row in rows]
        dataset_cls = {
            "tok1": PrecomputedBeatDataset,
            "tok2": PrecomputedRawBeatDataset,
            "tok3": PrecomputedBeatRRDataset,
        }[args.tokenizer]
        dataset = dataset_cls(paths)
        collate_fn = {
            "tok1": beat_collate_fn,
            "tok2": raw_beat_collate_fn,
            "tok3": beat_rr_collate_fn,
        }[args.tokenizer]

    print(f"Precomputed arrays loaded: {len(paths):,}")
    if args.max_records is not None:
        paths = paths[:args.max_records]
        dataset = type(dataset)(paths)
        print(f"Using first {len(paths):,} cached records")

    n_total = len(dataset)
    n_train = int(args.train_frac * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train size: {len(train_dataset):,}")
    print(f"Val size  : {len(val_dataset):,}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4,
                     collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kw)

    # -------------------------
    # Model / optimizer
    # -------------------------
    if args.tokenizer == "fixed":
        seq_len = 5000 // args.patch_size
        model = ECGMaskedSSL(
            in_channels=IN_CHANNELS, seq_len=seq_len, d_model=D_MODEL,
            patch_size=args.patch_size, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        ).to(device)
        config = {
            "PATCH_SIZE": args.patch_size, "D_MODEL": D_MODEL, "NUM_HEADS": NUM_HEADS,
            "NUM_LAYERS": NUM_LAYERS, "MASK_RATIO": args.mask_ratio,
            "MASK_SPAN_LEN": args.span_len, "BATCH_SIZE": args.batch_size,
            "LR": args.lr, "objective": "raw_patch_reconstruction",
        }
    elif args.tokenizer == "tok1":
        model = ECGMaskedSSLBeat(
            in_channels=IN_CHANNELS, d_model=D_MODEL, beat_length=BEAT_LENGTH,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
            dropout=DROPOUT, max_beats=MAX_BEATS,
        ).to(device)
        config = {
            "BEAT_LENGTH": BEAT_LENGTH, "MAX_BEATS": MAX_BEATS, "D_MODEL": D_MODEL,
            "NUM_HEADS": NUM_HEADS, "NUM_LAYERS": NUM_LAYERS, "MASK_RATIO": args.mask_ratio,
            "MASK_SPAN_LEN": args.span_len, "BATCH_SIZE": args.batch_size,
            "LR": args.lr, "objective": "beat_sync_resample_reconstruction",
        }
    elif args.tokenizer == "tok2":
        model = ECGMaskedSSLBeatAdaptive(
            in_channels=IN_CHANNELS, d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
            max_beats=MAX_BEATS, max_target_dim=MAX_TARGET_DIM,
        ).to(device)
        config = {
            "MAX_BEATS": MAX_BEATS, "MAX_BEAT_LEN": MAX_BEAT_LEN,
            "MAX_TARGET_DIM": MAX_TARGET_DIM, "D_MODEL": D_MODEL,
            "NUM_HEADS": NUM_HEADS, "NUM_LAYERS": NUM_LAYERS, "MASK_RATIO": args.mask_ratio,
            "MASK_SPAN_LEN": args.span_len, "BATCH_SIZE": args.batch_size,
            "LR": args.lr, "objective": "beat_sync_adaptive_reconstruction",
        }
    else:
        model = ECGMaskedSSLBeatHR(
            in_channels=IN_CHANNELS, d_model=D_MODEL, beat_length=BEAT_LENGTH,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
            dropout=DROPOUT, max_beats=MAX_BEATS,
        ).to(device)
        config = {
            "BEAT_LENGTH": BEAT_LENGTH, "MAX_BEATS": MAX_BEATS, "D_MODEL": D_MODEL,
            "NUM_HEADS": NUM_HEADS, "NUM_LAYERS": NUM_LAYERS, "MASK_RATIO": args.mask_ratio,
            "MASK_SPAN_LEN": args.span_len, "BATCH_SIZE": args.batch_size,
            "LR": args.lr, "objective": "beat_sync_resample_hr_reconstruction",
        }

    if not args.no_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Total train steps : {total_steps:,}")
    print(f"Warmup steps      : {warmup_steps:,}")

    if args.tokenizer == "fixed":
        def run_epoch(loader, **kw):
            return run_epoch_fixed(model, loader, device, seq_len,
                                   args.mask_ratio, args.span_len,
                                   log_every=args.log_every, **kw)
    elif args.tokenizer == "tok3":
        def run_epoch(loader, **kw):
            return run_epoch_beat_hr(model, loader, device,
                                     args.mask_ratio, args.span_len,
                                     log_every=args.log_every, **kw)
    else:
        def run_epoch(loader, **kw):
            return run_epoch_beat(model, loader, device,
                                  args.mask_ratio, args.span_len,
                                  log_every=args.log_every, **kw)

    # -------------------------
    # Training loop
    # -------------------------
    best_val_loss = float("inf")
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, global_step = run_epoch(
            train_loader, optimizer=optimizer, scheduler=scheduler,
            train=True, global_step=global_step,
        )
        val_loss, global_step = run_epoch(
            val_loader, train=False, global_step=global_step,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        }
        torch.save(state, ckpt_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, ckpt_dir / "best.pt")
            print(f"  Saved new best checkpoint to: {ckpt_dir / 'best.pt'}")

    print("\nDone training.")
    print("Best val loss:", best_val_loss)
    print("Checkpoint dir:", ckpt_dir)


if __name__ == "__main__":
    main()
