#!/usr/bin/env python3
"""
Training script for beat-synchronous ECG Masked SSL.

For sanity check on local GPU:
    python train_beat.py --data_dir data/ptb-xl --epochs 3 --batch_size 4

For full training on A100/H100 (point to MIMIC):
    python train_beat.py --data_dir /path/to/mimic --epochs 3 --batch_size 64
"""

import argparse
import math
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from ecg_ssl.model_beat import ECGMaskedSSLBeat
from ecg_ssl.dataset import BeatECGDataset, beat_collate_fn
from ecg_ssl.loss import masked_patch_mse_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Beat-synchronous ECG SSL training")
    parser.add_argument("--data_dir", type=str, default="data/ptb-xl",
                        help="Path to ECG data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_beat_sync_resample",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beat_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.20)
    parser.add_argument("--mask_span_len", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 for Windows compatibility)")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_record_paths(data_dir):
    data_path = Path(data_dir)
    assert data_path.exists(), f"Data directory does not exist: {data_path}"

    hea_files = sorted(data_path.rglob("*.hea"))
    record_paths = [str(p.with_suffix("")) for p in hea_files]
    record_paths = [
        rp for rp in record_paths
        if Path(rp + ".hea").exists() and Path(rp + ".dat").exists()
    ]
    return record_paths


def run_beat_epoch(model, loader, device, mask_ratio, mask_span_len,
                   optimizer=None, scheduler=None, train=True,
                   global_step=0, log_every=5, step_history=None):
    """Training/validation loop for beat-synchronous model."""
    model.train(train)

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            beats = batch["beats"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = model(
                beats,
                padding_mask=padding_mask,
                mask_ratio=mask_ratio,
                span_len=mask_span_len,
            )
            loss = masked_patch_mse_loss(
                out["pred_patches"],
                out["target_patches"],
                out["mask"],
            )

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                batch_loss = loss.item()

                if step_history is not None:
                    step_history["step"].append(global_step)
                    step_history["batch_loss"].append(batch_loss)

                    if len(step_history["ema_loss"]) == 0:
                        ema = batch_loss
                    else:
                        ema = 0.03 * batch_loss + 0.97 * step_history["ema_loss"][-1]
                    step_history["ema_loss"].append(ema)

            running_loss += loss.item()
            n_batches += 1

            if train and (step % log_every == 0):
                elapsed = time.time() - start_time
                avg_loss = running_loss / n_batches
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step:5d}/{len(loader):5d} | "
                    f"batch_loss {loss.item():.4f} | "
                    f"avg_loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

    return running_loss / max(n_batches, 1), global_step


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset
    record_paths = build_record_paths(args.data_dir)
    print(f"Found {len(record_paths):,} records in {args.data_dir}")

    dataset = BeatECGDataset(
        record_paths,
        beat_length=args.beat_length,
        sampling_rate=500,
        min_beats=3,
        max_retries=50,
    )

    # Train/val split
    n_total = len(dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=beat_collate_fn,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    print(f"Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,}")

    # Build model
    model = ECGMaskedSSLBeat(
        in_channels=12,
        d_model=args.d_model,
        beat_length=args.beat_length,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,}")

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    history = []
    step_history = {"step": [], "batch_loss": [], "ema_loss": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss, global_step = run_beat_epoch(
            model=model,
            loader=train_loader,
            device=device,
            mask_ratio=args.mask_ratio,
            mask_span_len=args.mask_span_len,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            global_step=global_step,
            log_every=args.log_every,
            step_history=step_history,
        )

        val_loss, _ = run_beat_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mask_ratio=args.mask_ratio,
            mask_span_len=args.mask_span_len,
            train=False,
            global_step=global_step,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {
                "BEAT_LENGTH": args.beat_length,
                "D_MODEL": args.d_model,
                "NUM_HEADS": args.num_heads,
                "NUM_LAYERS": args.num_layers,
                "MASK_RATIO": args.mask_ratio,
                "MASK_SPAN_LEN": args.mask_span_len,
                "BATCH_SIZE": args.batch_size,
                "LR": args.lr,
                "objective": "beat_sync_raw_reconstruction",
                "tokenizer": "ResampleCNN",
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"  Saved new best checkpoint (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Plot loss curve
    if len(step_history["step"]) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(step_history["step"], step_history["batch_loss"],
                 alpha=0.20, label="Batch loss")
        plt.plot(step_history["step"], step_history["ema_loss"],
                 linewidth=2, label="EMA loss")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Beat-Synchronous SSL Training Monitor")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(checkpoint_dir / "loss_curve.png")
        print(f"Loss curve saved to {checkpoint_dir / 'loss_curve.png'}")
        plt.close()

    # Plot epoch-level losses
    if len(history) > 1:
        plt.figure(figsize=(8, 5))
        plt.plot([h["epoch"] for h in history],
                 [h["train_loss"] for h in history], marker="o", label="Train")
        plt.plot([h["epoch"] for h in history],
                 [h["val_loss"] for h in history], marker="o", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Beat-Synchronous SSL Epoch Losses")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(checkpoint_dir / "epoch_losses.png")
        print(f"Epoch losses saved to {checkpoint_dir / 'epoch_losses.png'}")
        plt.close()


if __name__ == "__main__":
    main()
