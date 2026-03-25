#!/usr/bin/env python3
"""Main training entry point for ECG Masked SSL."""
import math
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from ecg_ssl import config
from ecg_ssl.dataset import build_dataloaders
from ecg_ssl.model import ECGMaskedSSL
from ecg_ssl.trainer import run_epoch


def build_record_paths():
    hea_files = sorted(config.BASE.rglob("*.hea"))
    record_paths = [str(p.with_suffix("")) for p in hea_files]
    record_paths = [
        rp for rp in record_paths
        if Path(rp + ".hea").exists() and Path(rp + ".dat").exists()
    ]
    if config.MAX_RECORDS is not None:
        record_paths = record_paths[:config.MAX_RECORDS]
    return record_paths


def main():
    assert config.BASE.exists(), f"Base path does not exist: {config.BASE}"
    print("Device:", config.device)

    record_paths = build_record_paths()
    print(f"Using {len(record_paths):,} records")

    train_loader, val_loader = build_dataloaders(record_paths)
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")

    model = ECGMaskedSSL(
        in_channels=config.IN_CHANNELS,
        seq_len=config.SEQ_LEN,
        d_model=config.D_MODEL,
        patch_size=config.PATCH_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        mlp_ratio=config.MLP_RATIO,
        dropout=config.DROPOUT,
    ).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = config.EPOCHS * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,}")

    checkpoint_dir = Path("./checkpoints_fixed_ssl_rawpatch")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []
    step_history = {"step": [], "batch_loss": [], "ema_loss": []}
    global_step = 0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            train=True,
            global_step=global_step,
            plot_every=config.PLOT_EVERY,
            step_history=step_history,
        )

        val_loss, _ = run_epoch(
            model=model,
            loader=val_loader,
            train=False,
            global_step=global_step,
        )

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": {
                "PATCH_SIZE": config.PATCH_SIZE,
                "D_MODEL": config.D_MODEL,
                "NUM_HEADS": config.NUM_HEADS,
                "NUM_LAYERS": config.NUM_LAYERS,
                "MASK_RATIO": config.MASK_RATIO,
                "MASK_SPAN_LEN": config.MASK_SPAN_LEN,
                "BATCH_SIZE": config.BATCH_SIZE,
                "LR": config.LR,
                "objective": "raw_patch_reconstruction",
            },
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            print(f"  Saved best checkpoint")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print("Checkpoint dir:", checkpoint_dir)

    epochs_list = [x["epoch"] for x in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, [x["train_loss"] for x in history], marker="o", label="Train loss")
    plt.plot(epochs_list, [x["val_loss"] for x in history], marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch-Level Raw Patch Reconstruction Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(checkpoint_dir / "loss_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
