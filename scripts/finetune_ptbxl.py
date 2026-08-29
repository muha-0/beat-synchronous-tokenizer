"""Fine-tune a pretrained encoder on PTB-XL five-superclass diagnostic classification."""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.ptbxl import (
    SUPERCLASSES,
    PTBXLBeatDataset,
    PTBXLBeatHRDataset,
    PTBXLDataset,
    PTBXLRawBeatDataset,
    beat_collate_fn,
    beat_hr_collate_fn,
    load_ptbxl_splits,
    raw_beat_collate_fn,
)
from bst.models.classifiers import (
    PTBXLAdaptiveBeatClassifier,
    PTBXLBeatClassifier,
    PTBXLBeatHRClassifier,
    PTBXLClassifier,
)
from bst.models.masked_ssl import (
    ECGMaskedSSL,
    ECGMaskedSSLBeat,
    ECGMaskedSSLBeatAdaptive,
    ECGMaskedSSLBeatHR,
)
from bst.training.ptbxl_finetune import (
    evaluate_beat,
    evaluate_beat_hr,
    evaluate_fixed,
    train_one_epoch_beat,
    train_one_epoch_beat_hr,
    train_one_epoch_fixed,
)

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


def build_pretrained(tokenizer, patch_size, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if tokenizer == "fixed":
        model = ECGMaskedSSL(
            in_channels=IN_CHANNELS, seq_len=5000 // patch_size, d_model=D_MODEL,
            patch_size=patch_size, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        )
    elif tokenizer == "tok1":
        model = ECGMaskedSSLBeat(
            in_channels=IN_CHANNELS, d_model=D_MODEL, beat_length=BEAT_LENGTH,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
            dropout=DROPOUT, max_beats=MAX_BEATS,
        )
    elif tokenizer == "tok2":
        model = ECGMaskedSSLBeatAdaptive(
            in_channels=IN_CHANNELS, d_model=D_MODEL, num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
            max_beats=MAX_BEATS, max_target_dim=MAX_TARGET_DIM,
        )
    else:
        model = ECGMaskedSSLBeatHR(
            in_channels=IN_CHANNELS, d_model=D_MODEL, beat_length=BEAT_LENGTH,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS, mlp_ratio=MLP_RATIO,
            dropout=DROPOUT, max_beats=MAX_BEATS,
        )

    model = model.to(device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded checkpoint from:", ckpt_path)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", choices=["fixed", "tok1", "tok2", "tok3"], required=True)
    p.add_argument("--patch-size", type=int, default=50, choices=[50, 100, 250, 500],
                   help="Fixed tokenizer only")
    p.add_argument("--ptbxl-root", type=Path, required=True)
    p.add_argument("--ssl-ckpt", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    name = f"fixed{args.patch_size}" if args.tokenizer == "fixed" else args.tokenizer
    ssl_ckpt = args.ssl_ckpt or Path(f"checkpoints_{name}_ssl/best.pt")
    ckpt_dir = args.ckpt_dir or Path(f"checkpoints_{name}_downstream")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Model
    # -------------------------
    pretrained_model = build_pretrained(args.tokenizer, args.patch_size, ssl_ckpt, device)

    classifier_cls = {
        "fixed": PTBXLClassifier,
        "tok1": PTBXLBeatClassifier,
        "tok2": PTBXLAdaptiveBeatClassifier,
        "tok3": PTBXLBeatHRClassifier,
    }[args.tokenizer]
    model = classifier_cls(pretrained_model, num_classes=5).to(device)

    # -------------------------
    # Data
    # -------------------------
    train_df, val_df, test_df = load_ptbxl_splits(args.ptbxl_root)
    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)}")

    if args.tokenizer == "fixed":
        datasets = [PTBXLDataset(df) for df in (train_df, val_df, test_df)]
        collate_fn = None
    elif args.tokenizer == "tok1":
        datasets = [PTBXLBeatDataset(df, beat_length=BEAT_LENGTH)
                    for df in (train_df, val_df, test_df)]
        collate_fn = beat_collate_fn
    elif args.tokenizer == "tok2":
        datasets = [PTBXLRawBeatDataset(df, max_beat_len=MAX_BEAT_LEN)
                    for df in (train_df, val_df, test_df)]
        collate_fn = raw_beat_collate_fn
    else:
        datasets = [PTBXLBeatHRDataset(df, beat_length=BEAT_LENGTH)
                    for df in (train_df, val_df, test_df)]
        collate_fn = beat_hr_collate_fn

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn)
    train_loader = DataLoader(datasets[0], shuffle=True, **loader_kw)
    val_loader = DataLoader(datasets[1], shuffle=False, **loader_kw)
    test_loader = DataLoader(datasets[2], shuffle=False, **loader_kw)

    train_fn, eval_fn = {
        "fixed": (train_one_epoch_fixed, evaluate_fixed),
        "tok1": (train_one_epoch_beat, evaluate_beat),
        "tok2": (train_one_epoch_beat, evaluate_beat),
        "tok3": (train_one_epoch_beat_hr, evaluate_beat_hr),
    }[args.tokenizer]

    # -------------------------
    # Training loop
    # -------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_auprc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, val_auprc, val_class_aucs, val_class_auprcs = eval_fn(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss     : {train_loss:.4f}")
        print(f"  Val Loss       : {val_loss:.4f}")
        print(f"  Val Macro AUC  : {val_auc:.4f}")
        print(f"  Val Macro AUPRC: {val_auprc:.4f}")
        for sc in SUPERCLASSES:
            if sc in val_class_aucs:
                print(f"    {sc}: AUC={val_class_aucs[sc]:.4f}  AUPRC={val_class_auprcs[sc]:.4f}")
        print("-" * 40)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  Saved new best model (val Macro AUPRC: {val_auprc:.4f})")

    best_state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state)
    print(f"\nLoaded best model (val Macro AUPRC: {best_val_auprc:.4f})")

    test_loss, test_auc, test_auprc, test_class_aucs, test_class_auprcs = eval_fn(
        model, test_loader, criterion, device
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss       : {round(test_loss, 4)}")
    print(f"Test Macro AUC  : {round(test_auc, 4)}")
    print(f"Test Macro AUPRC: {round(test_auprc, 4)}")
    print("\nPer-class results:")
    for sc in SUPERCLASSES:
        if sc in test_class_aucs:
            print(f"  {sc}: AUC={round(test_class_aucs[sc], 4)}  AUPRC={round(test_class_auprcs[sc], 4)}")


if __name__ == "__main__":
    main()
