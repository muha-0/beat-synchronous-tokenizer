"""Icentia11k 60-second N vs AFib/AFL classification from the contrastive checkpoint.

fixed   -> fixed patch tokenizer (pretrained encoder used directly)
beat    -> beat-synchronous tokenizer (Tok1, pretrained conv/Transformer weights)
beat_hr -> beat-synchronous + R-R interval tokenizer (Tok3)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.icentia import (
    BEAT_LENGTH,
    MAX_BEATS,
    IcentiaManifestDataset,
    IcentiaTrainDataset,
    beat_collate,
    beat_hr_collate,
    build_manifest,
    fixed_collate,
    list_patients,
    split_patients,
)
from bst.models.icentia import (
    BeatSyncClassifier,
    BeatSyncECGEncoder,
    BeatSyncHRClassifier,
    BeatSyncHRECGEncoder,
    FixedClassifier,
    FixedPatchECGEncoder,
)
from bst.training.icentia_finetune import (
    evaluate_beat,
    evaluate_beat_hr,
    evaluate_fixed,
    run_training,
    train_one_epoch_beat,
    train_one_epoch_beat_hr,
    train_one_epoch_fixed,
)

D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
DROPOUT = 0.1
P_AF_TRAIN = 0.5                  # balanced training

DEFAULT_CKPT_DIRS = {
    "fixed": "checkpoints_fixed_icentia",
    "beat": "checkpoints_beatsync_icentia",
    "beat_hr": "checkpoints_beatsynchr_icentia",
}


def build_model(mode, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    print(f"Loaded checkpoint at step {ckpt.get('step', 'unknown')}")

    if mode == "fixed":
        encoder = FixedPatchECGEncoder(
            d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            dropout=DROPOUT, max_len=95, patch_len=BEAT_LENGTH,
        ).to(device)

        # the pretrained model has extra proj-head keys — filter to matching keys only
        enc_state = {k: v for k, v in state_dict.items()
                     if k in encoder.state_dict()
                     and v.shape == encoder.state_dict()[k].shape}
        missing = encoder.load_state_dict(enc_state, strict=False)
        print(f"Fixed encoder — missing: {missing.missing_keys}")
        return FixedClassifier(encoder, D_MODEL, num_classes=2).to(device)

    if mode == "beat":
        encoder = BeatSyncECGEncoder(
            d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            dropout=DROPOUT, beat_length=BEAT_LENGTH, max_beats=MAX_BEATS,
        ).to(device)
        classifier_cls = BeatSyncClassifier
    else:
        encoder = BeatSyncHRECGEncoder(
            d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
            dropout=DROPOUT, beat_length=BEAT_LENGTH, max_beats=MAX_BEATS,
        ).to(device)
        classifier_cls = BeatSyncHRClassifier

    # remap tokenizer.conv.* -> conv.* and truncate positional embedding
    remap = {}
    for k, v in state_dict.items():
        if k.startswith("tokenizer.conv."):
            remap[k.replace("tokenizer.conv.", "conv.")] = v
        else:
            remap[k] = v
    remap["pos_embed"] = state_dict["pos_embed"][:, :MAX_BEATS, :]

    enc_state = {k: v for k, v in remap.items()
                 if k in encoder.state_dict()
                 and v.shape == encoder.state_dict()[k].shape}
    missing = encoder.load_state_dict(enc_state, strict=False)
    print(f"{mode} encoder — missing: {missing.missing_keys}")
    return classifier_cls(encoder, D_MODEL, num_classes=2).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["fixed", "beat", "beat_hr"], required=True)
    p.add_argument("--icentia-root", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=Path("CNN_tokenizer_1min/checkpoint_50000.pth"),
                   help="Contrastive pretraining checkpoint")
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = args.ckpt_dir or Path(DEFAULT_CKPT_DIRS[args.mode])

    patients = list_patients(args.icentia_root)
    ssl_patients, train_patients, val_patients, test_patients = split_patients(patients)
    print(f"SSL: {len(ssl_patients)} | Train: {len(train_patients)} | "
          f"Val: {len(val_patients)} | Test: {len(test_patients)}")

    model = build_model(args.mode, args.ckpt, device)

    # frozen evaluation manifests, identical for all tokenizers (seeds 43 / 44)
    print("Building val manifest...")
    val_manifest = build_manifest(val_patients, windows_per_patient=2, seed=43)
    print(f"Val manifest:  {len(val_manifest)} windows")

    print("\nBuilding test manifest...")
    test_manifest = build_manifest(test_patients, windows_per_patient=2, seed=44)
    print(f"Test manifest: {len(test_manifest)} windows")

    for name, mf in [("Val", val_manifest), ("Test", test_manifest)]:
        labels = np.array([e["label"] for e in mf])
        n_af = (labels == 1).sum()
        n_nsr = (labels == 0).sum()
        print(f"{name}: N={len(labels)} | AF={n_af} ({100*n_af/len(labels):.1f}%) | "
              f"NSR={n_nsr} ({100*n_nsr/len(labels):.1f}%)")

    train_ds = IcentiaTrainDataset(train_patients, mode=args.mode, p_af=P_AF_TRAIN, seed=0)
    val_ds = IcentiaManifestDataset(val_manifest, mode=args.mode)
    test_ds = IcentiaManifestDataset(test_manifest, mode=args.mode)

    collate_fn = {
        "fixed": fixed_collate,
        "beat": beat_collate,
        "beat_hr": beat_hr_collate,
    }[args.mode]

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              persistent_workers=False, prefetch_factor=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=collate_fn, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, collate_fn=collate_fn, **kw)

    train_fn, eval_fn = {
        "fixed": (train_one_epoch_fixed, evaluate_fixed),
        "beat": (train_one_epoch_beat, evaluate_beat),
        "beat_hr": (train_one_epoch_beat_hr, evaluate_beat_hr),
    }[args.mode]

    print("=" * 50)
    print(f"MODE: {args.mode}")
    print("=" * 50)

    results = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_fn=train_fn,
        eval_fn=eval_fn,
        ckpt_dir=str(ckpt_dir),
        device=device,
        num_epochs=args.epochs,
    )
    print(f"\n{args.mode}: AUROC={results['AUROC']:.4f}  AUPRC={results['AUPRC']:.4f}")


if __name__ == "__main__":
    main()
