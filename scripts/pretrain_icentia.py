"""Patient-level contrastive pretraining on Icentia11k (60s single-lead windows)."""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.icentia import (
    FS,
    WINDOW_SAMPLES,
    ECGAugment,
    IcentiaPatientPairDataset,
    collate_patient_pairs,
    list_patients,
    seed_worker,
    split_patients,
)
from bst.models.contrastive import ECGEncoder
from bst.models.tokenizers import ConvPatchTokenizer

PATCH_LEN = 160
D_MODEL = 256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--icentia-root", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, required=True,
                   help="Directory produced by preprocess_icentia.py")
    p.add_argument("--exp-dir", type=Path, default=Path("CNN_tokenizer_1min"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.exp_dir, exist_ok=True)

    from bst.training.contrastive import train_ssl

    patients = list_patients(args.icentia_root)
    ssl_patients, train_patients, val_patients, test_patients = split_patients(patients)
    print(f"SSL: {len(ssl_patients)} | Train: {len(train_patients)} | "
          f"Val: {len(val_patients)} | Test: {len(test_patients)}")

    augment = ECGAugment()
    ssl_ds = IcentiaPatientPairDataset(
        patient_dirs=ssl_patients,
        window_samples=WINDOW_SAMPLES,
        processed_root=args.processed_root,
        root=args.icentia_root,
        fs=FS,
        augment=augment,
        seed=123,
        cache_size=64,
    )

    g = torch.Generator()
    g.manual_seed(42)

    ssl_loader = DataLoader(
        ssl_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_patient_pairs,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=4,
    )

    max_len = (WINDOW_SAMPLES // PATCH_LEN) + 2
    tokenizer = ConvPatchTokenizer(d_model=D_MODEL, patch_len=PATCH_LEN)
    model = ECGEncoder(tokenizer=tokenizer, d_model=D_MODEL, max_len=max_len, use_cls=False)

    train_ssl(model, ssl_loader, device, epochs=args.epochs, lr=args.lr,
              max_steps=args.max_steps, exp_dir=str(args.exp_dir))
    print("Done. Checkpoints in:", args.exp_dir)


if __name__ == "__main__":
    main()
