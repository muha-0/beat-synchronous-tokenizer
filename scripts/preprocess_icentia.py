"""Bandpass-filter all Icentia11k records once and cache them as .npy files."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.icentia import list_patients, preprocess_to_memmap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--icentia-root", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, required=True)
    args = p.parse_args()

    patients = list_patients(args.icentia_root)
    print("num patients:", len(patients))
    preprocess_to_memmap(patients, args.processed_root, root=args.icentia_root)


if __name__ == "__main__":
    main()
