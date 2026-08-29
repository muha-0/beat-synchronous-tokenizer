"""Build the clean MIMIC-IV-ECG record list and cache preprocessed (12, 5000) arrays."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.mimic import record_to_cache_name, sanitize, zscore_per_lead

CLIP_VALUE = 5.0
MAX_MISSING_FRAC = 0.05


def build_clean_list(base, clean_list_path):
    hea_files = sorted(base.rglob("*.hea"))
    clean_record_paths = []

    for hea in tqdm(hea_files, desc="Scanning headers"):
        rp = str(hea.with_suffix(""))
        dat = Path(rp + ".dat")

        if not dat.exists():
            continue
        if hea.stat().st_size == 0:
            continue
        if dat.stat().st_size == 0:
            continue

        # reject duplicated header files
        txt = hea.read_text(errors="replace")
        lines = [x.rstrip("\n\r") for x in txt.splitlines() if x.strip() != ""]

        if len(lines) % 2 == 0:
            half = len(lines) // 2
            if lines[:half] == lines[half:]:
                continue

        clean_record_paths.append(rp)

    clean_list_path.write_text("\n".join(clean_record_paths))
    print(f"Clean usable record paths : {len(clean_record_paths):,}")
    print(f"Saved to                  : {clean_list_path.resolve()}")
    return clean_record_paths


def precompute(record_paths, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    manifest = []

    for rp in tqdm(record_paths, desc="Precomputing"):
        out_name = record_to_cache_name(rp)
        out_path = cache_dir / out_name

        if out_path.exists():
            manifest.append({"record_path": rp, "npy_path": str(out_path)})
            saved += 1
            continue

        rec = wfdb.rdrecord(rp)
        x = rec.p_signal.astype(np.float32).T   # (12, 5000)

        if x.shape != (12, 5000):
            skipped += 1
            continue

        x = sanitize(x, max_missing_frac=MAX_MISSING_FRAC)
        if x is None:
            skipped += 1
            continue

        x = zscore_per_lead(x, clip_value=CLIP_VALUE)
        if x is None:
            skipped += 1
            continue

        np.save(out_path, x)
        manifest.append({"record_path": rp, "npy_path": str(out_path)})
        saved += 1

    manifest_path = cache_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"Saved precomputed arrays : {saved:,}")
    print(f"Skipped during preprocess: {skipped:,}")
    print(f"Manifest                 : {manifest_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mimic-root", type=Path, required=True,
                   help="Path to mimic-iv-ecg/files")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Output directory for preprocessed .npy arrays")
    p.add_argument("--clean-list", type=Path, default=Path("clean_record_paths.txt"))
    args = p.parse_args()

    assert args.mimic_root.exists(), f"Base path does not exist: {args.mimic_root}"

    if args.clean_list.exists():
        record_paths = args.clean_list.read_text().splitlines()
        print(f"Loaded {len(record_paths):,} record paths from {args.clean_list}")
    else:
        record_paths = build_clean_list(args.mimic_root, args.clean_list)

    precompute(record_paths, args.cache_dir)


if __name__ == "__main__":
    main()
