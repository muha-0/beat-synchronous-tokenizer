"""Extract beat caches from the precomputed MIMIC signal cache.

tok1 -> resampled beats (N, 12, 300)              [.npy]
tok2 -> raw variable-length beats + lengths       [.npz]
tok3 -> resampled beats + R-R intervals           [.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bst.data.beats import (
    extract_and_resample_beats,
    extract_beats_with_rr,
    extract_raw_beats,
)
from bst.data.mimic import cache_name_npz, record_to_cache_name

BEAT_LENGTH = 300
SAMPLING_RATE = 500
MIN_BEATS = 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["tok1", "tok2", "tok3"], required=True)
    p.add_argument("--signal-cache-dir", type=Path, required=True,
                   help="Directory produced by precompute_mimic.py")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    manifest_path = args.signal_cache_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    print(f"Loaded {len(rows):,} records from signal manifest")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    beat_manifest = []

    for row in tqdm(rows):
        npy_path = row["npy_path"]
        if args.variant == "tok1":
            out_name = record_to_cache_name(npy_path)
        else:
            out_name = cache_name_npz(npy_path)
        out_path = args.out_dir / out_name

        if out_path.exists():
            beat_manifest.append({"npy_path": npy_path, "beat_path": str(out_path)})
            saved += 1
            continue

        x = np.load(npy_path, mmap_mode=None)   # (12, 5000)

        if x.shape != (12, 5000):
            skipped += 1
            continue

        if args.variant == "tok1":
            beats = extract_and_resample_beats(
                x, beat_length=BEAT_LENGTH, sampling_rate=SAMPLING_RATE, min_beats=MIN_BEATS
            )
            if beats is None:
                skipped += 1
                continue
            np.save(out_path, beats)
        elif args.variant == "tok2":
            beat_array, lengths = extract_raw_beats(
                x, sampling_rate=SAMPLING_RATE, min_beats=MIN_BEATS
            )
            if beat_array is None:
                skipped += 1
                continue
            np.savez(out_path, beats=beat_array, lengths=lengths)
        else:
            beat_array, rr_array = extract_beats_with_rr(
                x, beat_length=BEAT_LENGTH, sampling_rate=SAMPLING_RATE, min_beats=MIN_BEATS
            )
            if beat_array is None:
                skipped += 1
                continue
            np.savez(out_path, beats=beat_array, rr_intervals=rr_array)

        beat_manifest.append({"npy_path": npy_path, "beat_path": str(out_path)})
        saved += 1

    beat_manifest_path = args.out_dir / "manifest.jsonl"
    with beat_manifest_path.open("w") as f:
        for row in beat_manifest:
            f.write(json.dumps(row) + "\n")

    print(f"Saved beat arrays : {saved:,}")
    print(f"Skipped           : {skipped:,}")
    print(f"Manifest          : {beat_manifest_path}")


if __name__ == "__main__":
    main()
