"""Download pretrained checkpoints from GitHub Releases into the expected layout.

Usage:
    python scripts/download_checkpoints.py            # everything
    python scripts/download_checkpoints.py tok1       # only assets matching "tok1"
"""
import sys
import urllib.request
from pathlib import Path

REPO = "muha-0/beat-synchronous-tokenizer"
TAG = "v1.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

# release asset name -> local path
ASSETS = {
    # PTB-XL track: pretrained encoders (SSL) and fine-tuned classifiers (downstream)
    "fixed50_ssl.pt": "checkpoints_fixed50_ssl/best.pt",
    "fixed50_downstream.pt": "checkpoints_fixed50_downstream/best.pt",
    "fixed100_ssl.pt": "checkpoints_fixed100_ssl/best.pt",
    "fixed100_downstream.pt": "checkpoints_fixed100_downstream/best.pt",
    "fixed250_ssl.pt": "checkpoints_fixed250_ssl/best.pt",
    "fixed250_downstream.pt": "checkpoints_fixed250_downstream/best.pt",
    "fixed500_ssl.pt": "checkpoints_fixed500_ssl/best.pt",
    "fixed500_downstream.pt": "checkpoints_fixed500_downstream/best.pt",
    "tok1_ssl.pt": "checkpoints_tok1_ssl/best.pt",
    "tok1_downstream.pt": "checkpoints_tok1_downstream/best.pt",
    "tok2_ssl.pt": "checkpoints_tok2_ssl/best.pt",
    "tok2_downstream.pt": "checkpoints_tok2_downstream/best.pt",
    "tok3_ssl.pt": "checkpoints_tok3_ssl/best.pt",
    "tok3_downstream.pt": "checkpoints_tok3_downstream/best.pt",
    # Icentia11k track
    "icentia_contrastive.pth": "CNN_tokenizer_1min/checkpoint_50000.pth",
    "icentia_fixed.pt": "checkpoints_fixed_icentia/best.pt",
    "icentia_beatsync.pt": "checkpoints_beatsync_icentia/best.pt",
    "icentia_beatsynchr.pt": "checkpoints_beatsynchr_icentia/best.pt",
}


def download(asset, dest):
    dest = Path(dest)
    if dest.exists():
        print(f"skip (exists)  {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{asset}"
    print(f"downloading    {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    selected = {a: p for a, p in ASSETS.items() if pattern in a}
    if not selected:
        print(f"No assets match '{pattern}'. Available:")
        for a in ASSETS:
            print(f"  {a}")
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    for asset, rel_path in selected.items():
        download(asset, root / rel_path)

    print(f"\nDone. {len(selected)} checkpoint(s) ready.")


if __name__ == "__main__":
    main()
