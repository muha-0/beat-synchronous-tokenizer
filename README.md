# Beat-Synchronous Tokenization for ECG Transformers

Official code for **"Beat-Synchronous Tokenization for ECG Transformers"** (IEEE MLSP 2026).

Ahmed Sameh, Nolan Wilson, Max Enderlein, Yogatheesan Varatharajah — University of Minnesota Twin Cities.

Transformer-based ECG models commonly tokenize waveforms into fixed temporal patches, which can split heartbeat structures across token boundaries. This repo compares fixed patching against three **beat-synchronous** tokenizers under matched Transformer pretraining and downstream evaluation protocols:

| Tokenizer | Description |
|---|---|
| **Fixed** | Non-overlapping temporal patches (p = 50/100/250/500) embedded by a strided Conv1d |
| **Tok1** | Each cardiac cycle (R-peak to R-peak) resampled to a fixed length, then Conv1d-embedded |
| **Tok2** | Variable-length beats compressed to one token via adaptive average pooling |
| **Tok3** | Tok1 plus an R-R interval embedding added to each beat token |

Evaluated in two settings:

1. **PTB-XL** five-superclass diagnostic classification after masked-reconstruction pretraining on **MIMIC-IV-ECG** (10s, 12-lead, 500 Hz).
2. **Icentia11k** 60-second N vs AFib/AFL rhythm classification after patient-level contrastive pretraining (single-lead, 250 Hz).

## Results

**PTB-XL** (mean ± std over five runs):

| Tokenizer | Tokens | Macro AUROC | Macro AUPRC |
|---|---|---|---|
| Fixed p=50 | 100 | 0.8903 ± 0.0014 | **0.7419 ± 0.0025** |
| Fixed p=100 | 50 | 0.8858 ± 0.0007 | 0.7345 ± 0.0043 |
| Fixed p=250 | 20 | 0.8717 ± 0.0008 | 0.7033 ± 0.0017 |
| Fixed p=500 | 10 | 0.8479 ± 0.0019 | 0.6530 ± 0.0055 |
| Tok1 | 11.2 avg. | **0.8945 ± 0.0012** | 0.7414 ± 0.0037 |
| Tok2 | 11.2 avg. | 0.8276 ± 0.0034 | 0.6328 ± 0.0072 |
| Tok3 | 11.2 avg. | 0.8928 ± 0.0015 | 0.7399 ± 0.0039 |

**Icentia11k** (mean ± std over five runs):

| Tokenizer | Tokens | AUROC | AUPRC |
|---|---|---|---|
| Fixed p=160 | 93 | **0.9888 ± 0.0031** | 0.8514 ± 0.0676 |
| Tok1 | 68.1 avg. | 0.9715 ± 0.0050 | 0.8514 ± 0.0076 |
| Tok3 | 68.1 avg. | 0.9669 ± 0.0065 | **0.8515 ± 0.0202** |

## Repository structure

```
bst/                      # library code
├── models/
│   ├── layers.py         # positional encoding, Transformer encoder
│   ├── tokenizers.py     # fixed / beat / adaptive / +HR / single-lead tokenizers
│   ├── masking.py        # contiguous token masking, masked MSE loss
│   ├── masked_ssl.py     # masked-reconstruction SSL models (MIMIC / PTB-XL)
│   ├── contrastive.py    # contrastive encoder + InfoNCE (Icentia)
│   ├── icentia.py        # Icentia downstream encoders and classifiers
│   └── classifiers.py    # PTB-XL classifier heads
├── data/
│   ├── mimic.py          # MIMIC-IV-ECG preprocessing and cached datasets
│   ├── beats.py          # R-peak detection and beat extraction
│   ├── ptbxl.py          # PTB-XL labels, splits, datasets
│   └── icentia.py        # Icentia11k parsing, labeling, datasets, augmentations
└── training/             # train/eval loops per experiment

scripts/                  # CLI entry points (see below)
notebooks/                # original experiment notebooks (kept as a record)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

All three datasets are publicly available:

- [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) (PhysioNet, credentialed access) — 12-lead pretraining corpus
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet, open access) — diagnostic classification
- [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) — long-term single-lead rhythm classification

## Pretrained checkpoints

Weights are distributed via [GitHub Releases](https://github.com/muha-0/beat-synchronous-tokenizer/releases) and placed into the expected local layout with:

```bash
python scripts/download_checkpoints.py          # all checkpoints
python scripts/download_checkpoints.py tok1     # only Tok1
```

For each PTB-XL tokenizer this fetches the pretrained encoder (`checkpoints_*_ssl/best.pt`) and the fine-tuned classifier (`checkpoints_*_downstream/best.pt`). For Icentia11k it fetches the contrastive encoder (`CNN_tokenizer_1min/checkpoint_50000.pth`) and the three downstream models.

## Reproducing the PTB-XL experiments

```bash
# 1. cache preprocessed MIMIC-IV-ECG signals
python scripts/precompute_mimic.py --mimic-root /data/mimic-iv-ecg/files \
    --cache-dir /data/mimic_precomputed_npys

# 2. cache beat tokens (tok1 / tok2 / tok3)
python scripts/precompute_beats.py --variant tok1 \
    --signal-cache-dir /data/mimic_precomputed_npys \
    --out-dir /data/mimic_precomputed_beats_300

# 3. masked-reconstruction pretraining
python scripts/pretrain_mimic.py --tokenizer fixed --patch-size 50 \
    --cache-dir /data/mimic_precomputed_npys
python scripts/pretrain_mimic.py --tokenizer tok1 \
    --cache-dir /data/mimic_precomputed_beats_300

# 4. PTB-XL fine-tuning + test evaluation
python scripts/finetune_ptbxl.py --tokenizer fixed --patch-size 50 --ptbxl-root /data/PTB-XL
python scripts/finetune_ptbxl.py --tokenizer tok1 --ptbxl-root /data/PTB-XL
```

Repeat step 4 (and the paper's protocol) five times to reproduce the reported mean ± std.

## Reproducing the Icentia11k experiments

```bash
# 1. bandpass-filter and cache all records
python scripts/preprocess_icentia.py --icentia-root /data/icentia11k \
    --processed-root /data/icentia_processed

# 2. patient-level contrastive pretraining (fixed patch tokenizer)
python scripts/pretrain_icentia.py --icentia-root /data/icentia11k \
    --processed-root /data/icentia_processed

# 3. downstream N vs AFib/AFL fine-tuning per tokenizer
python scripts/finetune_icentia.py --mode fixed   --icentia-root /data/icentia11k
python scripts/finetune_icentia.py --mode beat    --icentia-root /data/icentia11k
python scripts/finetune_icentia.py --mode beat_hr --icentia-root /data/icentia11k
```

Validation/test window manifests are regenerated deterministically (fixed seeds), so all tokenizers are evaluated on identical patients, records, window start times, and labels.

## Notebooks

`notebooks/` contains the original research notebooks used to produce the paper's results, kept with their outputs as a record. The library and scripts above are a modular reorganization of the same code.

## Citation

```bibtex
@misc{sameh2026beat,
  title={Beat-Synchronous Tokenization for ECG Transformers},
  author={Sameh, Ahmed and Wilson, Nolan and Enderlein, Max and Varatharajah, Yogatheesan},
  year={2026},
  eprint={2608.30367},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2608.30367}
}
```

The citation will be updated to the final IEEE MLSP proceedings citation once the published version is available.

## License

[MIT](LICENSE)
