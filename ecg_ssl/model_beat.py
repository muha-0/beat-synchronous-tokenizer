"""
Beat-synchronous version of the ECG Masked SSL model.

This module adapts the original ECGMaskedSSL to work with variable-length
beat token sequences from the beat-synchronous tokenizer.

Key differences from the original:
- Accepts pre-segmented beat tensors instead of raw ECG signals
- Uses sinusoidal positional encoding (handles variable sequence lengths)
- Padding mask support for batches with different numbers of beats
- Reconstruction targets are resampled beat segments, not fixed patches
"""

import math
import torch
import torch.nn as nn

from .tokenizer import ResampleCNNTokenizer
from .masking import contiguous_token_mask


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding that works with any sequence length.

    Unlike the learnable positional encoding in the original model (which
    is hardcoded to 100 positions), this generates position encodings
    on the fly for whatever sequence length it receives.
    """

    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves to GPU with the model but isn't a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, N, D) — just slice the positional encoding to match N
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderWithMask(nn.Module):
    """
    Same Transformer encoder as the original, but accepts a
    src_key_padding_mask to ignore padding tokens.
    """

    def __init__(self, d_model=256, num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class ECGMaskedSSLBeat(nn.Module):
    """
    Self-supervised masked modeling for beat-synchronous ECG tokenization.

    This model:
    1. Receives pre-segmented, resampled beat tensors (from BeatECGDataset)
    2. Encodes each beat with the ResampleCNNTokenizer
    3. Adds sinusoidal positional encoding
    4. Masks a fraction of beat tokens
    5. Runs the Transformer encoder
    6. Predicts the raw values of the masked beats
    """

    def __init__(
        self,
        in_channels=12,
        d_model=256,
        beat_length=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        max_beats=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.beat_length = beat_length
        self.beat_dim = in_channels * beat_length  # 12 * 256 = 3072

        # Beat tokenizer: encodes each (12, 256) beat into a d_model-dim vector
        self.tokenizer = ResampleCNNTokenizer(in_channels, d_model, beat_length)

        # Positional encoding: sinusoidal, handles variable sequence lengths
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_beats)

        # Mask token: learned vector that replaces masked beat positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer encoder with padding mask support
        self.encoder = TransformerEncoderWithMask(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Prediction head: reconstruct the raw beat values at masked positions
        # Output is in_channels * beat_length = 12 * 256 = 3072
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.beat_dim),
        )

    def beats_to_targets(self, beats):
        """
        Flatten beat tensors into reconstruction targets.

        Args:
            beats: (B, N, 12, beat_length)

        Returns:
            targets: (B, N, 12 * beat_length) = (B, N, 3072)
        """
        B, N, C, T = beats.shape
        return beats.reshape(B, N, C * T)

    def forward(self, beats, padding_mask=None, mask_ratio=0.20, span_len=1):
        """
        Args:
            beats: (B, N, 12, beat_length) — pre-segmented, resampled beats
            padding_mask: (B, N) bool — True where token is padding
            mask_ratio: fraction of real (non-padding) tokens to mask
            span_len: length of contiguous mask spans

        Returns:
            dict with:
                pred_patches: (B, N, beat_dim) — predicted raw beat values
                target_patches: (B, N, beat_dim) — actual raw beat values
                mask: (B, N) bool — True where token is masked for prediction
                encoded: (B, N, D) — Transformer output embeddings
                pooled: (B, D) — mean-pooled representation (ignoring padding)
        """
        # Step 1: Encode each beat into a d_model-dim token embedding
        tokens = self.tokenizer(beats)  # (B, N, D)
        B, N, D = tokens.shape

        # Step 2: Create reconstruction targets by flattening beats
        target_patches = self.beats_to_targets(beats)  # (B, N, beat_dim)

        # Step 3: Create mask (only mask real tokens, not padding)
        mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)

        # Don't mask padding positions
        if padding_mask is not None:
            mask = mask & ~padding_mask  # only mask real tokens

        # Step 4: Replace masked tokens with the learned mask token
        mask_token = self.mask_token.expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)

        # Step 5: Add positional encoding
        masked_tokens = self.posenc(masked_tokens)

        # Step 6: Run Transformer (with padding mask)
        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)

        # Step 7: Predict raw beat values at all positions
        pred_patches = self.pred_head(encoded)  # (B, N, beat_dim)

        # Step 8: Mean pooling (ignore padding tokens)
        if padding_mask is not None:
            # Set padding positions to 0 before averaging
            real_mask = (~padding_mask).unsqueeze(-1).float()  # (B, N, 1)
            pooled = (encoded * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }
