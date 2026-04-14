"""
Beat-synchronous model using Tokenizer 3: Resample + CNN + Heart Rate Feature.

Nearly identical to model_beat.py (Tokenizer 1), with one addition:
the R-R interval duration is encoded and added to each beat embedding
before the Transformer sees it.
"""

import math
import torch
import torch.nn as nn

from .tokenizer import ResampleCNNWithHRTokenizer
from .masking import contiguous_token_mask


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding that handles any sequence length."""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderWithMask(nn.Module):
    """Transformer encoder that accepts a padding mask."""

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


class ECGMaskedSSLBeatHR(nn.Module):
    """
    Self-supervised masked modeling for Tokenizer 3.

    Same as Tokenizer 1's model, but the tokenizer also receives
    R-R intervals and encodes heart rate information into each
    beat embedding.
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

        # Tokenizer 3: Resample + CNN + Heart Rate
        self.tokenizer = ResampleCNNWithHRTokenizer(in_channels, d_model, beat_length)

        # Positional encoding
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_beats)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer encoder
        self.encoder = TransformerEncoderWithMask(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Prediction head: reconstruct raw beat values
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.beat_dim),
        )

    def beats_to_targets(self, beats):
        """Flatten beat tensors into reconstruction targets."""
        B, N, C, T = beats.shape
        return beats.reshape(B, N, C * T)

    def forward(self, beats, rr_intervals, padding_mask=None,
                mask_ratio=0.20, span_len=1):
        """
        Args:
            beats: (B, N, 12, beat_length) — resampled beats
            rr_intervals: (B, N) — R-R interval in seconds per beat
            padding_mask: (B, N) bool — True where token is padding
            mask_ratio: fraction of real tokens to mask
            span_len: contiguous mask span length

        Returns:
            dict with pred_patches, target_patches, mask, encoded, pooled
        """
        # Step 1: Encode beats WITH heart rate information
        tokens = self.tokenizer(beats, rr_intervals)  # (B, N, D)
        B, N, D = tokens.shape

        # Step 2: Reconstruction targets
        target_patches = self.beats_to_targets(beats)  # (B, N, beat_dim)

        # Step 3: Create mask
        mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)
        if padding_mask is not None:
            mask = mask & ~padding_mask

        # Step 4: Apply mask
        mask_expanded = self.mask_token.expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_expanded, tokens)

        # Step 5: Positional encoding
        masked_tokens = self.posenc(masked_tokens)

        # Step 6: Transformer
        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)

        # Step 7: Predict
        pred_patches = self.pred_head(encoded)

        # Step 8: Pooling
        if padding_mask is not None:
            real_mask = (~padding_mask).unsqueeze(-1).float()
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
