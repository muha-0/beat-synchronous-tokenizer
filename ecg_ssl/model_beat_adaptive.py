"""
Beat-synchronous model using Tokenizer 2: Adaptive Pooling CNN.

Key difference from model_beat.py (Tokenizer 1):
- Uses AdaptivePoolingCNNTokenizer instead of ResampleCNNTokenizer
- Beat length is variable (not fixed at 256), so the reconstruction
  target dimension varies per batch
- The prediction head outputs a fixed-size target per beat, using the
  global max beat length in the batch
"""

import math
import torch
import torch.nn as nn

from .tokenizer import AdaptivePoolingCNNTokenizer
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


class ECGMaskedSSLBeatAdaptive(nn.Module):
    """
    Self-supervised masked modeling for Tokenizer 2 (Adaptive Pooling CNN).

    Unlike the Tokenizer 1 model, beats are variable length. The reconstruction
    target uses a fixed projection size (max_target_dim) to keep the prediction
    head weight matrix a fixed size. We project the Transformer output to this
    fixed dimension and only compute loss on the valid (non-padded) portion.
    """

    def __init__(
        self,
        in_channels=12,
        d_model=256,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
        max_beats=128,
        max_target_dim=8192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_target_dim = max_target_dim

        # Tokenizer 2: Adaptive Pooling CNN (no beat_length param needed)
        self.tokenizer = AdaptivePoolingCNNTokenizer(in_channels, d_model)

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

        # Prediction head: outputs max_target_dim values
        # For reconstruction, we only use the first (in_channels * actual_beat_len) values
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_target_dim),
        )

    def forward(self, beats, padding_mask=None, mask_ratio=0.20, span_len=1):
        """
        Args:
            beats: (B, N, 12, T_padded) — variable-length beats, zero-padded
            padding_mask: (B, N) bool — True where beat position is padding
            mask_ratio: fraction of real tokens to mask
            span_len: contiguous mask span length

        Returns:
            dict with:
                pred_patches: (B, N, target_dim) — predicted values
                target_patches: (B, N, target_dim) — actual values (flattened beats)
                mask: (B, N) bool — True where masked for prediction
                encoded: (B, N, D)
                pooled: (B, D)
        """
        B, N, C, T = beats.shape
        target_dim = C * T  # 12 * T_padded (varies per batch)

        # Step 1: Encode beats
        tokens = self.tokenizer(beats)  # (B, N, D)

        # Step 2: Create reconstruction targets (flatten beats)
        target_patches = beats.reshape(B, N, target_dim)  # (B, N, 12*T_padded)

        # Step 3: Create mask
        mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)
        if padding_mask is not None:
            mask = mask & ~padding_mask

        # Step 4: Apply mask
        mask_expanded = self.mask_token.expand(B, N, self.d_model)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_expanded, tokens)

        # Step 5: Positional encoding
        masked_tokens = self.posenc(masked_tokens)

        # Step 6: Transformer
        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)

        # Step 7: Predict — output is max_target_dim, we truncate to target_dim
        pred_full = self.pred_head(encoded)  # (B, N, max_target_dim)
        pred_patches = pred_full[:, :, :target_dim]  # (B, N, target_dim)

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
