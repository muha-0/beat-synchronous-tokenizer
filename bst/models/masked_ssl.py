import torch
import torch.nn as nn

from .layers import SinCosPositionalEncoding, TransformerEncoder
from .masking import contiguous_token_mask
from .tokenizers import (
    AdaptivePoolingCNNTokenizer,
    FixedCNNTokenizer,
    ResampleCNNWithHRTokenizer,
)


class ECGMaskedSSL(nn.Module):
    """Fixed temporal patch tokenizer with masked raw-patch reconstruction."""

    def __init__(
        self,
        in_channels=12,
        seq_len=100,
        d_model=256,
        patch_size=50,
        num_heads=8,
        num_layers=4,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size

        self.tokenizer = FixedCNNTokenizer(in_channels, d_model, patch_size)
        self.posenc = SinCosPositionalEncoding(seq_len, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.patch_dim),
        )

    def patchify(self, x):
        # x: (B, C, T) -> raw patches (B, N, C*P)
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0
        N = T // P
        patches = x.view(B, C, N, P).permute(0, 2, 1, 3).contiguous()
        patches = patches.view(B, N, C * P)
        return patches

    def forward(self, x, mask=None, mask_ratio=0.50, span_len=5):
        tokens = self.tokenizer(x)                # (B, N, D)
        B, N, D = tokens.shape

        target_patches = self.patchify(x)
        if mask is None:
            mask = contiguous_token_mask(B, N, mask_ratio, x.device, span_len=span_len)

        mask_token = self.mask_token.to(dtype=tokens.dtype, device=tokens.device).expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)

        masked_tokens = self.posenc(masked_tokens)
        encoded = self.encoder(masked_tokens)
        pred_patches = self.pred_head(encoded)
        pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }


class ECGMaskedSSLBeat(nn.Module):
    """Tok1 — resampled beat tokens."""

    def __init__(self, in_channels=12, d_model=256, beat_length=300,
                 num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.1, max_beats=35):
        super().__init__()
        self.beat_dim = in_channels * beat_length

        self.tokenizer = FixedCNNTokenizer(in_channels, d_model, beat_length)
        self.posenc = SinCosPositionalEncoding(max_beats, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, mlp_ratio, dropout)

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.beat_dim),
        )

    def forward(self, beats, padding_mask=None, mask=None, mask_ratio=0.50, span_len=1):
        B, N, C, T = beats.shape
        tokens = self.tokenizer(beats.view(B * N, C, T))  # (B*N, 1, D)
        tokens = tokens.squeeze(1).view(B, N, -1)         # (B, N, D)
        B, N, D = tokens.shape

        target_patches = beats.reshape(B, N, self.beat_dim)

        if mask is None:
            mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)
        if padding_mask is not None:
            mask = mask & ~padding_mask

        mask_token = self.mask_token.expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
        masked_tokens = self.posenc(masked_tokens)

        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)
        pred_patches = self.pred_head(encoded)

        if padding_mask is not None:
            real = (~padding_mask).unsqueeze(-1).float()
            pooled = (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }


class ECGMaskedSSLBeatAdaptive(nn.Module):
    """Tok2 — adaptive pooled beat tokens."""

    def __init__(self, in_channels=12, d_model=256,
                 num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.1,
                 max_beats=35, max_target_dim=7200):
        super().__init__()
        self.max_target_dim = max_target_dim

        self.tokenizer = AdaptivePoolingCNNTokenizer(in_channels, d_model)
        self.posenc = SinCosPositionalEncoding(max_beats, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, mlp_ratio, dropout)

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_target_dim),
        )

    def forward(self, beats, padding_mask=None, mask=None, mask_ratio=0.50, span_len=1):
        B, N, C, T = beats.shape
        target_dim = C * T

        tokens = self.tokenizer(beats.view(B * N, C, T))
        tokens = tokens.squeeze(1).view(B, N, -1)
        B, N, D = tokens.shape

        target_patches = beats.reshape(B, N, target_dim)

        if mask is None:
            mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)
        if padding_mask is not None:
            mask = mask & ~padding_mask

        mask_token = self.mask_token.expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
        masked_tokens = self.posenc(masked_tokens)

        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)
        pred_full = self.pred_head(encoded)
        pred_patches = pred_full[:, :, :target_dim]

        if padding_mask is not None:
            real = (~padding_mask).unsqueeze(-1).float()
            pooled = (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }


class ECGMaskedSSLBeatHR(nn.Module):
    """Tok3 — resampled beat tokens with R-R interval embedding."""

    def __init__(self, in_channels=12, d_model=256, beat_length=300,
                 num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.1, max_beats=35):
        super().__init__()
        self.beat_dim = in_channels * beat_length

        self.tokenizer = ResampleCNNWithHRTokenizer(in_channels, d_model, beat_length)
        self.posenc = SinCosPositionalEncoding(max_beats, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, mlp_ratio, dropout)

        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.beat_dim),
        )

    def forward(self, beats, rr_intervals, padding_mask=None, mask=None,
                mask_ratio=0.50, span_len=1):
        tokens = self.tokenizer(beats, rr_intervals)   # (B, N, d_model)
        B, N, D = tokens.shape

        target_patches = beats.reshape(B, N, self.beat_dim)

        if mask is None:
            mask = contiguous_token_mask(B, N, mask_ratio, tokens.device, span_len=span_len)
        if padding_mask is not None:
            mask = mask & ~padding_mask

        mask_token = self.mask_token.expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
        masked_tokens = self.posenc(masked_tokens)

        encoded = self.encoder(masked_tokens, src_key_padding_mask=padding_mask)
        pred_patches = self.pred_head(encoded)

        if padding_mask is not None:
            real = (~padding_mask).unsqueeze(-1).float()
            pooled = (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }
