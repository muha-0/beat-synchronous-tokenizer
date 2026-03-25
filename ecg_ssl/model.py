import torch
import torch.nn as nn

from .masking import contiguous_token_mask


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos.to(dtype=x.dtype, device=x.device)


class FixedCNNTokenizer(nn.Module):
    def __init__(self, in_channels=12, d_model=256, patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        # x: (B, 12, 5000)
        z = self.proj(x)          # (B, D, N)
        z = z.transpose(1, 2)     # (B, N, D)
        return z


class TransformerEncoder(nn.Module):
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

    def forward(self, x):
        return self.encoder(x)


class ECGMaskedSSL(nn.Module):
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
        self.posenc = LearnablePositionalEncoding(seq_len, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # reconstruct raw patch values: 12 * 50 = 600 outputs per token
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.patch_dim),
        )

    def patchify(self, x):
        """
        x: (B, C, T)
        returns raw patches: (B, N, C*P)
        """
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0
        N = T // P
        patches = x.view(B, C, N, P).permute(0, 2, 1, 3).contiguous()   # (B, N, C, P)
        patches = patches.view(B, N, C * P)                              # (B, N, C*P)
        return patches

    def forward(self, x, mask_ratio=0.30, span_len=3):
        """
        x: (B, 12, 5000)
        returns:
          pred_patches: (B, N, patch_dim)
          target_patches: (B, N, patch_dim)
          mask: (B, N) bool
          encoded: (B, N, D)
          pooled: (B, D)
        """
        tokens = self.tokenizer(x)                # (B, N, D)
        B, N, D = tokens.shape

        target_patches = self.patchify(x)         # (B, N, 600)
        mask = contiguous_token_mask(B, N, mask_ratio, x.device, span_len=span_len)

        mask_token = self.mask_token.to(dtype=tokens.dtype, device=tokens.device).expand(B, N, D)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)

        masked_tokens = self.posenc(masked_tokens)
        encoded = self.encoder(masked_tokens)
        pred_patches = self.pred_head(encoded)    # (B, N, 600)
        pooled = encoded.mean(dim=1)

        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "mask": mask,
            "encoded": encoded,
            "pooled": pooled,
        }
