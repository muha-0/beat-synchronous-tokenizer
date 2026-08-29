import torch.nn as nn


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
        # x: (B, C, T)
        z = self.proj(x)          # (B, D, N)
        z = z.transpose(1, 2)     # (B, N, D)
        return z


class AdaptivePoolingCNNTokenizer(nn.Module):
    def __init__(self, in_channels=12, d_model=256):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=50,
            stride=50,
            bias=True,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, C, T) — T is variable
        z = self.proj(x)      # (B, d_model, T)
        z = self.pool(z)      # (B, d_model, 1)
        z = z.squeeze(-1)     # (B, d_model)
        return z.unsqueeze(1)  # (B, 1, d_model)


class HREncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, rr):
        # rr: (B, N) — R-R intervals in seconds
        return self.mlp(rr.unsqueeze(-1))   # (B, N, d_model)


class ResampleCNNWithHRTokenizer(nn.Module):
    def __init__(self, in_channels=12, d_model=256, beat_length=300):
        super().__init__()
        self.beat_length = beat_length
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=beat_length,
            stride=beat_length,
            bias=True,
        )
        self.hr_encoder = HREncoder(d_model)

    def forward(self, beats, rr_intervals):
        # beats: (B, N, C, T), rr_intervals: (B, N)
        B, N, C, T = beats.shape
        x = beats.view(B * N, C, T)
        z = self.proj(x)                    # (B*N, d_model, 1)
        z = z.squeeze(-1)                   # (B*N, d_model)
        cnn_embeddings = z.view(B, N, -1)   # (B, N, d_model)
        hr_embeddings = self.hr_encoder(rr_intervals)   # (B, N, d_model)
        return cnn_embeddings + hr_embeddings


class ConvPatchTokenizer(nn.Module):
    def __init__(self, d_model=256, patch_len=160):
        super().__init__()
        self.patch_len = patch_len
        self.conv = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=patch_len)

    def forward(self, x):   # x: [B, T]
        return self.conv(x.unsqueeze(1)).transpose(1, 2)   # [B, L, d_model]
