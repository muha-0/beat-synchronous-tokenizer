import torch
import torch.nn as nn

from .tokenizers import ConvPatchTokenizer, HREncoder


class FixedPatchECGEncoder(nn.Module):
    """Fixed patch tokenizer encoder — identical architecture to pretraining."""

    def __init__(self, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, max_len=95, patch_len=160):
        super().__init__()
        self.tokenizer = ConvPatchTokenizer(d_model, patch_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):   # x: [B, T]
        tok = self.tokenizer(x)
        tok = tok + self.pos_embed[:, :tok.size(1), :]
        h = self.norm(self.tr(tok)).mean(dim=1)        # mean pool
        return h


class BeatSyncECGEncoder(nn.Module):
    """Same Conv1d kernel as ConvPatchTokenizer but applied per-beat (Tok1).
    Weights are loaded from the pretrained fixed-patch checkpoint."""

    def __init__(self, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, beat_length=160, max_beats=93):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, kernel_size=beat_length, stride=beat_length)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_beats, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, beats, padding_mask=None):
        # beats: [B, N, 1, beat_length]
        B, N, C, T = beats.shape
        tok = self.conv(beats.view(B * N, C, T))           # [B*N, D, 1]
        tok = tok.squeeze(-1).view(B, N, -1)               # [B, N, D]
        tok = tok + self.pos_embed[:, :N, :]
        h = self.norm(self.tr(tok, src_key_padding_mask=padding_mask))

        if padding_mask is not None:
            real = (~padding_mask).unsqueeze(-1).float()
            pooled = (h * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        return pooled


class BeatSyncHRECGEncoder(nn.Module):
    """Beat-sync + explicit R-R interval encoding (Tok3)."""

    def __init__(self, d_model=256, n_layers=6, n_heads=8,
                 dropout=0.1, beat_length=160, max_beats=93):
        super().__init__()
        self.conv = nn.Conv1d(1, d_model, kernel_size=beat_length, stride=beat_length)
        self.hr_encoder = HREncoder(d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_beats, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, beats, rr_intervals, padding_mask=None):
        # beats: [B, N, 1, beat_length], rr_intervals: [B, N] in seconds
        B, N, C, T = beats.shape
        tok = self.conv(beats.view(B * N, C, T))
        tok = tok.squeeze(-1).view(B, N, -1)
        tok = tok + self.hr_encoder(rr_intervals)
        tok = tok + self.pos_embed[:, :N, :]
        h = self.norm(self.tr(tok, src_key_padding_mask=padding_mask))

        if padding_mask is not None:
            real = (~padding_mask).unsqueeze(-1).float()
            pooled = (h * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        return pooled


class FixedClassifier(nn.Module):
    def __init__(self, encoder, d_model=256, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class BeatSyncClassifier(nn.Module):
    def __init__(self, encoder, d_model=256, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, beats, padding_mask):
        return self.classifier(self.encoder(beats, padding_mask))


class BeatSyncHRClassifier(nn.Module):
    def __init__(self, encoder, d_model=256, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, beats, rr_intervals, padding_mask):
        return self.classifier(self.encoder(beats, rr_intervals, padding_mask))
