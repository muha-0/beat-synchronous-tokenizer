import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGEncoder(nn.Module):
    """Contrastive pretraining encoder (Icentia11k)."""

    def __init__(
        self,
        tokenizer: nn.Module,
        d_model=256,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        proj_dim=128,
        max_len=1024,          # set based on (WINDOW_SAMPLES // patch_len)
        use_cls=False,         # False => mean pooling
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_cls = use_cls

        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + (1 if use_cls else 0), d_model))

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_h=False):  # x: [B, T]
        tok = self.tokenizer(x)  # [B, L, d_model]
        B, L, D = tok.shape

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            tok = tok + self.pos_embed[:, : (L + 1), :]
        else:
            tok = tok + self.pos_embed[:, :L, :]

        z = self.tr(tok)
        z = self.norm(z)

        if self.use_cls:
            h = z[:, 0]
        else:
            h = z.mean(dim=1)

        y = self.proj(h)
        y = F.normalize(y, dim=-1)

        if return_h:
            return h, y
        return y


def info_nce(z, pair_ids, temperature=0.1):
    z = z.float()  # important under autocast
    sim = ((z @ z.T) / temperature).float()

    N = z.size(0)
    mask = torch.eye(N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

    pid = pair_ids
    pos = (pid.unsqueeze(0) == pid.unsqueeze(1)) & (~mask)

    logsumexp = torch.logsumexp(sim, dim=1)
    pos_sim = sim[pos].view(N)
    return -(pos_sim - logsumexp).mean()
