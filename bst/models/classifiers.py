import torch
import torch.nn as nn


def _no_mask(beats):
    # downstream fine-tuning applies no token masking
    B, N = beats.shape[0], beats.shape[1]
    return torch.zeros(B, N, dtype=torch.bool, device=beats.device)


class PTBXLClassifier(nn.Module):
    def __init__(self, pretrained_model, feature_dim=256, num_classes=1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        N = x.shape[-1] // self.pretrained_model.patch_size
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        out = self.pretrained_model(x, mask=mask)
        pooled = out["pooled"]
        return self.classifier(pooled)


class PTBXLBeatClassifier(nn.Module):
    def __init__(self, pretrained_model, feature_dim=256, num_classes=1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, beats, padding_mask):
        out = self.pretrained_model(beats, padding_mask=padding_mask, mask=_no_mask(beats))
        pooled = out["pooled"]
        return self.classifier(pooled)


class PTBXLAdaptiveBeatClassifier(nn.Module):
    def __init__(self, pretrained_model, feature_dim=256, num_classes=1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, beats, padding_mask):
        out = self.pretrained_model(beats, padding_mask=padding_mask, mask=_no_mask(beats))
        pooled = out["pooled"]
        return self.classifier(pooled)


class PTBXLBeatHRClassifier(nn.Module):
    def __init__(self, pretrained_model, feature_dim=256, num_classes=1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, beats, rr_intervals, padding_mask):
        out = self.pretrained_model(beats, rr_intervals, padding_mask=padding_mask,
                                    mask=_no_mask(beats))
        pooled = out["pooled"]
        return self.classifier(pooled)
