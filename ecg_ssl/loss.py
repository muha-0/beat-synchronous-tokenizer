import torch


def masked_patch_mse_loss(pred_patches, target_patches, mask, norm_target=True):
    pred_patches = pred_patches.float()
    target_patches = target_patches.float()

    if norm_target:
        mean = target_patches.mean(dim=-1, keepdim=True)
        std = target_patches.std(dim=-1, keepdim=True).clamp(min=1e-6)

        target_patches = (target_patches - mean) / std
        pred_patches = (pred_patches - mean) / std

    mask_expanded = mask.unsqueeze(-1).expand_as(pred_patches)
    diff2 = (pred_patches - target_patches) ** 2
    return diff2[mask_expanded].mean()
