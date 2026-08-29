import torch


def contiguous_token_mask(batch_size, seq_len, mask_ratio, device, span_len=1, max_tries=1000):
    """Returns mask: (B, N) bool, True = masked."""
    num_mask = int(round(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        tries = 0
        while int(mask[b].sum().item()) < num_mask and tries < max_tries:
            tries += 1
            remaining = num_mask - int(mask[b].sum().item())
            current_span = min(span_len, remaining)
            start_max = max(1, seq_len - current_span + 1)
            start = torch.randint(0, start_max, (1,), device=device).item()
            end = start + current_span
            if mask[b, start:end].any():
                continue
            mask[b, start:end] = True

        if int(mask[b].sum().item()) < num_mask:
            unmasked_idx = (~mask[b]).nonzero(as_tuple=False).squeeze(1)
            remaining = num_mask - int(mask[b].sum().item())
            if len(unmasked_idx) > 0:
                pick = unmasked_idx[torch.randperm(len(unmasked_idx), device=device)[:remaining]]
                mask[b, pick] = True

    return mask


def masked_patch_mse_loss(pred_patches, target_patches, mask, norm_target=False):
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
