import math
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast

from ..models.contrastive import info_nce


def cosine_lr(step, total_steps, lr_max, lr_min=1e-5, warmup_steps=200):
    if warmup_steps > 0 and step < warmup_steps:
        return lr_max * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def train_ssl(
    model, loader, device,
    epochs=1,
    lr=3e-4,
    wd=0.05,
    temperature=0.1,
    max_steps=None,
    log_every=20,
    grad_clip=1.0,
    warmup_steps=200,
    lr_min=1e-5,
    use_cosine=True,
    exp_dir=None,
):
    model.to(device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=(device.type == "cuda"))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    losses, lrs, grad_norms, step_times = [], [], [], []
    step = 0

    if max_steps is None:
        max_steps = epochs * len(loader)

    for ep in range(epochs):
        for batch in loader:
            if step >= max_steps:
                return {"losses": losses, "lrs": lrs, "grad_norms": grad_norms, "step_times": step_times}

            t0 = time.time()

            x = batch["x"].to(device, non_blocking=True).float()
            pair_ids = batch["pair_ids"].to(device, non_blocking=True)

            if use_cosine:
                cur_lr = cosine_lr(step, max_steps, lr_max=lr, lr_min=lr_min, warmup_steps=warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
            else:
                cur_lr = opt.param_groups[0]["lr"]

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                z = model(x)  # [2B, D]
                loss = info_nce(z, pair_ids, temperature=temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip else 0.0

            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            lrs.append(float(cur_lr))
            grad_norms.append(float(gnorm.detach().cpu()) if torch.is_tensor(gnorm) else float(gnorm))
            step_times.append(time.time() - t0)

            if step % log_every == 0:
                views_per_sec = x.size(0) / max(step_times[-1], 1e-9)
                print(f"ep={ep} step={step} loss={losses[-1]:.4f} lr={cur_lr:.2e} "
                      f"gnorm={grad_norms[-1]:.2f} views/s={views_per_sec:.1f}")

            step += 1
            if exp_dir and step % 1000 == 0:
                checkpoint_path = os.path.join(exp_dir, f"checkpoint_{step}.pth")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'loss': losses[-1],
                }, checkpoint_path)
                print(f"--- Checkpoint saved at step {step} ---")

    return {"losses": losses, "lrs": lrs, "grad_norms": grad_norms, "step_times": step_times}
