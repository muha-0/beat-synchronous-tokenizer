import time

import torch

from ..models.masking import contiguous_token_mask, masked_patch_mse_loss


def run_epoch_fixed(model, loader, device, seq_len, mask_ratio, span_len,
                    optimizer=None, scheduler=None, train=True,
                    global_step=0, log_every=50):
    model.train(train)

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            mask = contiguous_token_mask(x.shape[0], seq_len, mask_ratio, device, span_len=span_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, mask=mask)
                loss = masked_patch_mse_loss(out["pred_patches"], out["target_patches"],
                                             out["mask"], norm_target=False)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            running_loss += loss.item()
            n_batches += 1

            if train and (step % log_every == 0):
                elapsed = time.time() - start_time
                avg_loss = running_loss / n_batches
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step:5d}/{len(loader):5d} | "
                    f"batch_loss {loss.item():.4f} | "
                    f"avg_loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

    return running_loss / max(n_batches, 1), global_step


def run_epoch_beat(model, loader, device, mask_ratio, span_len,
                   optimizer=None, scheduler=None, train=True,
                   global_step=0, log_every=50):
    model.train(train)

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            beats = batch["beats"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            mask = contiguous_token_mask(
                beats.shape[0], beats.shape[1], mask_ratio, device, span_len=span_len
            )
            mask = mask & ~padding_mask   # never mask padding

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(beats, padding_mask=padding_mask, mask=mask)

            loss = masked_patch_mse_loss(out["pred_patches"], out["target_patches"], out["mask"])

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            running_loss += loss.item()
            n_batches += 1

            if train and (step % log_every == 0):
                elapsed = time.time() - start_time
                avg_loss = running_loss / n_batches
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step:5d}/{len(loader):5d} | "
                    f"batch_loss {loss.item():.4f} | "
                    f"avg_loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

    return running_loss / max(n_batches, 1), global_step


def run_epoch_beat_hr(model, loader, device, mask_ratio, span_len,
                      optimizer=None, scheduler=None, train=True,
                      global_step=0, log_every=50):
    model.train(train)

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            beats = batch["beats"].to(device, non_blocking=True)
            rr_intervals = batch["rr_intervals"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            mask = contiguous_token_mask(
                beats.shape[0], beats.shape[1], mask_ratio, device, span_len=span_len
            )
            mask = mask & ~padding_mask

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(beats, rr_intervals, padding_mask=padding_mask, mask=mask)

            loss = masked_patch_mse_loss(out["pred_patches"], out["target_patches"], out["mask"])

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            running_loss += loss.item()
            n_batches += 1

            if train and (step % log_every == 0):
                elapsed = time.time() - start_time
                avg_loss = running_loss / n_batches
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step:5d}/{len(loader):5d} | "
                    f"batch_loss {loss.item():.4f} | "
                    f"avg_loss {avg_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )

    return running_loss / max(n_batches, 1), global_step
