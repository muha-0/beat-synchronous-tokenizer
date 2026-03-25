import time
import torch
import matplotlib.pyplot as plt

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(**kwargs):
        pass

from . import config
from .loss import masked_patch_mse_loss


def run_epoch(
    model,
    loader,
    optimizer=None,
    scheduler=None,
    train=True,
    global_step=0,
    plot_every=500,
    step_history=None,
    ema_alpha=0.03,
):
    model.train(train)

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(config.device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            out = model(x, mask_ratio=config.MASK_RATIO, span_len=config.MASK_SPAN_LEN)
            loss = masked_patch_mse_loss(out["pred_patches"], out["target_patches"], out["mask"])

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                batch_loss = loss.item()

                if step_history is not None:
                    step_history["step"].append(global_step)
                    step_history["batch_loss"].append(batch_loss)

                    if len(step_history["ema_loss"]) == 0:
                        ema = batch_loss
                    else:
                        ema = ema_alpha * batch_loss + (1 - ema_alpha) * step_history["ema_loss"][-1]
                    step_history["ema_loss"].append(ema)

                if step_history is not None and global_step % plot_every == 0:
                    clear_output(wait=True)

                    plt.figure(figsize=(9, 5))
                    plt.plot(step_history["step"], step_history["batch_loss"], alpha=0.20, label="Batch loss")
                    plt.plot(step_history["step"], step_history["ema_loss"], linewidth=2, label="EMA loss")
                    plt.xlabel("Training step")
                    plt.ylabel("Loss")
                    plt.title("Raw Patch Reconstruction Training Monitor")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.show()

            running_loss += loss.item()
            n_batches += 1

            if train and (step % config.LOG_EVERY == 0):
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
