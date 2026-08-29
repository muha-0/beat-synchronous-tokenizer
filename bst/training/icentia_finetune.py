import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


def train_one_epoch_fixed(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_one_epoch_beat(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for beats, mask, y in loader:
        beats, mask, y = beats.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(beats, mask), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_one_epoch_beat_hr(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for beats, rr, mask, y in loader:
        beats, rr, mask, y = beats.to(device), rr.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(beats, rr, mask), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_fixed(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probs = torch.softmax(model(x), dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return {
        "AUROC": roc_auc_score(all_labels, all_probs),
        "AUPRC": average_precision_score(all_labels, all_probs),
    }


@torch.no_grad()
def evaluate_beat(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for beats, mask, y in loader:
        beats, mask, y = beats.to(device), mask.to(device), y.to(device)
        probs = torch.softmax(model(beats, mask), dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return {
        "AUROC": roc_auc_score(all_labels, all_probs),
        "AUPRC": average_precision_score(all_labels, all_probs),
    }


@torch.no_grad()
def evaluate_beat_hr(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for beats, rr, mask, y in loader:
        beats, rr, mask, y = beats.to(device), rr.to(device), mask.to(device), y.to(device)
        probs = torch.softmax(model(beats, rr, mask), dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    return {
        "AUROC": roc_auc_score(all_labels, all_probs),
        "AUPRC": average_precision_score(all_labels, all_probs),
    }


def run_training(model, train_loader, val_loader, test_loader,
                 train_fn, eval_fn, ckpt_dir, device,
                 num_epochs=10, lr_enc=1e-4, lr_head=1e-3):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_enc},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ], weight_decay=1e-4)

    best_auprc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        train_loader.dataset.epoch = epoch

        train_loss = train_fn(model, train_loader, optimizer, device)
        val_metrics = eval_fn(model, val_loader, device)

        print(f"Epoch {epoch+1:02d} | loss={train_loss:.4f} | "
              f"val AUROC={val_metrics['AUROC']:.4f}  AUPRC={val_metrics['AUPRC']:.4f}")

        if val_metrics["AUPRC"] > best_auprc:
            best_auprc = val_metrics["AUPRC"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, f"{ckpt_dir}/best.pt")
            print(f"  -> new best (AUPRC={best_auprc:.4f})")

    model.load_state_dict(best_state)
    test_metrics = eval_fn(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"TEST  AUROC={test_metrics['AUROC']:.4f}  AUPRC={test_metrics['AUPRC']:.4f}")
    print(f"{'='*50}")
    return test_metrics
