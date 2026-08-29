import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.ptbxl import SUPERCLASSES


def _metrics(all_probs, all_labels):
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    aucs = []
    auprcs = []
    class_aucs = {}
    class_auprcs = {}

    for i, sc in enumerate(SUPERCLASSES):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            auprc = average_precision_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
            auprcs.append(auprc)
            class_aucs[sc] = auc
            class_auprcs[sc] = auprc

    return np.mean(aucs), np.mean(auprcs), class_aucs, class_auprcs


def train_one_epoch_fixed(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)   # (B, 5)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_fixed(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    macro_auc, macro_auprc, class_aucs, class_auprcs = _metrics(all_probs, all_labels)
    return total_loss / len(loader), macro_auc, macro_auprc, class_aucs, class_auprcs


def train_one_epoch_beat(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for beats, padding_mask, y in loader:
        beats = beats.to(device)
        padding_mask = padding_mask.to(device)
        y = y.to(device)   # (B, 5)

        optimizer.zero_grad()
        logits = model(beats, padding_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_beat(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for beats, padding_mask, y in loader:
            beats = beats.to(device)
            padding_mask = padding_mask.to(device)
            y = y.to(device)

            logits = model(beats, padding_mask)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    macro_auc, macro_auprc, class_aucs, class_auprcs = _metrics(all_probs, all_labels)
    return total_loss / len(loader), macro_auc, macro_auprc, class_aucs, class_auprcs


def train_one_epoch_beat_hr(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for beats, rr_intervals, padding_mask, y in loader:
        beats = beats.to(device)
        rr_intervals = rr_intervals.to(device)
        padding_mask = padding_mask.to(device)
        y = y.to(device)   # (B, 5)

        optimizer.zero_grad()
        logits = model(beats, rr_intervals, padding_mask)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_beat_hr(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for beats, rr_intervals, padding_mask, y in loader:
            beats = beats.to(device)
            rr_intervals = rr_intervals.to(device)
            padding_mask = padding_mask.to(device)
            y = y.to(device)

            logits = model(beats, rr_intervals, padding_mask)
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    macro_auc, macro_auprc, class_aucs, class_auprcs = _metrics(all_probs, all_labels)
    return total_loss / len(loader), macro_auc, macro_auprc, class_aucs, class_auprcs
