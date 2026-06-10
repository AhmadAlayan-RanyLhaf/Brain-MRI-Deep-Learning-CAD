# train_fold0_best_acc.py
import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from model import build_model
from dataset import MiriadMRIDataset


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.total += float(val) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate_subject_level(model, loader, device):
    model.eval()
    scan_rows = []
    losses = AverageMeter()
    crit = nn.BCEWithLogitsLoss()

    for x, y, meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x)
        loss = crit(logits, y)
        losses.update(loss.item(), n=x.size(0))

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        y_np = y.cpu().numpy().reshape(-1)
        subj_ids = meta["subject_id"]

        for i in range(len(probs)):
            sid = subj_ids[i].item() if hasattr(subj_ids[i], "item") else int(subj_ids[i])
            scan_rows.append({
                "subject_id": sid,
                "label": int(y_np[i]),
                "prob_scan": float(probs[i])
            })

    df = pd.DataFrame(scan_rows)
    subj_df = df.groupby("subject_id").agg(
        label=("label", "first"),
        prob_subject=("prob_scan", "mean")
    ).reset_index()

    y_true = subj_df["label"].values
    y_prob = subj_df["prob_subject"].values

    subj_acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    try:
        subj_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        subj_auc = float("nan")

    return losses.avg, subj_acc, subj_auc, subj_df


def train_one_epoch(model, loader, optimizer, scaler, device, crit):
    model.train()
    losses = AverageMeter()

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = crit(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), n=x.size(0))

    return losses.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="folds/fold0_train.csv")
    ap.add_argument("--val_csv", type=str, default="folds/fold0_val.csv")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", type=str, default="densenet3d121")
    ap.add_argument("--out_dir", type=str, default="runs/cv_fold0")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MiriadMRIDataset(csv_path=args.train_csv, is_train=True)
    val_ds = MiriadMRIDataset(csv_path=args.val_csv, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args.model).to(device)
    crit = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_subj_acc = 0.0
    best_val_loss = float("inf")
    best_path = out_dir / "best.pt"

    # Optional: log config
    (out_dir / "config.txt").write_text(
        "\n".join([f"{k}={v}" for k, v in vars(args).items()]) + f"\ndevice={device}\npos_weight=None\n"
    )

    print(f"Starting training on device: {device} | seed: {args.seed}")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, crit)
        va_loss, va_acc, va_auc, subj_df = evaluate_subject_level(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_subj_acc={va_acc*100:.2f}% | val_subj_auc={va_auc:.4f}"
        )

        # We want to maximize validation subject-level accuracy.
        # Tie-breaker: lower validation loss.
        is_best = False
        if va_acc > best_subj_acc:
            is_best = True
        elif va_acc == best_subj_acc:
            if va_loss < best_val_loss:
                is_best = True

        if is_best:
            best_subj_acc = va_acc
            best_val_loss = va_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_val_loss,
                    "best_subj_acc": best_subj_acc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  [SUCCESS] Saved best checkpoint: {best_path} (val_subj_acc={best_subj_acc*100:.2f}%, val_loss={best_val_loss:.4f})")
            if best_subj_acc == 1.0:
                print(f"  [PERFECT] Reached 100% validation subject-level accuracy at Epoch {epoch}!")

    print(f"Done. Best validation subject-level accuracy: {best_subj_acc*100:.2f}% | checkpoint: {best_path}")


if __name__ == "__main__":
    main()
