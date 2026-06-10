# train.py
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

from src.models.densenet3d import build_model

# IMPORTANT: adjust this import to your actual dataset location/name
from src.datasets.miriad import MiriadMRIDataset  # must return (x, y, meta)


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
    # Determinism can slow you down; keep it simple for now.
    torch.backends.cudnn.benchmark = True


def compute_pos_weight(train_csv: str) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    # label: 1=AD (positive), 0=HC (negative)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    # BCEWithLogitsLoss pos_weight = neg/pos
    if pos == 0:
        raise ValueError("No positive samples found in train CSV.")
    return torch.tensor([neg / pos], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    losses = AverageMeter()

    # loss is only for reporting; use plain unweighted here (metrics matter more)
    crit = nn.BCEWithLogitsLoss()

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        logits = model(x)
        loss = crit(logits, y)
        losses.update(loss.item(), n=x.size(0))

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_y).astype(np.int64)
    y_prob = np.concatenate(all_probs)

    # Accuracy at 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)

    # ROC-AUC (handle edge case if a class is missing in val)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return losses.avg, acc, auc


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
    ap.add_argument("--train_csv", type=str, default="train.csv")
    ap.add_argument("--val_csv", type=str, default="val.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_ch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--model", type=str, default="simple3dcnn", help="simple3dcnn | densenet3d121")
    ap.add_argument("--out_dir", type=str, default="runs/baseline3dcnn")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets / Loaders (adjust constructor if needed) ----
    train_ds = MiriadMRIDataset(csv_path=args.train_csv, is_train=True)
    val_ds = MiriadMRIDataset(csv_path=args.val_csv, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model ----
    model = build_model(args.model, base_ch=args.base_ch, dropout=args.dropout).to(device)

    # ---- Imbalance handling ----
    # Set pos_weight to None to avoid downward probability shift and keep threshold centered at 0.5
    pos_weight = None
    crit = nn.BCEWithLogitsLoss()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- AMP ----
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = float("inf")
    best_path = out_dir / "best.pt"

    # Optional: log config
    (out_dir / "config.txt").write_text(
        "\n".join([f"{k}={v}" for k, v in vars(args).items()]) + f"\ndevice={device}\npos_weight=None\n"
    )

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, crit)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_auc={va_auc:.4f}"
        )

        # Save best by validation loss (minimizing loss centers probabilities around 0.5)
        if va_loss < best_loss:
            best_loss = va_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_loss,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  [SUCCESS] Saved best checkpoint: {best_path} (val_loss={best_loss:.4f})")

    print(f"Done. Best val_loss: {best_loss:.4f} | checkpoint: {best_path}")


if __name__ == "__main__":
    main()
