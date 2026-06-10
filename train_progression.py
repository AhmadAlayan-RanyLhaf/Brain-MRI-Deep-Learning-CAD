# train_progression.py
import os
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from src.datasets.progression import MiriadProgressionDataset
from src.models.progression import build_progression_model


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_split(csv_path: str, val_frac: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        raise ValueError("For automatic split, CSV must contain 'subject_id'.")
    subjects = sorted(df["subject_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_frac))
    val_subj = set(subjects[:n_val])
    train_df = df[~df["subject_id"].isin(val_subj)].copy()
    val_df = df[df["subject_id"].isin(val_subj)].copy()
    return train_df, val_df


def metas_to_list(meta_batch):
    # DataLoader may collate meta into dict-of-lists/tensors; standardize to list-of-dicts
    if isinstance(meta_batch, list):
        return meta_batch
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        first = meta_batch[keys[0]]
        B = len(first)
        out = []
        for i in range(B):
            d = {}
            for k in keys:
                v = meta_batch[k][i]
                if torch.is_tensor(v):
                    v = v.item()
                d[k] = v
            out.append(d)
        return out
    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")


def pairwise_ranking_loss(preds: torch.Tensor, meta_batch, margin: float = 0.05) -> torch.Tensor:
    """
    Enforce: within a subject, later visits should have higher predicted stage.

    preds: [B] in [0,1]
    meta_batch: list[dict] or dict[...] from DataLoader
    """
    metas = metas_to_list(meta_batch)
    device = preds.device
    B = preds.shape[0]
    loss_terms = []

    by_subj = {}
    for i in range(B):
        sid = metas[i].get("subject_id", None)
        v = metas[i].get("visit", None)
        if sid is None or v is None:
            continue
        try:
            v = int(v)
        except Exception:
            continue
        by_subj.setdefault(sid, []).append((v, i))

    for _sid, items in by_subj.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda x: x[0])  # visit order
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                v1, i1 = items[a]
                v2, i2 = items[b]
                if v2 <= v1:
                    continue
                diff = preds[i2] - preds[i1]  # late - early
                loss_terms.append(F.relu(margin - diff))

    if len(loss_terms) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_terms).mean()


class SubjectBatchSampler(Sampler):
    """
    Makes batches that contain multiple visits per subject, so ranking loss is active.

    We sample:
      - subjects_per_batch subjects
      - visits_per_subject indices per subject
    Batch size = subjects_per_batch * visits_per_subject

    Requires dataset.df to contain: subject_id (and preferably visit).
    """
    def __init__(
        self,
        dataset: MiriadProgressionDataset,
        subjects_per_batch: int = 2,
        visits_per_subject: int = 2,
        seed: int = 42,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.subjects_per_batch = int(subjects_per_batch)
        self.visits_per_subject = int(visits_per_subject)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        df = dataset.df
        if "subject_id" not in df.columns:
            raise ValueError("SubjectBatchSampler requires 'subject_id' in the dataset CSV.")
        # map subject -> list of row indices
        self.subj_to_indices = {}
        for idx, sid in enumerate(df["subject_id"].tolist()):
            self.subj_to_indices.setdefault(sid, []).append(idx)

        self.subjects = list(self.subj_to_indices.keys())

        # Number of batches per epoch (approx)
        # each batch uses subjects_per_batch subjects
        self.num_batches = len(self.subjects) // self.subjects_per_batch
        if not self.drop_last and (len(self.subjects) % self.subjects_per_batch) != 0:
            self.num_batches += 1

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + int(time.time()))  # change each epoch naturally
        subjects = self.subjects[:]
        rng.shuffle(subjects)

        # chunk subjects into batches
        for b in range(self.num_batches):
            start = b * self.subjects_per_batch
            end = start + self.subjects_per_batch
            batch_subjects = subjects[start:end]
            if len(batch_subjects) < self.subjects_per_batch:
                if self.drop_last:
                    break

            batch_indices = []
            for sid in batch_subjects:
                inds = self.subj_to_indices[sid]
                if len(inds) >= self.visits_per_subject:
                    chosen = rng.sample(inds, self.visits_per_subject)
                else:
                    # if subject has fewer visits than needed, sample with replacement
                    chosen = [rng.choice(inds) for _ in range(self.visits_per_subject)]
                batch_indices.extend(chosen)

            yield batch_indices


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y, _meta in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        preds.append(p.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))

    try:
        import scipy.stats
        spearman = float(scipy.stats.spearmanr(trues, preds).correlation)
    except Exception:
        spearman = float("nan")

    return {"mae": mae, "rmse": rmse, "spearman": spearman}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_csv", type=str, default="miriad_ad_progression.csv")
    ap.add_argument("--val_csv", type=str, default="")
    ap.add_argument("--model", type=str, default="prog3dcnn",
                    help="prog3dcnn | densenet3d121 | efficientnet_b0_2p5d")

    ap.add_argument("--target_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--amp", action="store_true")

    # Ranking controls
    ap.add_argument("--rank_lambda", type=float, default=0.3)
    ap.add_argument("--rank_margin", type=float, default=0.05)

    # Subject-aware batching controls (default makes batch size = 2*2 = 4)
    ap.add_argument("--subjects_per_batch", type=int, default=2)
    ap.add_argument("--visits_per_subject", type=int, default=2)

    args = ap.parse_args()

    # Batch size implied:
    batch_size = args.subjects_per_batch * args.visits_per_subject

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = (
        f"progression_{args.model}_S{args.subjects_per_batch}xV{args.visits_per_subject}"
        f"_rank{args.rank_lambda}_m{args.rank_margin}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # subject-wise train/val split
    if args.val_csv.strip():
        train_csv_path = args.train_csv
        val_csv_path = args.val_csv
    else:
        train_df, val_df = subject_split(args.train_csv, val_frac=args.val_frac, seed=args.seed)
        train_csv_path = os.path.join(out_dir, "train_split.csv")
        val_csv_path = os.path.join(out_dir, "val_split.csv")
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

    train_ds = MiriadProgressionDataset(train_csv_path, target_size=args.target_size, augment=args.augment)
    val_ds = MiriadProgressionDataset(val_csv_path, target_size=args.target_size, augment=False)

    # Subject-aware batch sampler
    batch_sampler = SubjectBatchSampler(
        train_ds,
        subjects_per_batch=args.subjects_per_batch,
        visits_per_subject=args.visits_per_subject,
        seed=args.seed,
        drop_last=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_progression_model(args.model).to(device)

    criterion = nn.SmoothL1Loss(beta=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # save config
    config = vars(args).copy()
    config["batch_size_effective"] = batch_size
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(out_dir, "metrics.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_reg_loss,train_rank_loss,val_mae,val_rmse,val_spearman\n")

    best_mae = float("inf")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses_total = []
        losses_reg = []
        losses_rank = []

        for x, y, meta in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(x)  # [B] in [0,1]
                loss_reg = criterion(pred, y)
                loss_rank = pairwise_ranking_loss(pred, meta, margin=args.rank_margin)
                loss = loss_reg + args.rank_lambda * loss_rank

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses_total.append(loss.detach().cpu().item())
            losses_reg.append(loss_reg.detach().cpu().item())
            losses_rank.append(loss_rank.detach().cpu().item())

        train_loss = float(np.mean(losses_total)) if losses_total else float("nan")
        train_reg = float(np.mean(losses_reg)) if losses_reg else float("nan")
        train_rank = float(np.mean(losses_rank)) if losses_rank else float("nan")

        metrics = evaluate(model, val_loader, device)

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_reg:.6f},{train_rank:.6f},"
                f"{metrics['mae']:.6f},{metrics['rmse']:.6f},{metrics['spearman']}\n"
            )

        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, last_path)

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, best_path)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} (reg={train_reg:.4f}, rank={train_rank:.4f}) "
            f"val_mae={metrics['mae']:.4f} val_rmse={metrics['rmse']:.4f} spearman={metrics['spearman']}"
        )

    print(f"\nDone. Best MAE={best_mae:.4f}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
