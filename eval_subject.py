# eval_subject.py
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import MiriadMRIDataset
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV to evaluate (e.g., folds/fold0_val.csv)")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (e.g., runs/cv_fold0/best.pt)")
    ap.add_argument("--out_csv", required=True, help="Output subject-level CSV path")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="simple3dcnn", help="simple3dcnn | densenet3d121")
    args = ap.parse_args()

    # ---- Load model ----
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model = build_model(args.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    # ---- Data ----
    ds = MiriadMRIDataset(args.csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    scan_rows = []

    for x, y, meta in loader:
        x = x.to(DEVICE)
        logits = model(x).detach()
        probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)  # always 1D

        y_np = y.cpu().numpy().reshape(-1)

        # meta fields are lists/tensors; handle both safely
        subj_ids = meta["subject_id"]
        paths = meta["path"]

        for i in range(len(probs)):
            sid = subj_ids[i].item() if hasattr(subj_ids[i], "item") else int(subj_ids[i])
            pth = paths[i]
            scan_rows.append({
                "path": pth,
                "subject_id": sid,
                "label": int(y_np[i]),
                "prob_scan": float(probs[i]),
            })

    scan_df = pd.DataFrame(scan_rows)

    # ---- Scan-level metrics ----
    scan_auc = roc_auc_score(scan_df["label"], scan_df["prob_scan"])
    scan_acc = accuracy_score(scan_df["label"], (scan_df["prob_scan"] >= args.threshold).astype(int))

    # ---- Subject-level aggregation ----
    subj_df = (
        scan_df.groupby("subject_id")
        .agg(label=("label", "first"), prob_subject=("prob_scan", "mean"))
        .reset_index()
    )

    subj_auc = roc_auc_score(subj_df["label"], subj_df["prob_subject"])
    subj_acc = accuracy_score(subj_df["label"], (subj_df["prob_subject"] >= args.threshold).astype(int))

    print("\nSCAN-LEVEL")
    print("  AUC:", round(scan_auc, 4))
    print("  ACC:", round(scan_acc, 4))

    print("\nSUBJECT-LEVEL (MEAN PROB)")
    print("  AUC:", round(subj_auc, 4))
    print("  ACC:", round(subj_acc, 4))

    # ---- Save ----
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    subj_df.to_csv(args.out_csv, index=False)

    scan_out = os.path.splitext(args.out_csv)[0] + "_scan.csv"
    scan_df.to_csv(scan_out, index=False)

    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {scan_out}")


if __name__ == "__main__":
    main()
