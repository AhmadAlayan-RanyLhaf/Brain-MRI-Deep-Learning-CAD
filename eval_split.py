import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.datasets.miriad import MiriadMRIDataset
from src.models.densenet3d import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="val.csv or test.csv")
    ap.add_argument("--ckpt", type=str, default="runs/baseline3dcnn/best.pt")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)  # keep 0 on Windows to avoid RAM issues
    ap.add_argument("--out_csv", type=str, default="predictions_scan_level.csv")
    args = ap.parse_args()

    # Safer load (avoids pickle objects in future torch versions)
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)

    # Rebuild model with same hyperparams (if present)
    margs = ckpt.get("args", {})
    model_name = margs.get("model", "simple3dcnn")
    model = build_model(
        name=model_name,
        base_ch=int(margs.get("base_ch", 16)),
        dropout=float(margs.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = MiriadMRIDataset(csv_path=args.csv)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    rows = []
    for x, y, meta in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        # meta is a dict of lists/tensors because DataLoader collates it
        for i in range(len(probs)):
            rows.append({
                "path": meta["path"][i],
                "subject_id": int(meta["subject_id"][i]),
                "sex": meta["sex"][i],
                "label": int(y[i]),
                "prob": float(probs[i]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved scan preds -> {args.out_csv}")

    y_true = df["label"].values
    y_prob = df["prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    # Scan-level
    scan_acc = accuracy_score(y_true, y_pred)
    scan_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    print("\nSCAN-LEVEL")
    print(f"  ACC: {scan_acc:.4f}")
    print(f"  AUC: {scan_auc:.4f}")
    print("  CM:\n", confusion_matrix(y_true, y_pred))

    # Subject-level aggregation (mean prob)
    subj = (df.groupby("subject_id")
              .agg(label=("label", "first"), prob=("prob", "mean"))
              .reset_index())

    sy_true = subj["label"].values
    sy_prob = subj["prob"].values
    sy_pred = (sy_prob >= 0.5).astype(int)

    subj_acc = accuracy_score(sy_true, sy_pred)
    subj_auc = roc_auc_score(sy_true, sy_prob) if len(np.unique(sy_true)) == 2 else float("nan")
    print("\nSUBJECT-LEVEL (MEAN PROB)")
    print(f"  ACC: {subj_acc:.4f}")
    print(f"  AUC: {subj_auc:.4f}")
    print("  CM:\n", confusion_matrix(sy_true, sy_pred))

    subj_out = "predictions_subject_level.csv"
    subj.to_csv(subj_out, index=False)
    print(f"\nSaved subject preds -> {subj_out}")

if __name__ == "__main__":
    main()
