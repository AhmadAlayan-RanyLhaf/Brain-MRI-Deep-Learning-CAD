# aggregate_cv.py
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

THRESH = 0.5

rows = []
paths = sorted(glob.glob("runs/cv_fold*/val_subject_predictions.csv"))

if not paths:
    raise SystemExit("No files found: runs/cv_fold*/val_subject_predictions.csv")

for p in paths:
    df = pd.read_csv(p)
    # expects: subject_id, label, prob_subject
    auc = roc_auc_score(df["label"], df["prob_subject"])
    acc = accuracy_score(df["label"], (df["prob_subject"] >= THRESH).astype(int))
    fold = p.split("cv_fold")[-1].split("/")[0].split("\\")[0]
    rows.append({"fold": fold, "AUC": auc, "ACC": acc, "file": p})

res = pd.DataFrame(rows).sort_values("fold")
print("\nPer-fold results:")
print(res[["fold", "AUC", "ACC"]].to_string(index=False))

print("\nMean ± Std (subject-level):")
print(f"AUC: {res['AUC'].mean():.4f} ± {res['AUC'].std(ddof=1):.4f}")
print(f"ACC: {res['ACC'].mean():.4f} ± {res['ACC'].std(ddof=1):.4f}")

res.to_csv("runs/cv_summary.csv", index=False)
print("\nSaved: runs/cv_summary.csv")
