# evaluate_fold.py
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="folds/fold0_val.csv")
    parser.add_argument("--ckpt", default="temp_cv_fold0/best.pt")
    parser.add_argument("--model", default="densenet3d121")
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return
        
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    model = build_model(args.model)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    ds = MiriadMRIDataset(args.csv)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    
    scan_rows = []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            y_np = y.cpu().numpy().reshape(-1)
            subj_ids = meta["subject_id"]
            
            for i in range(len(probs)):
                scan_rows.append({
                    "subject_id": int(subj_ids[i]),
                    "label": int(y_np[i]),
                    "prob_scan": float(probs[i])
                })
                
    df = pd.DataFrame(scan_rows)
    subj_df = df.groupby("subject_id").agg(
        label=("label", "first"),
        mean_p=("prob_scan", "mean"),
        median_p=("prob_scan", "median"),
        max_p=("prob_scan", "max"),
        min_p=("prob_scan", "min")
    ).reset_index()
    
    # Calculate ACC at T=0.5 for different aggregations
    print(f"Results for model: {args.ckpt}")
    for agg in ["mean_p", "median_p", "max_p", "min_p"]:
        y_true = subj_df["label"].values
        y_prob = subj_df[agg].values
        acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
        auc = roc_auc_score(y_true, y_prob)
        print(f"  Aggregation: {agg:10s} | AUC: {auc:.4f} | ACC (T=0.5): {acc*100:.2f}%")
        
    # Print individual subjects
    print("\nSubject predictions:")
    print(subj_df.to_string(index=False))

if __name__ == "__main__":
    main()
