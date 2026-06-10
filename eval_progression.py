# eval_progression.py
import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset_progression import MiriadProgressionDataset
from model_progression import build_progression_model


def metas_to_list(meta_batch):
    """
    Convert DataLoader-collated metadata into list[dict].
    Handles dict-of-lists / dict-of-tensors.
    """
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="prog3dcnn")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="progression_preds.csv")
    ap.add_argument("--target_size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset / loader
    ds = MiriadProgressionDataset(
        args.csv,
        target_size=args.target_size,
        augment=False
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    model = build_progression_model(args.model).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []

    for x, _y, meta in loader:
        x = x.to(device)
        preds = model(x).detach().cpu().numpy()

        metas = metas_to_list(meta)

        for i in range(len(preds)):
            row = metas[i].copy()
            row["stage_pred"] = float(preds[i])
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    print("Saved:", args.out_csv)
    print("Rows:", len(out_df))
    print("Columns:", list(out_df.columns))


if __name__ == "__main__":
    main()
