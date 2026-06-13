# eval_oasis.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Add workspace to path to allow importing from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.densenet3d import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OasisDataset(Dataset):
    def __init__(self, csv_path="oasis_index.csv"):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row["path"])
        label = int(row["label"])
        subject_id = int(row["subject_id"])
        sex = str(row["sex"])
        
        # Load preprocessed .npy volume
        vol = np.load(path)
        
        # Convert to torch tensor [1, D, H, W]
        x = torch.from_numpy(vol).unsqueeze(0).float()
        y = torch.tensor(label, dtype=torch.long)
        
        meta = {"path": path, "subject_id": subject_id, "sex": sex}
        return x, y, meta

@torch.no_grad()
def evaluate_fold(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    for x, y, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
        
    return np.array(all_probs), np.array(all_labels)

def main():
    index_csv = "oasis_index.csv"
    if not os.path.exists(index_csv):
        print(f"ERROR: {index_csv} not found. Run preprocess_oasis.py first!")
        sys.exit(1)
        
    ds = OasisDataset(index_csv)
    if len(ds) == 0:
        print("ERROR: No subjects found in oasis_index.csv.")
        sys.exit(1)
        
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Loaded {len(ds)} OASIS-1 subjects for validation.")
    
    # Store probability predictions from each of the 5 folds
    fold_probs = {}
    labels = None
    
    print("\n--- Running Inference Across Folds ---")
    for fold in range(5):
        ckpt_path = f"runs/cv_fold{fold}/best.pt"
        if not os.path.exists(ckpt_path):
            print(f"Skipping Fold {fold} (checkpoint not found at {ckpt_path})")
            continue
            
        print(f"Evaluating Fold {fold} using {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        
        margs = ckpt.get("args", {})
        model_name = margs.get("model", "densenet121")
        model = build_model(
            name=model_name,
            base_ch=int(margs.get("base_ch", 16)),
            dropout=float(margs.get("dropout", 0.2)),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        
        probs, fold_labels = evaluate_fold(model, loader)
        fold_probs[fold] = probs
        
        if labels is None:
            labels = fold_labels
            
        # Calculate fold-specific metrics
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        print(f"  Fold {fold} Performance -> ACC: {acc:.4f} | AUC: {auc:.4f}")

    if not fold_probs:
        print("ERROR: No fold checkpoints could be loaded.")
        sys.exit(1)

    # 4. Ensemble Metrics (Average Probabilities across all loaded folds)
    all_fold_matrices = np.array(list(fold_probs.values()))  # [num_folds, num_subjects]
    ensemble_probs = np.mean(all_fold_matrices, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    
    ensemble_acc = accuracy_score(labels, ensemble_preds)
    try:
        ensemble_auc = roc_auc_score(labels, ensemble_probs)
    except ValueError:
        ensemble_auc = float("nan")
        
    print("\n========================================")
    print("      ENSEMBLE VALIDATION SUMMARY")
    print("========================================")
    print(f"Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"Ensemble AUC:      {ensemble_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, ensemble_preds))
    print("========================================")
    
    # Save predictions
    df_results = pd.read_csv(index_csv)
    df_results["ensemble_prob"] = ensemble_probs
    df_results["ensemble_pred"] = ensemble_preds
    for fold, probs in fold_probs.items():
        df_results[f"fold{fold}_prob"] = probs
        
    out_csv = "oasis_predictions.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nDetailed predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
