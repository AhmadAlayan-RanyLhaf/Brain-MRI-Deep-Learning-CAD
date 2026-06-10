# calc_all_model_metrics.py
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Import models
from segmentation_model import UNet3D
from segmentation_dataset import _center_crop_or_pad_3d

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Helper to compute classification/segmentation metrics for a class
def compute_class_metrics(pred_bin: np.ndarray, pred_prob: np.ndarray, gt_bin: np.ndarray):
    # Flatten
    y_true = gt_bin.flatten()
    y_pred = pred_bin.flatten()
    y_prob = pred_prob.flatten()
    
    # Check if target is all background or empty
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        
    # Standard metrics
    # Dice
    inter = np.sum(y_pred & y_true)
    union = np.sum(y_pred) + np.sum(y_true)
    dice = (2.0 * inter) / union if union > 0 else 0.0
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Sensitivity / Recall
    sens = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity (TN / (TN + FP))
    tn = np.sum((~y_pred) & (~y_true))
    fp = np.sum(y_pred & (~y_true))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    # F1-Score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUROC
    try:
        # Check if true labels have both classes
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 1.0 if np.all(y_pred == y_true) else 0.5
    except Exception as e:
        auc = 0.5
        
    return dice, acc, sens, spec, prec, f1, auc

def main():
    raw_path = "segmentation_data/images/sub-188_img.nii.gz"
    gt_path = "segmentation_data/labels/sub-188_lbl.nii.gz"
    
    # 1. Load volumes
    print("Loading test NIfTI volumes...")
    orig_vol = nib.load(raw_path).get_fdata(dtype=np.float32)
    orig_gt = nib.load(gt_path).get_fdata(dtype=np.float32).astype(np.uint8)
    
    target_size = 128
    vol_cropped = _center_crop_or_pad_3d(orig_vol, target_size)
    gt_cropped = _center_crop_or_pad_3d(orig_gt, target_size)
    
    # Normalise raw crop for model inputs
    fg_mask = vol_cropped > 0
    vol_normalized = vol_cropped.copy()
    if fg_mask.sum() > 0:
        mean = vol_cropped[fg_mask].mean()
        std = vol_cropped[fg_mask].std() + 1e-8
        vol_normalized = (vol_cropped - mean) / std
        vol_normalized[~fg_mask] = 0.0
    
    # Prepare dictionaries for predicted volumes and probabilities
    # Shape: (4, 128, 128, 128) for background + 3 classes
    predictions = {
        "3D U-Net + TTA": {"probs": np.zeros((4, target_size, target_size, target_size)), "labels": np.zeros_like(gt_cropped)},
        "3D U-Net (No TTA)": {"probs": np.zeros((4, target_size, target_size, target_size)), "labels": np.zeros_like(gt_cropped)}
    }
    
    # 2. RUN 3D U-NET (TTA vs No TTA)
    print("Running 3D U-Net inference...")
    unet_ckpt = "runs/segmentation/best.pt"
    if os.path.exists(unet_ckpt):
        model_unet = UNet3D(in_channels=1, out_channels=4).to(device)
        ckpt = torch.load(unet_ckpt, map_location=device)
        model_unet.load_state_dict(ckpt["model_state"])
        model_unet.eval()
        
        x_tensor = torch.from_numpy(vol_normalized).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model_unet(x_tensor)
            probs = F.softmax(logits, dim=1)
            
            # 4-way TTA
            x_flip_d = torch.flip(x_tensor, dims=[2])
            probs_flip_d = torch.flip(F.softmax(model_unet(x_flip_d), dim=1), dims=[2])
            x_flip_h = torch.flip(x_tensor, dims=[3])
            probs_flip_h = torch.flip(F.softmax(model_unet(x_flip_h), dim=1), dims=[3])
            x_flip_w = torch.flip(x_tensor, dims=[4])
            probs_flip_w = torch.flip(F.softmax(model_unet(x_flip_w), dim=1), dims=[4])
            
            avg_probs = (probs + probs_flip_d + probs_flip_h + probs_flip_w) / 4.0
            
            # Save No TTA predictions
            predictions["3D U-Net (No TTA)"]["probs"] = probs.squeeze(0).cpu().numpy()
            predictions["3D U-Net (No TTA)"]["labels"] = np.argmax(predictions["3D U-Net (No TTA)"]["probs"], axis=0).astype(np.uint8)

            # Save TTA predictions
            predictions["3D U-Net + TTA"]["probs"] = avg_probs.squeeze(0).cpu().numpy()
            predictions["3D U-Net + TTA"]["labels"] = np.argmax(predictions["3D U-Net + TTA"]["probs"], axis=0).astype(np.uint8)
        print("3D U-Net inference complete.")
    else:
        print("U-Net checkpoint not found!")

    # 3. COMPUTE METRICS FOR ALL MODELS
    print("\nCalculating metrics across models...")
    classes = {1: "CSF", 2: "Gray Matter", 3: "White Matter"}
    
    rows = []
    for model_name, data in predictions.items():
        if np.sum(data["labels"]) == 0:
            print(f"Skipping {model_name} because its predicted label volume is empty.")
            continue
            
        print(f"\nComputing metrics for: {model_name}")
        class_dices = []
        class_aucs = []
        
        for class_id, class_name in classes.items():
            gt_bin = (gt_cropped == class_id)
            pred_bin = (data["labels"] == class_id)
            pred_prob = data["probs"][class_id]
            
            dice, acc, sens, spec, prec, f1, auc = compute_class_metrics(pred_bin, pred_prob, gt_bin)
            
            class_dices.append(dice)
            class_aucs.append(auc)
            
            rows.append({
                "Model": model_name,
                "Class": class_name,
                "Dice": dice,
                "Accuracy": acc,
                "Sensitivity": sens,
                "Specificity": spec,
                "Precision": prec,
                "F1": f1,
                "AUROC": auc
            })
            print(f"  {class_name:15s} | Dice: {dice:.4f} | AUROC: {auc:.4f}")
            
        mean_dice = np.mean(class_dices)
        mean_auc = np.mean(class_aucs)
        
        # Mean Foreground row
        rows.append({
            "Model": model_name,
            "Class": "Mean Foreground",
            "Dice": mean_dice,
            "Accuracy": np.nan,
            "Sensitivity": np.nan,
            "Specificity": np.nan,
            "Precision": np.nan,
            "F1": np.nan,
            "AUROC": mean_auc
        })
        print(f"  {'Mean Foreground':15s} | Dice: {mean_dice:.4f} | AUROC: {mean_auc:.4f}")
        
    # Create final DataFrame
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("results/segmentation_comparison_metrics.csv", index=False)
    print("\nSaved metrics to results/segmentation_comparison_metrics.csv!")
    print(df_metrics[df_metrics["Class"] == "Mean Foreground"][["Model", "Dice", "AUROC"]].to_string(index=False))

if __name__ == "__main__":
    main()
