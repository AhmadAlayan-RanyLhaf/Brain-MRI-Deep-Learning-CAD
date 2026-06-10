# segmentation_eval.py
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

from segmentation_model import UNet3D
from segmentation_dataset import _center_crop_or_pad_3d

# Helper to compute Dice and IoU
def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = 4):
    metrics = {}
    for c in range(1, num_classes):
        p_mask = (pred == c)
        t_mask = (target == c)
        
        inter = np.sum(p_mask & t_mask)
        union = np.sum(p_mask) + np.sum(t_mask)
        
        if union == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = (2.0 * inter) / union
            iou = inter / (union - inter)
            
        metrics[c] = {"dice": dice, "iou": iou}
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Evaluate Subject 188 (unseen test subject)
    raw_path = "segmentation_data/images/sub-188_img.nii.gz"
    gt_path = "segmentation_data/labels/sub-188_lbl.nii.gz"
    ckpt_path = "runs/segmentation/best.pt"
    
    if not os.path.exists(raw_path) or not os.path.exists(gt_path):
        print("Missing Subject 188 files!")
        return
        
    if not os.path.exists(ckpt_path):
        print(f"Missing U-Net checkpoint: {ckpt_path}")
        return
        
    print(f"Loading test volume: {raw_path}")
    img = nib.load(raw_path)
    orig_vol = img.get_fdata(dtype=np.float32)
    
    print(f"Loading ground truth: {gt_path}")
    lbl_img = nib.load(gt_path)
    gt_vol = lbl_img.get_fdata(dtype=np.float32).astype(np.uint8)
    
    # 2. Preprocess: Crop/Pad and Normalise
    target_size = 128
    vol = _center_crop_or_pad_3d(orig_vol, target_size)
    gt = _center_crop_or_pad_3d(gt_vol, target_size)
    
    # Z-score Normalization (foreground-only to match dataset)
    fg_mask = vol > 0
    if fg_mask.sum() > 0:
        mean = vol[fg_mask].mean()
        std = vol[fg_mask].std() + 1e-8
        vol = (vol - mean) / std
        vol[~fg_mask] = 0.0
    else:
        mean = vol.mean()
        std = vol.std() + 1e-8
        vol = (vol - mean) / std
        
    # To Tensor: [1, 1, D, H, W]
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 3. Load 3D U-Net Model
    print(f"Loading 3D U-Net checkpoint: {ckpt_path}")
    model = UNet3D(in_channels=1, out_channels=4)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    # 4. Predict with Test-Time Augmentation (TTA)
    print("Running evaluation with 4-way Test-Time Augmentation (TTA)...")
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        
        # Flip along depth (dim 2)
        x_flip_d = torch.flip(x, dims=[2])
        probs_flip_d = torch.flip(F.softmax(model(x_flip_d), dim=1), dims=[2])
        
        # Flip along height (dim 3)
        x_flip_h = torch.flip(x, dims=[3])
        probs_flip_h = torch.flip(F.softmax(model(x_flip_h), dim=1), dims=[3])
        
        # Flip along width (dim 4)
        x_flip_w = torch.flip(x, dims=[4])
        probs_flip_w = torch.flip(F.softmax(model(x_flip_w), dim=1), dims=[4])
        
        # Average probability maps
        avg_probs = (probs + probs_flip_d + probs_flip_h + probs_flip_w) / 4.0
        pred_vol = torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        
    # Save predicted test volume for visualization script
    test_pred_path = "test_pred_seg_sub188.nii.gz"
    
    # Map prediction back to original space using the reverse method
    from segmentation_infer import reverse_center_crop_or_pad_3d
    orig_pred_mask = reverse_center_crop_or_pad_3d(pred_vol, orig_vol.shape)
    out_img = nib.Nifti1Image(orig_pred_mask, img.affine, img.header)
    nib.save(out_img, test_pred_path)
    print(f"Saved predicted volume back to original shape: {test_pred_path}")
    
    # Compute metrics on 128^3 cropped space
    unet_metrics = compute_metrics(pred_vol, gt)
    
    # Evaluate on comparison slice 117
    unet_slice_metrics = compute_metrics(pred_vol[117, :, :], gt[117, :, :])
    
    print("\n" + "="*50)
    print("         3D U-NET PERFORMANCE REPORT (TEST SUB-188)         ")
    print("="*50)
    print(f"Global 3D Volume Metrics:")
    classes = {1: "CSF", 2: "Gray Matter", 3: "White Matter"}
    for c, name in classes.items():
        print(f"  {name:15s} | Dice: {unet_metrics[c]['dice']:.4f} | IoU: {unet_metrics[c]['iou']:.4f}")
    mean_dice = np.mean([unet_metrics[c]["dice"] for c in range(1, 4)])
    mean_iou = np.mean([unet_metrics[c]["iou"] for c in range(1, 4)])
    print("-" * 50)
    print(f"  {'Mean Foreground':15s} | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")
    print("="*50)
    
    # Write metrics to CSV and JSON
    results_rows = []
    results_dict = {}
    for c, name in classes.items():
        class_key = name.lower().replace(" ", "_")
        results_rows.append({
            "class_id": c,
            "class_name": name,
            "dice": unet_metrics[c]["dice"],
            "iou": unet_metrics[c]["iou"],
            "slice_117_dice": unet_slice_metrics[c]["dice"]
        })
        results_dict[class_key] = {
            "dice": unet_metrics[c]["dice"],
            "iou": unet_metrics[c]["iou"],
            "slice_117_dice": unet_slice_metrics[c]["dice"]
        }
    results_dict["mean_foreground"] = {
        "dice": mean_dice, 
        "iou": mean_iou,
        "slice_117_dice": np.mean([unet_slice_metrics[c]["dice"] for c in range(1, 4)])
    }
    
    # Save files
    pd.DataFrame(results_rows).to_csv("results/segmentation_results.csv", index=False)
    with open("results/segmentation_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    print("Saved metrics to results/segmentation_results.csv and results/segmentation_metrics.json!")

if __name__ == "__main__":
    main()
