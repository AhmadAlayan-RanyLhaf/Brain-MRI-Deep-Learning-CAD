# segmentation_train.py
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path

from src.models.unet3d import UNet3D
from src.datasets.segmentation import MiriadSegmentationDataset

class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss. Computes Dice over foreground classes (1, 2, 3) 
    and ignores background class 0.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> torch.Tensor:
        # logits: [B, C, D, H, W]
        # targets: [B, D, H, W]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Sum over batch (0) and spatial dimensions (2, 3, 4)
        intersection = torch.sum(probs * one_hot, dim=(0, 2, 3, 4))
        union = torch.sum(probs, dim=(0, 2, 3, 4)) + torch.sum(one_hot, dim=(0, 2, 3, 4))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # Average Dice loss over foreground classes (1, 2, 3)
        return 1.0 - torch.mean(dice[1:])

def calculate_class_dices(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4) -> dict:
    """
    Compute Dice Coefficient for each foreground class (1=CSF, 2=GM, 3=WM).
    """
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1) # [B, D, H, W]
        dices = {}
        for c in range(1, num_classes):
            p_mask = (preds == c).float()
            t_mask = (targets == c).float()
            
            intersection = torch.sum(p_mask * t_mask).item()
            union = torch.sum(p_mask).item() + torch.sum(t_mask).item()
            
            if union == 0:
                # If neither contains the class, overlap is perfect (1.0)
                dices[c] = 1.0
            else:
                dices[c] = (2.0 * intersection) / union
        return dices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="segmentation_data/segmentation_index.csv")
    ap.add_argument("--epochs", type=int, default=50) # Increased epoch limit since early stopping is active
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="runs/segmentation")
    ap.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset / DataLoaders
    if not os.path.exists(args.csv):
        print(f"Index CSV not found: {args.csv}. Run segmentation_pseudo_gen.py first!")
        return

    # Hardcoded subject splits to prevent data leakage and ensure Subject 188 is unseen
    train_subjects = [190, 191, 192, 193, 194, 195, 196, 197]
    val_subjects = [189]
    test_subjects = [188]

    print(f"Dataset subject-level splits:")
    print(f"  Train subjects: {train_subjects}")
    print(f"  Val subjects:   {val_subjects}")
    print(f"  Test subjects:  {test_subjects}")

    train_ds = MiriadSegmentationDataset(args.csv, target_size=args.target_size, augment=True, subject_ids=train_subjects)
    val_ds = MiriadSegmentationDataset(args.csv, target_size=args.target_size, augment=False, subject_ids=val_subjects)

    n_train = len(train_ds)
    n_val = len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset splits: {n_train} train | {n_val} validation")

    # 2. Model & Optimizers
    model = UNet3D(in_channels=1, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    # 3. Hybrid Losses
    ce_crit = nn.CrossEntropyLoss()
    dice_crit = SoftDiceLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    log_path = out_dir / "training_log.csv"
    
    best_dice = -1.0
    best_class_dices = {}
    epochs_no_improve = 0
    history = []

    print("\nStarting 3D U-Net Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss_ce = ce_crit(logits, y)
                loss_dice = dice_crit(logits, y)
                loss = loss_ce + 1.5 * loss_dice  # Hybrid loss weighting

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

        train_loss = train_loss / len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Accumulate class-wise intersection/union over the entire validation dataset for correct Dice calculation
        total_intersections = torch.zeros(4, device=device)
        total_unions = torch.zeros(4, device=device)
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss_ce = ce_crit(logits, y)
                    loss_dice = dice_crit(logits, y)
                    loss = loss_ce + 1.5 * loss_dice
                
                val_loss += loss.item() * x.size(0)
                
                # Accumulate overlap components
                preds = torch.argmax(logits, dim=1)
                for c in range(1, 4):
                    p_mask = (preds == c)
                    t_mask = (y == c)
                    total_intersections[c] += torch.sum(p_mask & t_mask).item()
                    total_unions[c] += (torch.sum(p_mask) + torch.sum(t_mask)).item()

        val_loss = val_loss / len(val_ds)
        scheduler.step(val_loss)

        # Calculate final validation Dice scores
        val_class_dices = {}
        for c in range(1, 4):
            inter = total_intersections[c].item()
            uni = total_unions[c].item()
            if uni == 0:
                val_class_dices[c] = 1.0
            else:
                val_class_dices[c] = (2.0 * inter) / uni
        
        val_dice = np.mean([val_class_dices[c] for c in range(1, 4)])

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice (Mean): {val_dice:.4f} "
              f"(CSF: {val_class_dices[1]:.4f}, GM: {val_class_dices[2]:.4f}, WM: {val_class_dices[3]:.4f})")

        # Save history log
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice_mean": val_dice,
            "val_dice_csf": val_class_dices[1],
            "val_dice_gm": val_class_dices[2],
            "val_dice_wm": val_class_dices[3]
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        # Early Stopping & Checkpoint Saving
        if val_dice > best_dice:
            best_dice = val_dice
            best_class_dices = val_class_dices
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_dice": best_dice,
                "class_dices": best_class_dices
            }, best_path)
            print(f"  [SUCCESS] Saved best checkpoint to {best_path} (Mean Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  [INFO] Early stopping triggered. No improvement for {args.patience} epochs.")
                break

    print(f"\nTraining Finished! Best Val Dice Score: {best_dice:.4f} "
          f"(CSF: {best_class_dices.get(1, 0):.4f}, GM: {best_class_dices.get(2, 0):.4f}, WM: {best_class_dices.get(3, 0):.4f})")

if __name__ == "__main__":
    main()
