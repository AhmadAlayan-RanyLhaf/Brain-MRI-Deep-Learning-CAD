# segmentation_infer.py
import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from src.models.unet3d import UNet3D
from src.datasets.segmentation import _center_crop_or_pad_3d

def reverse_center_crop_or_pad_3d(pred_mask: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """
    Reverses the crop/pad operation to place the 128x128x128 predicted mask back
    into the exact coordinates of the original raw NIfTI scan volume.
    """
    orig_d, orig_h, orig_w = orig_shape
    t = pred_mask.shape[0] # target_size (e.g. 128)
    
    out_mask = np.zeros(orig_shape, dtype=np.uint8)
    
    # 1. Depth (D)
    if orig_d > t:
        s_d = (orig_d - t) // 2
        d_slice_out = slice(s_d, s_d + t)
        d_slice_in = slice(0, t)
    else:
        pad_d = t - orig_d
        p_left = pad_d // 2
        d_slice_out = slice(0, orig_d)
        d_slice_in = slice(p_left, p_left + orig_d)
        
    # 2. Height (H)
    if orig_h > t:
        s_h = (orig_h - t) // 2
        h_slice_out = slice(s_h, s_h + t)
        h_slice_in = slice(0, t)
    else:
        pad_h = t - orig_h
        p_top = pad_h // 2
        h_slice_out = slice(0, orig_h)
        h_slice_in = slice(p_top, p_top + orig_h)
        
    # 3. Width (W)
    if orig_w > t:
        s_w = (orig_w - t) // 2
        w_slice_out = slice(s_w, s_w + t)
        w_slice_in = slice(0, t)
    else:
        pad_w = t - orig_w
        p_front = pad_w // 2
        w_slice_out = slice(0, orig_w)
        w_slice_in = slice(p_front, p_front + orig_w)
        
    out_mask[d_slice_out, h_slice_out, w_slice_out] = pred_mask[d_slice_in, h_slice_in, w_slice_in]
    return out_mask

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to input brain NIfTI scan (.nii or .nii.gz)")
    ap.add_argument("--ckpt", type=str, default="runs/segmentation/best.pt", help="Path to trained model checkpoint")
    ap.add_argument("--output", type=str, required=True, help="Path to save output segmented label NIfTI scan")
    ap.add_argument("--target_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
        
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint file not found: {args.ckpt}")
        return

    # 1. Load original volume
    print(f"Loading input volume: {args.input}")
    img = nib.load(args.input)
    orig_vol = img.get_fdata(dtype=np.float32)
    orig_shape = orig_vol.shape
    print(f"Original shape: {orig_shape}")

    # 2. Preprocess: Crop/Pad and Z-score Normalize
    vol = _center_crop_or_pad_3d(orig_vol, args.target_size)
    mean = vol.mean()
    std = vol.std() + 1e-8
    vol = (vol - mean) / std
    
    # To Tensor: [1, 1, D, H, W]
    x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(device)

    # 3. Load 3D U-Net Model
    print(f"Loading 3D U-Net checkpoint: {args.ckpt}")
    model = UNet3D(in_channels=1, out_channels=4)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # 4. Predict
    print("Running 3D segmentation inference...")
    logits = model(x) # [1, 4, D, H, W]
    pred_vol = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8) # [D, H, W]

    # 5. Reverse crop/pad back to original space
    print("Mapping segmentation back to original voxel grid...")
    orig_pred_mask = reverse_center_crop_or_pad_3d(pred_vol, orig_shape)

    # 6. Save NIfTI output
    print(f"Saving predicted label volume to: {args.output}")
    out_img = nib.Nifti1Image(orig_pred_mask, img.affine, img.header)
    nib.save(out_img, args.output)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
