# segmentation_dataset.py
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

def _center_crop_or_pad_3d(vol: np.ndarray, target_size: int = 128) -> np.ndarray:
    """
    Crop or pad a 3D volume to a symmetric target size.
    """
    d, h, w = vol.shape
    t = target_size
    
    # 1. Handle D dimension
    if d > t:
        s_d = (d - t) // 2
        vol = vol[s_d : s_d + t, :, :]
    elif d < t:
        pad_d = t - d
        p_left = pad_d // 2
        p_right = pad_d - p_left
        vol = np.pad(vol, ((p_left, p_right), (0, 0), (0, 0)), mode="constant", constant_values=0)
        
    # 2. Handle H dimension
    d, h, w = vol.shape
    if h > t:
        s_h = (h - t) // 2
        vol = vol[:, s_h : s_h + t, :]
    elif h < t:
        pad_h = t - h
        p_top = pad_h // 2
        p_bottom = pad_h - p_top
        vol = np.pad(vol, ((0, 0), (p_top, p_bottom), (0, 0)), mode="constant", constant_values=0)
        
    # 3. Handle W dimension
    d, h, w = vol.shape
    if w > t:
        s_w = (w - t) // 2
        vol = vol[:, :, s_w : s_w + t]
    elif w < t:
        pad_w = t - w
        p_front = pad_w // 2
        p_back = pad_w - p_front
        vol = np.pad(vol, ((0, 0), (0, 0), (p_front, p_back)), mode="constant", constant_values=0)
        
    return vol

class MiriadSegmentationDataset(Dataset):
    """
    Dataset for 3D multi-class brain tissue segmentation.
    Loads raw 3D volume and returns:
      - x: [1, D, H, W] normalized float32 image volume
      - y: [D, H, W] long label volume (0=BG, 1=CSF, 2=GM, 3=WM)
    """
    def __init__(self, csv_path: str, target_size: int = 128, augment: bool = False, subject_ids: list = None):
        self.df = pd.read_csv(csv_path)
        if subject_ids is not None:
            self.df = self.df[self.df["subject_id"].isin(subject_ids)].reset_index(drop=True)
        self.target_size = int(target_size)
        self.augment = bool(augment)
        
    def __len__(self):
        return len(self.df)
        
    def _load_volume(self, path: str) -> np.ndarray:
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32)
        
    def _augment(self, x: np.ndarray) -> np.ndarray:
        # Intensity scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        # Random noise addition
        if np.random.rand() < 0.5:
            noise = np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
            x = x + noise
        return x
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        lbl_path = row["label_path"]
        
        # 1. Load volumes
        vol_img = self._load_volume(img_path)
        vol_lbl = self._load_volume(lbl_path).astype(np.int64) # Keep label indices as long integer
        
        # 2. Match sizes symmetrically
        vol_img = _center_crop_or_pad_3d(vol_img, self.target_size)
        vol_lbl = _center_crop_or_pad_3d(vol_lbl, self.target_size)
        
        # 3. Z-score Normalize the image volume (foreground-only to prevent background padding skew)
        fg_mask = vol_img > 0
        if fg_mask.sum() > 0:
            mean = vol_img[fg_mask].mean()
            std = vol_img[fg_mask].std() + 1e-8
            vol_img = (vol_img - mean) / std
            vol_img[~fg_mask] = 0.0
        else:
            mean = vol_img.mean()
            std = vol_img.std() + 1e-8
            vol_img = (vol_img - mean) / std
        
        # 4. Augment if training
        if self.augment:
            vol_img = self._augment(vol_img)
            # Random flips along 3D axes (apply same transformations to label to keep aligned)
            if np.random.rand() < 0.5:
                vol_img = np.flip(vol_img, axis=0).copy()
                vol_lbl = np.flip(vol_lbl, axis=0).copy()
            if np.random.rand() < 0.5:
                vol_img = np.flip(vol_img, axis=1).copy()
                vol_lbl = np.flip(vol_lbl, axis=1).copy()
            if np.random.rand() < 0.5:
                vol_img = np.flip(vol_img, axis=2).copy()
                vol_lbl = np.flip(vol_lbl, axis=2).copy()
            
        # 5. Convert to PyTorch Tensors
        x = torch.from_numpy(vol_img).unsqueeze(0).float() # [1, D, H, W]
        y = torch.from_numpy(vol_lbl).long()               # [D, H, W]
        
        return x, y
