# dataset_progression.py
import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset


def _center_crop_or_pad(vol: np.ndarray, target_size: int) -> np.ndarray:
    """
    vol: (D,H,W)
    returns: (target_size,target_size,target_size)
    """
    d, h, w = vol.shape
    ts = target_size

    out = np.zeros((ts, ts, ts), dtype=vol.dtype)

    # compute start indices for input and output
    def _start(in_size, out_size):
        if in_size >= out_size:
            in_start = (in_size - out_size) // 2
            out_start = 0
            length = out_size
        else:
            in_start = 0
            out_start = (out_size - in_size) // 2
            length = in_size
        return in_start, out_start, length

    d_in, d_out, d_len = _start(d, ts)
    h_in, h_out, h_len = _start(h, ts)
    w_in, w_out, w_len = _start(w, ts)

    out[d_out:d_out + d_len, h_out:h_out + h_len, w_out:w_out + w_len] = \
        vol[d_in:d_in + d_len, h_in:h_in + h_len, w_in:w_in + w_len]

    return out


def _normalize_zscore(vol: np.ndarray) -> np.ndarray:
    # simple, stable normalization
    mean = vol.mean()
    std = vol.std()
    if std < 1e-6:
        return vol * 0.0
    return (vol - mean) / (std + 1e-6)


class MiriadProgressionDataset(Dataset):
    """
    Expects a CSV with at least:
      - path (to .nii or .nii.gz)
      - stage (float in [0,1])
    Optional (but recommended, used for eval outputs):
      - subject_id, visit, scan_number, group, sex
    """
    def __init__(self, csv_path: str, target_size: int = 128, augment: bool = False):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns:
            raise ValueError("CSV must contain a 'path' column.")
        if "stage" not in self.df.columns:
            raise ValueError("CSV must contain a 'stage' column for regression.")
        self.target_size = int(target_size)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.df)

    def _load_nii(self, path: str) -> np.ndarray:
        path = path.replace("\\", "/")
        if "miriad/" in path:
            suffix = path.split("miriad/", 1)[1]
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(workspace_root, "miriad", suffix).replace("\\", "/")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        img = nib.load(path)
        vol = img.get_fdata(dtype=np.float32)  # (D,H,W)
        return vol

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # Safe augmentations for brain MRI (no left-right flips)
        # intensity scale + small gaussian noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        if np.random.rand() < 0.5:
            noise = np.random.normal(0.0, 0.05, size=x.shape).astype(np.float32)
            x = x + noise
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["path"]
        stage = float(row["stage"])

        vol = self._load_nii(path)
        vol = _center_crop_or_pad(vol, self.target_size)
        vol = _normalize_zscore(vol)
        if self.augment:
            vol = self._augment(vol)

        # to torch: [C,D,H,W]
        x = torch.from_numpy(vol).unsqueeze(0).float()
        y = torch.tensor(stage, dtype=torch.float32)

        return x, y, row.to_dict()
