# dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    # Target volume size after center-crop/pad (X, Y, Z)
    target_shape: Tuple[int, int, int] = (128, 128, 128)

    # Intensity normalization
    clip_percentiles: Tuple[float, float] = (0.5, 99.5)  # robust clipping
    eps: float = 1e-6

    # Caching (keeps already-loaded NIfTIs in RAM to speed up training)
    use_cache: bool = False


def _center_crop_or_pad(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or zero-pad a 3D volume to target shape."""
    out = np.zeros(target, dtype=vol.dtype)
    src = vol

    for axis in range(3):
        src_len = src.shape[axis]
        tgt_len = target[axis]

        if src_len >= tgt_len:
            # crop
            start = (src_len - tgt_len) // 2
            end = start + tgt_len
            slicer = [slice(None)] * 3
            slicer[axis] = slice(start, end)
            src = src[tuple(slicer)]
        else:
            # will pad into out later
            pass

    # now src is <= target on each axis (after possible cropping)
    insert_slices = []
    src_slices = []
    for axis in range(3):
        src_len = src.shape[axis]
        tgt_len = target[axis]
        start = (tgt_len - src_len) // 2
        insert_slices.append(slice(start, start + src_len))
        src_slices.append(slice(0, src_len))

    out[tuple(insert_slices)] = src[tuple(src_slices)]
    return out


def _robust_normalize(vol: np.ndarray, p_lo: float, p_hi: float, eps: float) -> np.ndarray:
    """Clip by percentiles then z-score normalize."""
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    vol = np.clip(vol, lo, hi)

    mean = float(vol.mean())
    std = float(vol.std())
    vol = (vol - mean) / (std + eps)
    return vol.astype(np.float32)


class MiriadMRIDataset(Dataset):
    """
    Loads MIRIAD MRI scans listed in miriad_index.csv.
    Expects CSV columns: path, label, subject_id, sex

    Returns:
      x: torch.FloatTensor [1, D, H, W] (channel-first 3D)
      y: torch.LongTensor scalar (0=HC, 1=AD)
      meta: dict with path, subject_id, sex
    """

    def __init__(
        self,
        csv_path: str = "miriad_index.csv",
        config: Optional[DatasetConfig] = None,
        is_train: bool = False,
    ):
        self.csv_path = csv_path
        self.cfg = config or DatasetConfig()
        self.is_train = is_train

        df = pd.read_csv(csv_path)
        required = {"path", "label", "subject_id", "sex"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}. Found: {list(df.columns)}")

        # Keep a clean copy
        self.df = df.reset_index(drop=True)

        # Optional cache: path -> np.ndarray
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_nifti(self, path: str) -> np.ndarray:
        import os
        path = path.replace("\\", "/")
        if "miriad/" in path:
            suffix = path.split("miriad/", 1)[1]
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(workspace_root, "miriad", suffix).replace("\\", "/")

        if self.cfg.use_cache and path in self._cache:
            return self._cache[path]

        img = nib.load(path)

        # Use float32 early to save memory
        data = img.get_fdata(dtype=np.float32)

        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape} for {path}")

        if self.cfg.use_cache:
            self._cache[path] = data
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        path = str(row["path"])
        label = int(row["label"])
        subject_id = int(row["subject_id"])
        sex = str(row["sex"])

        # JIT Disk Caching to speed up training from ~70s/epoch to ~1s/epoch
        import os
        processed_dir = "miriad_processed"
        os.makedirs(processed_dir, exist_ok=True)
        safe_name = path.replace("\\", "/").replace("/", "_").replace(":", "")
        if safe_name.endswith(".nii.gz"):
            safe_name = safe_name[:-7] + ".npy"
        elif safe_name.endswith(".nii"):
            safe_name = safe_name[:-4] + ".npy"
        else:
            safe_name = safe_name + ".npy"
        npy_path = os.path.join(processed_dir, safe_name)

        if os.path.exists(npy_path):
            vol = np.load(npy_path)
        else:
            vol = self._load_nifti(path)
            # Preprocess: crop/pad -> normalize
            vol = _center_crop_or_pad(vol, self.cfg.target_shape)
            vol = _robust_normalize(vol, self.cfg.clip_percentiles[0], self.cfg.clip_percentiles[1], self.cfg.eps)
            np.save(npy_path, vol)

        if self.is_train:
            import random
            # Left-Right sagittal reflection only (axis=0)
            if random.random() < 0.5:
                vol = np.flip(vol, axis=0).copy()

        # Convert to torch: [1, D, H, W]
        x = torch.from_numpy(vol).unsqueeze(0)  # channel
        y = torch.tensor(label, dtype=torch.long)

        meta = {"path": path, "subject_id": subject_id, "sex": sex}
        return x, y, meta
