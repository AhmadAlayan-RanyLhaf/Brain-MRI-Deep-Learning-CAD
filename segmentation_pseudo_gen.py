# segmentation_pseudo_gen.py
import os
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes

def extract_brain_mask(vol: np.ndarray) -> np.ndarray:
    """
    Simple threshold-based brain extraction (skull stripping) and morphology.
    """
    # Find a threshold based on high percentiles to ignore background noise
    thresh = np.percentile(vol, 2) + 0.10 * (np.percentile(vol, 98) - np.percentile(vol, 2))
    mask = vol > thresh
    
    # Fill holes and clean up using basic morphology
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=3)
    mask = binary_fill_holes(mask)
    
    return mask

def segment_brain_tissues(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Segment the extracted brain volume into 3 classes (CSF, GM, WM) using K-Means clustering.
    CSF = 1 (darkest tissue), GM = 2 (medium tissue), WM = 3 (brightest tissue).
    """
    label_map = np.zeros_like(vol, dtype=np.uint8)
    
    # Get active voxels inside the brain mask
    brain_voxels = vol[mask]
    if len(brain_voxels) == 0:
        return label_map
        
    # Fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(brain_voxels.reshape(-1, 1))
    
    # Order clusters by their mean intensity
    means = kmeans.cluster_centers_.flatten()
    ordered_indices = np.argsort(means) # [darkest_idx, medium_idx, brightest_idx]
    
    # Map cluster assignments to medical classes: 1=CSF, 2=GM, 3=WM
    mapped_clusters = np.zeros_like(clusters, dtype=np.uint8)
    for target_class, orig_idx in enumerate(ordered_indices, start=1):
        mapped_clusters[orig_idx == clusters] = target_class
        
    # Write back to 3D label map
    label_map[mask] = mapped_clusters
    return label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_subjects", type=int, default=5, help="Number of subjects to generate data for")
    ap.add_argument("--out_dir", type=str, default="segmentation_data", help="Output directory for segmentation data")
    args = ap.parse_args()

    print("==================================================")
    print("      GENERATING PSEUDO-GROUND TRUTH LABELS       ")
    print("==================================================")

    # Output paths
    img_dir = os.path.join(args.out_dir, "images")
    lbl_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Load master miriad index
    index_csv = "miriad_index.csv"
    if not os.path.exists(index_csv):
        # Fall back to folder search if csv not found
        print(f"Missing master index: {index_csv}")
        return
        
    df = pd.read_csv(index_csv)
    
    # Group by subject_id to select the first visit of selected subjects
    subjects = df["subject_id"].unique()[:args.num_subjects]
    selected_df = df[df["subject_id"].isin(subjects)].groupby("subject_id").first().reset_index()

    rows = []

    for idx, row in selected_df.iterrows():
        orig_path = row["path"].replace("\\", "/")
        
        # Dynamically resolve local path
        if "miriad/" in orig_path:
            suffix = orig_path.split("miriad/", 1)[1]
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_path = os.path.join(workspace_root, "miriad", suffix).replace("\\", "/")
        else:
            local_path = orig_path

        if not os.path.exists(local_path):
            print(f"Skipping missing file: {local_path}")
            continue

        print(f"Processing Subject {row['subject_id']} from: {os.path.basename(local_path)}")

        # 1. Load NIfTI
        img = nib.load(local_path)
        vol = img.get_fdata(dtype=np.float32)

        # 2. Extract Brain Mask
        mask = extract_brain_mask(vol)

        # 3. Cluster Tissues using intensity-based sorting.
        # This yields pseudo-ground truth labels generated using K-Means clustering.
        # It maps clusters by intensity: CSF = 1 (darkest), GM = 2 (medium), WM = 3 (brightest).
        lbl_vol = segment_brain_tissues(vol, mask)

        # Apply the brain mask to perform skull-stripping (background masking) on the raw image
        vol_masked = vol * mask

        # 4. Save Image and Label as NIfTI
        subject_str = f"sub-{row['subject_id']}"
        img_name = f"{subject_str}_img.nii.gz"
        lbl_name = f"{subject_str}_lbl.nii.gz"

        out_img_path = os.path.join(img_dir, img_name).replace("\\", "/")
        out_lbl_path = os.path.join(lbl_dir, lbl_name).replace("\\", "/")

        # Save volumes using the original affine and header to maintain spatial coordinates
        # We save the skull-stripped image (vol_masked) to improve model focus
        nib.save(nib.Nifti1Image(vol_masked, img.affine, img.header), out_img_path)
        nib.save(nib.Nifti1Image(lbl_vol, img.affine, img.header), out_lbl_path)

        rows.append({
            "subject_id": int(row["subject_id"]),
            "image_path": out_img_path,
            "label_path": out_lbl_path
        })
        print(f"  Saved Image: {out_img_path}")
        print(f"  Saved Label: {out_lbl_path}")

    # Save index CSV for our training dataset loader
    out_csv = os.path.join(args.out_dir, "segmentation_index.csv").replace("\\", "/")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nCreated segmentation index CSV: {out_csv}")
    print("Generation completed successfully!")

if __name__ == "__main__":
    main()
