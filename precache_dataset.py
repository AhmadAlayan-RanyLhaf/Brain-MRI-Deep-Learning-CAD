# precache_dataset.py
import os
import numpy as np
import pandas as pd
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataset import _center_crop_or_pad, _robust_normalize

def process_single_scan(path):
    try:
        # Resolve path
        resolved_path = path.replace("\\", "/")
        if "miriad/" in resolved_path:
            suffix = resolved_path.split("miriad/", 1)[1]
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            resolved_path = os.path.join(workspace_root, "miriad", suffix).replace("\\", "/")
            
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
            return "cached"
            
        # Load and preprocess
        img = nib.load(resolved_path)
        vol = img.get_fdata(dtype=np.float32)
        
        # Center crop or pad
        vol = _center_crop_or_pad(vol, (128, 128, 128))
        
        # Normalize
        vol = _robust_normalize(vol, 0.5, 99.5, 1e-6)
        
        # Save
        np.save(npy_path, vol)
        return "processed"
    except Exception as e:
        return f"failed: {e} for path: {path}"

def main():
    csv_path = "miriad_index.csv"
    if not os.path.exists(csv_path):
        print(f"Index CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    paths = df["path"].unique()
    print(f"Total unique paths to check: {len(paths)}")
    
    processed_count = 0
    cached_count = 0
    failed_count = 0
    
    # Run process pool
    print("Starting parallel precaching with ProcessPoolExecutor...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_scan, p): p for p in paths}
        
        for idx, future in enumerate(as_completed(futures)):
            res = future.result()
            if res == "cached":
                cached_count += 1
            elif res == "processed":
                processed_count += 1
            else:
                failed_count += 1
                print(f"Error: {res}")
                
            if (idx + 1) % 50 == 0:
                print(f"Progress: {idx+1}/{len(paths)} scans checked (processed: {processed_count}, cached: {cached_count}, failed: {failed_count})")
                
    print(f"\nCompleted! Processed: {processed_count}, Cached: {cached_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()
