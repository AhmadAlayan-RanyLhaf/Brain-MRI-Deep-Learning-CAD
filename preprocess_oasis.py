# preprocess_oasis.py
import os
import sys
import tarfile
import numpy as np
import pandas as pd
import nibabel as nib

# Add workspace to path to allow importing from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.datasets.miriad import _center_crop_or_pad, _robust_normalize

project_root = os.path.dirname(os.path.abspath(__file__))
oasis_raw_dir = os.path.join(project_root, "oasis_raw")
oasis_processed_dir = os.path.join(project_root, "oasis_processed")
os.makedirs(oasis_processed_dir, exist_ok=True)

# 1. Extract the tar.gz archive if not already done
tar_path = os.path.join(project_root, "oasis_cross-sectional_disc1.tar.gz")
disc1_dir = os.path.join(oasis_raw_dir, "disc1")

if not os.path.exists(disc1_dir):
    print(f"OASIS-1 disc1 not found at {disc1_dir}. Extracting {tar_path}...")
    if not os.path.exists(tar_path):
        print(f"ERROR: Tarball not found at {tar_path}")
        sys.exit(1)
    
    os.makedirs(oasis_raw_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        # Extract files one by one with progress printing
        members = tar.getmembers()
        total = len(members)
        print(f"Total entries to extract: {total}")
        for i, member in enumerate(members):
            tar.extract(member, path=oasis_raw_dir)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  Extracted {i + 1}/{total} files...")
    print("Extraction completed successfully!")
else:
    print("OASIS-1 disc1 already extracted.")

# 2. Find the Excel sheet
excel_files = [f for f in os.listdir(project_root) if f.startswith("oasis_cross-sectional") and f.endswith(".xlsx")]
if not excel_files:
    print("ERROR: Excel file not found in project root.")
    sys.exit(1)
excel_path = os.path.join(project_root, excel_files[0])
print(f"Using Excel file: {excel_path}")

# Load demographic information
df = pd.read_excel(excel_path)
print(f"Total rows in Excel: {len(df)}")

# Filter out rows without CDR (subjects under age 60/70 typically have NaN CDR in OASIS-1)
df_clean = df.dropna(subset=["CDR"]).copy()
print(f"Rows with valid CDR: {len(df_clean)}")

# 3. Process each subject found in disc1
processed_records = []
disc1_subjects = os.listdir(disc1_dir)
print(f"Found {len(disc1_subjects)} subjects in disc1 directory.")

for subject_folder in sorted(disc1_subjects):
    subject_path = os.path.join(disc1_dir, subject_folder)
    if not os.path.isdir(subject_path):
        continue
    
    # Check if this subject is in our cleaned dataframe
    subject_rows = df_clean[df_clean["ID"] == subject_folder]
    if subject_rows.empty:
        print(f"Skipping {subject_folder}: No valid CDR in Excel.")
        continue
        
    row = subject_rows.iloc[0]
    cdr = float(row["CDR"])
    sex = str(row["M/F"])  # 'M' or 'F'
    label = 1 if cdr > 0 else 0  # 0 = HC, 1 = AD
    
    # Locate the RAW mpr-1 volume
    raw_dir = os.path.join(subject_path, "RAW")
    if not os.path.exists(raw_dir):
        print(f"RAW directory not found for {subject_folder}, skipping.")
        continue
        
    # Analyze format typically has .img and .hdr files. We look for mpr-1 .hdr
    hdr_name = f"{subject_folder}_mpr-1_anon.hdr"
    hdr_path = os.path.join(raw_dir, hdr_name)
    
    # Fallback to look for any other mpr hdr if mpr-1 is missing
    if not os.path.exists(hdr_path):
        hdrs = [f for f in os.listdir(raw_dir) if f.endswith(".hdr")]
        if hdrs:
            hdr_path = os.path.join(raw_dir, sorted(hdrs)[0])
        else:
            print(f"No .hdr files found in RAW directory for {subject_folder}, skipping.")
            continue
            
    print(f"Processing {subject_folder} using {os.path.basename(hdr_path)} (Label: {label}, CDR: {cdr})...")
    
    try:
        # Load the Analyze volume using nibabel
        img = nib.load(hdr_path)
        vol_3d = np.squeeze(img.get_fdata(dtype=np.float32))
        img_3d = nib.Nifti1Image(vol_3d, img.affine)
        
        # Resample to MIRIAD spacing (L-R=0.9375, A-P=1.5, S-I=0.9375 in original LAS coordinates)
        import nibabel.processing as nib_proc
        img_resampled = nib_proc.resample_to_output(img_3d, (0.9375, 1.5, 0.9375))
        
        # Reorient from OASIS ('L', 'A', 'S') to MIRIAD ('L', 'S', 'P')
        orig_ornt = nib.io_orientation(img_resampled.affine)
        target_ornt = nib.orientations.axcodes2ornt(('L', 'S', 'P'))
        ornt_trans = nib.orientations.ornt_transform(orig_ornt, target_ornt)
        
        vol = img_resampled.get_fdata(dtype=np.float32)
        vol = nib.orientations.apply_orientation(vol, ornt_trans)
        
        # Center crop or pad to (128, 128, 128)
        vol = _center_crop_or_pad(vol, (128, 128, 128))
        
        # Z-score normalize
        vol = _robust_normalize(vol, 0.5, 99.5, 1e-6)
        
        # Save as .npy
        npy_name = f"{subject_folder}.npy"
        npy_path = os.path.join(oasis_processed_dir, npy_name)
        np.save(npy_path, vol)
        
        # Record metadata
        # Store both absolute and relative path for safety, we'll write relative path to the index
        rel_npy_path = f"oasis_processed/{npy_name}"
        
        # Parse subject ID to numeric
        # ID is usually like OAS1_0001_MR1
        sub_num = int(subject_folder.split("_")[1])
        
        processed_records.append({
            "path": rel_npy_path,
            "label": label,
            "subject_id": sub_num,
            "sex": sex
        })
    except Exception as e:
        print(f"  FAILED to process {subject_folder}: {e}")

# 4. Save the index file
index_df = pd.DataFrame(processed_records)
index_csv_path = os.path.join(project_root, "oasis_index.csv")
index_df.to_csv(index_csv_path, index=False)
print(f"\nPreprocessing finished! Saved {len(processed_records)} records to {index_csv_path}")
