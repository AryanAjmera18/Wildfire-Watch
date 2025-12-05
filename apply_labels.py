import pandas as pd
import shutil
from pathlib import Path
import os
from tqdm import tqdm

# User-provided mapping
CLUSTER_MAPPING = {
    # High Severity
    "high": [4, 5, 8, 12, 17, 18, 21, 23, 24, 25, 27, 31, 34, 46],
    
    # Medium Severity
    "medium": [9, 11, 13, 19, 22, 28, 29, 30, 33, 35, 38, 39, 41, 42, 45, 47, 14, 49],
    
    # Low Severity
    "low": [0, 1, 2, 3, 6, 7, 10, 15, 16, 20, 26, 32, 36, 37, 40, 43, 44, 48]
}

def get_severity(cluster_id):
    for severity, clusters in CLUSTER_MAPPING.items():
        if cluster_id in clusters:
            return severity
    return None

def apply_labels():
    # Load cluster assignments
    if not os.path.exists("image_clusters.csv"):
        print("Error: image_clusters.csv not found. Run cluster_assist.py first.")
        return
        
    df = pd.read_csv("image_clusters.csv")
    print(f"Loaded {len(df)} image assignments.")
    
    # Source and Target Dirs
    source_root = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
    target_root = Path(r"C:\NYU\ACV\the_wildfire_dataset_severity")
    
    if target_root.exists():
        print(f"Warning: Target directory {target_root} already exists. Merging/Overwriting...")
    
    # Create structure
    for split in ['train', 'val', 'test']:
        for label in ['low', 'medium', 'high']:
            (target_root / split / label).mkdir(parents=True, exist_ok=True)
            
    print("Copying files to new dataset structure...")
    
    copied_count = 0
    skipped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_path = Path(row['path'])
        cluster_id = row['cluster']
        
        severity = get_severity(cluster_id)
        if severity is None:
            print(f"Warning: Cluster {cluster_id} not found in mapping. Skipping {src_path.name}")
            skipped_count += 1
            continue
            
        # Determine split from original path
        # Assuming path structure: .../split/class/filename
        # We need to be robust here.
        parts = src_path.parts
        if 'train' in parts:
            split = 'train'
        elif 'val' in parts:
            split = 'val'
        elif 'test' in parts:
            split = 'test'
        else:
            # Fallback or skip? Let's skip for safety if we can't determine split
            print(f"Could not determine split for {src_path}. Skipping.")
            skipped_count += 1
            continue
            
        dest_path = target_root / split / severity / src_path.name
        
        try:
            shutil.copy2(src_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            skipped_count += 1

    print(f"\nDone! Copied {copied_count} images. Skipped {skipped_count}.")
    print(f"New dataset created at: {target_root}")

if __name__ == "__main__":
    apply_labels()
