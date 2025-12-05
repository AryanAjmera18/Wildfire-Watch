import os
import shutil
from pathlib import Path

# ================= CONFIGURATION =================
# Path to your dataset root
DATA_ROOT = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
# =================================================

SEVERITY_FOLDERS = ['severity_low', 'severity_medium', 'severity_high']
SPLITS = ['train', 'val', 'test']

print(f"↺ Reverting dataset at {DATA_ROOT} to Binary (Fire vs No Fire)...")

for split in SPLITS:
    split_dir = DATA_ROOT / split
    
    if not split_dir.exists():
        print(f"⚠️ Skipping {split} (not found)")
        continue
        
    print(f"\nProcessing {split} set...")
    
    # 1. Ensure 'fire' folder exists
    fire_dir = split_dir / 'fire'
    fire_dir.mkdir(exist_ok=True)
    
    # 2. Move files from severity folders back to 'fire'
    for sev_name in SEVERITY_FOLDERS:
        sev_dir = split_dir / sev_name
        
        if sev_dir.exists():
            files = list(sev_dir.glob('*.*'))
            if len(files) > 0:
                print(f"   Moving {len(files)} images from {sev_name} -> fire")
                
                for f in files:
                    dest = fire_dir / f.name
                    try:
                        shutil.move(str(f), str(dest))
                    except shutil.Error:
                        print(f"      ⚠️ File {f.name} already exists in target. Skipping.")
            
            # 3. Delete the empty severity folder
            try:
                sev_dir.rmdir()
                print(f"   🗑️  Deleted empty folder: {sev_name}")
            except OSError:
                print(f"   ⚠️ Could not delete {sev_name} (might not be empty)")

print("\n🎉 Done! Your dataset is now Binary again:")
print(f"   - {DATA_ROOT}\\train\\fire")
print(f"   - {DATA_ROOT}\\train\\nofire")