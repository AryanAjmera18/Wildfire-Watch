import os
import shutil
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import models, transforms

# ================= CONFIGURATION =================
# 🔴 CORRECTED PATH based on your input
# We point to the 'train' folder specifically
DATA_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version\train")

# We will look for the 'fire' folder here
FIRE_DIR = DATA_DIR / "fire"

# And create these new split folders
TARGET_DIRS = {
    0: DATA_DIR / "severity_low",
    1: DATA_DIR / "severity_medium",
    2: DATA_DIR / "severity_high"
}
# =================================================

# 0. Handle huge images if present
Image.MAX_IMAGE_PIXELS = None 

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Auto-Labeler using device: {device}")

# 2. Load Feature Extractor (ResNet18)
# We use a standard CNN to get the "visual fingerprint" of the smoke/fire
print("🧠 Loading feature extractor...")
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
# Remove the classification layer to get raw features
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Scan Images
print(f"🔍 Scanning {FIRE_DIR}...")
if not FIRE_DIR.exists():
    print(f"❌ ERROR: Could not find {FIRE_DIR}")
    print(f"Please check that 'train' and 'fire' folders exist inside {DATA_DIR.parent}")
    exit()

image_paths = list(FIRE_DIR.glob("*.jpg")) + list(FIRE_DIR.glob("*.png")) + list(FIRE_DIR.glob("*.jpeg"))

if len(image_paths) == 0:
    print("❌ No images found! The folder is empty.")
    exit()

print(f"📸 Found {len(image_paths)} fire images. Extracting features...")

# 4. Extract Features
features = []
valid_paths = []
BATCH_SIZE = 64

# Process in batches for speed
for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_tensors = []
    
    for p in batch_paths:
        try:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        except Exception as e:
            print(f"Skipping bad image: {p}")
            continue
            
    if not batch_tensors: continue
    
    batch_stack = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        # Get features (512-dimensional vector per image)
        emb = model(batch_stack).squeeze(-1).squeeze(-1)
        features.append(emb.cpu().numpy())
        valid_paths.extend(batch_paths)

all_features = np.concatenate(features)

# 5. K-Means Clustering (The "Weak Supervision")
print("✨ Clustering images into 3 severity levels...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(all_features)

# 6. Move Files to New Folders
print("📂 Reorganizing dataset...")
for d in TARGET_DIRS.values():
    d.mkdir(exist_ok=True)

counts = {0: 0, 1: 0, 2: 0}

for img_path, label in zip(valid_paths, labels):
    dest_folder = TARGET_DIRS[label]
    filename = img_path.name
    # Move the file
    shutil.move(str(img_path), str(dest_folder / filename))
    counts[label] += 1

# 7. Cleanup
# Try to remove the original 'fire' folder if it's empty
try:
    FIRE_DIR.rmdir()
    print("🗑️  Removed empty 'fire' folder.")
except:
    print("⚠️ 'fire' folder not empty (or locked), so I left it there.")

print("\n🎉 SUCCESS! Dataset automatically labeled via Weak Supervision.")
print(f"   severity_low (Cluster 0): {counts[0]} images")
print(f"   severity_medium (Cluster 1): {counts[1]} images")
print(f"   severity_high (Cluster 2): {counts[2]} images")
print("\n⚠️ IMPORTANT: Open the folders and check the visuals!")
print("If 'Cluster 0' looks like heavy fire, rename 'severity_low' to 'severity_high'.")