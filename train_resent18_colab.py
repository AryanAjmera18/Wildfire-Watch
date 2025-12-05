# ==========================================
# 🔥 Wildfire Smoke Detection: Colab Pipeline (Data Pre-loaded)
# Group 8: Aryan Ajmera, Rushabh Bhatt, Soham, Dhrity
# ==========================================

import os
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import torch.nn as nn

# --- CONFIGURATION ---
# Path to the dataset already present in the environment
# Based on your previous successful runs, it seems to be here:
SEARCH_DIR = Path("/kaggle/input/the-wildfire-dataset/the_wildfire_dataset_2n_version")

# Define writable directory for reorganized data
# We MUST copy data to /content/ because /kaggle/input is read-only
WRITABLE_DIR = Path("/content/wildfire_data")

# --- STEP 1: INSTALL LIBRARIES ---
print("\n🚀 Step 1: Installing Deep Learning Libraries...")
os.system('pip install -q transformers datasets accelerate scikit-learn grad-cam') 

# Fix for huge images
Image.MAX_IMAGE_PIXELS = None

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Compute Device: {device}")
if device.type != 'cuda':
    print("⚠️ WARNING: Running on CPU! Enable GPU in Runtime > Change Runtime Type.")

# --- STEP 2: PREPARE DATA (Copy to Writable Space) ---
print("\n🚀 Step 2: Preparing Writable Dataset...")

if not WRITABLE_DIR.exists():
    if not SEARCH_DIR.exists():
        print(f"❌ Critical Error: Source data not found at {SEARCH_DIR}")
        print("Please check where your data is located.")
        exit()
        
    print(f"   Copying data from {SEARCH_DIR} to {WRITABLE_DIR}...")
    # Find the 'train' folder inside the source
    train_found = list(SEARCH_DIR.rglob("train"))
    if not train_found:
        print("❌ Error: 'train' folder not found in source.")
        exit()
        
    # The parent of 'train' is the root we want to copy
    source_root = train_found[0].parent
    shutil.copytree(source_root, WRITABLE_DIR)
    print("✅ Data copied successfully.")
else:
    print("✅ Writable workspace already exists.")

# Update SEARCH_DIR to point to our new writable location
SEARCH_DIR = WRITABLE_DIR

# ==========================================
# MODULE: CLUSTER-ASSIST (Semi-Supervised Labeling)
# ==========================================
print("\n🚀 Step 3: Applying Severity Labels (Cluster-Assist)...")

TRAIN_FIRE_DIR = SEARCH_DIR / "train" / "fire"

# 🔴 HARDCODED CLUSTER MAPPING (From Human Review)
CLUSTER_MAPPING = {
    "severity_high": [4, 5, 8, 12, 17, 18, 21, 23, 24, 25, 27, 31, 34, 46],
    "severity_medium": [9, 11, 13, 14, 19, 22, 28, 29, 30, 33, 35, 38, 39, 41, 42, 45, 47, 49],
    "severity_low": [0, 1, 2, 3, 6, 7, 10, 15, 16, 20, 26, 32, 36, 37, 40, 43, 44, 48]
}

if TRAIN_FIRE_DIR.exists():
    print("   Extracting visual features (ResNet18)...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) 
    resnet.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_paths = list(TRAIN_FIRE_DIR.glob("*.jpg")) + list(TRAIN_FIRE_DIR.glob("*.png"))
    features = []
    
    for i in tqdm(range(0, len(image_paths), 64)):
        batch_paths = image_paths[i:i+64]
        batch = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(transform(img))
            except: pass
        
        if batch:
            with torch.no_grad():
                emb = resnet(torch.stack(batch).to(device)).squeeze().cpu().numpy()
                features.append(emb)
    
    if len(features) > 0:
        all_features = np.concatenate(features)
        
        print("   Re-creating 50 micro-clusters...")
        kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_features)
        
        print("   Sorting images into severity folders...")
        for severity, cluster_ids in CLUSTER_MAPPING.items():
            dest_dir = SEARCH_DIR / "train" / severity
            dest_dir.mkdir(exist_ok=True)
            
            count = 0
            for path, label in zip(image_paths, labels):
                if label in cluster_ids:
                    shutil.move(str(path), str(dest_dir / path.name))
                    count += 1
            print(f"      -> {severity}: {count} images")
                    
        try: TRAIN_FIRE_DIR.rmdir() 
        except: pass
        print("✅ Labeling Complete.")
    else:
        print("⚠️ No features extracted. Folder might be empty.")

else:
    print("⚠️ 'train/fire' not found. Assuming dataset is already labeled/sorted.")

# ==========================================
# MODULE: TRAINING (ResNet-18 Only)
# ==========================================
print("\n🚀 Step 4: Training ResNet-18 (4 Classes)...")

# 1. Dataset Class
class WildfireDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), torch.tensor(label)
        except: return torch.zeros((3,224,224)), torch.tensor(0)

# 2. Scan & Create Lists
LABEL_MAP = {'nofire': 0, 'severity_low': 1, 'severity_medium': 2, 'severity_high': 3}
train_files, val_files = [], []

for split in ['train', 'val']:
    for class_name, label_idx in LABEL_MAP.items():
        path = SEARCH_DIR / split / class_name
        
        if path.exists():
            files = list(path.glob("*.*"))
            target_list = train_files if split == 'train' else val_files
            for f in files: target_list.append((str(f), label_idx))
        
        # Validation Proxy Logic
        elif class_name == 'severity_medium':
             fallback = SEARCH_DIR / split / 'fire'
             if fallback.exists():
                 files = list(fallback.glob("*.*"))
                 target_list = train_files if split == 'train' else val_files
                 for f in files: target_list.append((str(f), label_idx))

print(f"   Training Set: {len(train_files)} images")
print(f"   Validation Set: {len(val_files)} images")

if len(train_files) == 0:
    print("❌ Critical Error: No training images found. Check folder structure.")
    exit()

# 3. Transforms
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_ds = WildfireDataset(train_files, train_tf)
val_ds = WildfireDataset(val_files, val_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# 4. Model Setup (ResNet-18)
resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 4) 
resnet_model.to(device)

optimizer_resnet = AdamW(resnet_model.parameters(), lr=2e-5, weight_decay=0.01)
criterion_resnet = torch.nn.CrossEntropyLoss()

# 5. Training Loop
print("\n🔥 Starting ResNet-18 Training (5 Epochs)...")
for epoch in range(5):
    resnet_model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer_resnet.zero_grad()
        outputs = resnet_model(imgs)
        loss = criterion_resnet(outputs, lbls)
        loss.backward()
        optimizer_resnet.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    # Validation
    resnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = resnet_model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            
    print(f"   ✅ ResNet-18 Val Accuracy: {100 * correct / total:.2f}%")

# Save
torch.save(resnet_model.state_dict(), "./resnet18_wildfire_severity_colab.pth")
print("\n💾 ResNet-18 Model Saved to ./resnet18_wildfire_severity_colab.pth")