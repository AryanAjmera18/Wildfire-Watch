# ==========================================
# 🔥 Wildfire Smoke Detection: Colab Pipeline (Fixed for Read-Only Error)
# Group 8: Aryan Ajmera, Rushabh Bhatt, Soham, Dhrity
# ==========================================

import os
import shutil
import json
from pathlib import Path

# --- STEP 1: SETUP KAGGLE API (Auto-Auth) ---
print("🚀 Step 1: Configuring Kaggle API...")

# 🔴 ACTION: Replace these with your actual details if not using kaggle.json file
os.environ['KAGGLE_USERNAME'] = "elmadafri" # Replace with your kaggle username
os.environ['KAGGLE_KEY'] = "5efee092e7ab4db72eac76af0a607969" # Your Key

# --- STEP 2: DOWNLOAD & PREPARE DATASET ---
print("\n🚀 Step 2: Downloading & Preparing Dataset...")

# Install kagglehub
try:
    import kagglehub
except ImportError:
    os.system('pip install -q kagglehub')
    import kagglehub

# 1. Download to Cache
try:
    print("   Downloading via KaggleHub...")
    cache_path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
    print(f"   ✅ Downloaded to cache: {cache_path}")
except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    exit()

# 2. Copy to Writable Directory (Fix for Read-Only Error)
# We need to move files to /content/ so we can reorganize them into severity folders
WRITABLE_DIR = Path("/content/wildfire_data")

if not WRITABLE_DIR.exists():
    print(f"   Copying data to writable folder {WRITABLE_DIR}...")
    # We copy the 'the_wildfire_dataset_2n_version' folder from cache to /content/
    # We need to find where the actual data folders (train/val/test) are inside the cache
    
    # Locate 'train' folder in cache
    cache_root = Path(cache_path)
    train_found = list(cache_root.rglob("train"))
    
    if not train_found:
        print("❌ Error: Could not find 'train' folder in downloaded cache.")
        exit()
        
    # The parent of 'train' is the root we want to copy
    source_root = train_found[0].parent
    
    # Copy to /content/wildfire_data
    shutil.copytree(source_root, WRITABLE_DIR)
    print("   ✅ Data copied to writable workspace.")
else:
    print("   ✅ Writable workspace already exists.")

# Set our working directory to the new writable path
SEARCH_DIR = WRITABLE_DIR

# --- STEP 3: INSTALL LIBRARIES ---
print("\n🚀 Step 3: Installing Deep Learning Libraries...")
os.system('pip install -q transformers datasets accelerate scikit-learn grad-cam') 

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, get_scheduler
from torch.optim import AdamW
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Compute Device: {device}")
if device.type != 'cuda':
    print("⚠️ WARNING: Running on CPU! Enable GPU in Runtime > Change Runtime Type.")

# ==========================================
# MODULE: CLUSTER-ASSIST (Semi-Supervised Labeling)
# ==========================================
print("\n🚀 Step 4: Applying Severity Labels (Cluster-Assist)...")

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
            # Use SEARCH_DIR which points to /content/wildfire_data (Writable!)
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
# MODULE: TRAINING (Vision Transformer)
# ==========================================
print("\n🚀 Step 5: Training Vision Transformer (4 Classes)...")

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

# 4. Model Setup
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=4,
    id2label={0:'None', 1:'Low', 2:'Med', 3:'High'},
    label2id={'None':0, 'Low':1, 'Med':2, 'High':3}
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 5. Training Loop
print("\n🔥 Starting Training (5 Epochs)...")
for epoch in range(5):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs).logits, lbls)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = torch.argmax(model(imgs).logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            
    print(f"   ✅ Val Accuracy: {100 * correct / total:.2f}%")

# Save
model.save_pretrained("./vit_wildfire_severity_colab")
print("\n💾 Model Saved to ./vit_wildfire_severity_colab")