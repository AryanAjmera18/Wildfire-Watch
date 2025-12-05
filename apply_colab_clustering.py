import os
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision import models, transforms
import pandas as pd

# --- CONFIGURATION ---
SOURCE_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
TARGET_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_severity_4class")

# Cluster Mapping from Colab Script
CLUSTER_MAPPING = {
    "severity_high": [4, 5, 8, 12, 17, 18, 21, 23, 24, 25, 27, 31, 34, 46],
    "severity_medium": [9, 11, 13, 14, 19, 22, 28, 29, 30, 33, 35, 38, 39, 41, 42, 45, 47, 49],
    "severity_low": [0, 1, 2, 3, 6, 7, 10, 15, 16, 20, 26, 32, 36, 37, 40, 43, 44, 48]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def get_severity(cluster_id):
    for severity, clusters in CLUSTER_MAPPING.items():
        if cluster_id in clusters:
            return severity
    return None

def extract_features(model, image_paths, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    features = []
    valid_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i+batch_size]
        batch = []
        current_batch_paths = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(transform(img))
                current_batch_paths.append(p)
            except Exception as e:
                print(f"Error loading {p}: {e}")
        
        if batch:
            with torch.no_grad():
                input_tensor = torch.stack(batch).to(device)
                emb = model(input_tensor).squeeze().cpu().numpy()
                if len(emb.shape) == 1: # Handle single item batch
                    emb = np.expand_dims(emb, axis=0)
                features.append(emb)
                valid_paths.extend(current_batch_paths)
                
    if features:
        return np.concatenate(features), valid_paths
    return np.array([]), []

def main():
    if TARGET_DIR.exists():
        print(f"Warning: Target directory {TARGET_DIR} already exists.")
        # shutil.rmtree(TARGET_DIR) # Uncomment to force clean
    
    # 1. Setup Feature Extractor (ResNet18)
    print("Loading ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) # Remove FC layer
    resnet.to(device).eval()
    
    # 2. Fit K-Means on TRAIN Fire images
    print("Scanning for TRAIN Fire images...")
    train_fire_dir = SOURCE_DIR / "train" / "fire"
    train_paths = list(train_fire_dir.glob("*.jpg")) + list(train_fire_dir.glob("*.png"))
    
    print(f"Found {len(train_paths)} training images.")
    train_features, train_valid_paths = extract_features(resnet, train_paths)
    
    print("Fitting K-Means (K=50)...")
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    kmeans.fit(train_features)
    
    # 3. Process All Splits (Train, Val, Test)
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        # A. Handle NoFire (Copy as is)
        nofire_src = SOURCE_DIR / split / "nofire"
        nofire_dst = TARGET_DIR / split / "nofire"
        nofire_dst.mkdir(parents=True, exist_ok=True)
        
        if nofire_src.exists():
            for img_path in nofire_src.glob("*.*"):
                try:
                    shutil.copy2(img_path, nofire_dst / img_path.name)
                except: pass
        
        # B. Handle Fire (Predict Cluster -> Map to Severity)
        fire_src = SOURCE_DIR / split / "fire"
        if not fire_src.exists():
            continue
            
        fire_paths = list(fire_src.glob("*.jpg")) + list(fire_src.glob("*.png"))
        if not fire_paths:
            continue
            
        # Extract features for this split
        features, valid_paths = extract_features(resnet, fire_paths)
        
        # Predict clusters
        labels = kmeans.predict(features)
        
        # Copy to appropriate folders
        for img_path, cluster_id in zip(valid_paths, labels):
            severity = get_severity(cluster_id)
            if severity:
                # Map 'severity_low' -> 'low' for cleaner folder names if desired, 
                # but user used 'severity_low' in script. Let's stick to user's folder names 
                # from the script: 'severity_low', 'severity_medium', 'severity_high'.
                # Actually, looking at train_resent18_colab.py, they used:
                # dest_dir = SEARCH_DIR / "train" / severity
                # where severity is "severity_low", etc.
                
                # However, for the benchmark script later, simpler names might be better?
                # The user's script uses: LABEL_MAP = {'nofire': 0, 'severity_low': 1, ...}
                # So we MUST use 'severity_low', 'severity_medium', 'severity_high'.
                
                dest_folder = TARGET_DIR / split / severity
                dest_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_folder / Path(img_path).name)

    print(f"\n✅ Dataset created at {TARGET_DIR}")

if __name__ == "__main__":
    main()
