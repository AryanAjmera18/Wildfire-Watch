import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import shutil

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, processor, image_paths, batch_size=32):
    features = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        valid_paths = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except:
                continue
        
        if not images:
            continue
            
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the CLS token (first token) as the image representation
            # Shape: [batch_size, hidden_dim]
            batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(batch_features)
            
    return np.vstack(features)

def create_cluster_grid(image_paths, labels, n_clusters, output_file="cluster_grid.jpg"):
    # Find representative images (closest to center would be ideal, but random is fast/okay)
    # Here we just pick the first image found for each cluster
    
    representatives = {}
    for img_path, label in zip(image_paths, labels):
        if label not in representatives:
            representatives[label] = img_path
        if len(representatives) == n_clusters:
            break
            
    # Create a grid
    cols = 10
    rows = math.ceil(n_clusters / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))
    axes = axes.flatten()
    
    sorted_clusters = sorted(representatives.keys())
    
    for i, ax in enumerate(axes):
        if i < len(sorted_clusters):
            cluster_id = sorted_clusters[i]
            img_path = representatives[cluster_id]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"Cluster {cluster_id}")
                ax.axis('off')
            except:
                ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved cluster grid to {os.path.abspath(output_file)}")
    plt.close()

def run_clustering():
    model_path = "./vit_wildfire_binary"
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Load as base ViTModel to get features, not classification logits
        model = ViTModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        processor = ViTImageProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Find 'Fire' images
    search_dir = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
    print(f"Scanning {search_dir} for Fire images...")
    
    fire_images = []
    # Look in all folders for 'fire' class
    for root, dirs, files in os.walk(search_dir):
        if 'fire' in Path(root).name.lower() and 'nofire' not in Path(root).name.lower():
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fire_images.append(os.path.join(root, f))
    
    if not fire_images:
        print("No Fire images found!")
        return
    
    print(f"Found {len(fire_images)} Fire images. Extracting features...")
    
    # Extract features
    features = extract_features(model, processor, fire_images)
    
    # Clustering
    n_clusters = 50
    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Generate Grid
    create_cluster_grid(fire_images, labels, n_clusters)
    
    # Save a mapping CSV for all images
    print("Saving cluster assignments to image_clusters.csv...")
    import pandas as pd
    df = pd.DataFrame({'path': fire_images, 'cluster': labels})
    df.to_csv("image_clusters.csv", index=False)
    print(f"Saved {len(df)} assignments to image_clusters.csv")

if __name__ == "__main__":
    run_clustering()
