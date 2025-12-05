import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np

# Configuration
DATASET_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
OUTPUT_FILE = "augmentation_demo.jpg"

# Define Augmentations (Matching train_binary_augmented.py)
# We separate PIL transforms and Tensor transforms for visualization
pil_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

def apply_random_erasing(image_tensor):
    # Random Erasing expects a tensor [C, H, W]
    if torch.rand(1) < 0.5: # Force higher probability for demo visibility
        eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15), value='random')
        return eraser(image_tensor)
    return image_tensor

def visualize():
    print("Generating Augmentation Demo...")
    
    # Find Fire Images
    fire_dir = DATASET_DIR / 'train' / 'fire'
    if not fire_dir.exists():
        print("Dataset not found!")
        return

    images = list(fire_dir.glob('*.jpg')) + list(fire_dir.glob('*.png'))
    if not images:
        print("No images found.")
        return

    # Select 4 random images
    selected_paths = random.sample(images, 4)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # Row 1: Original
    # Row 2: Augmented
    
    for i, img_path in enumerate(selected_paths):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224)) # Standardize size
        
        # 1. Original
        axes[0, i].imshow(img)
        axes[0, i].set_title("Original", fontsize=12)
        axes[0, i].axis('off')
        
        # 2. Augmented
        # Apply PIL transforms
        aug_img = pil_transforms(img)
        
        # Convert to Tensor for Random Erasing
        aug_tensor = T.ToTensor()(aug_img)
        
        # Apply Random Erasing
        aug_tensor = apply_random_erasing(aug_tensor)
        
        # Convert back to Numpy for plotting [H, W, C]
        aug_np = aug_tensor.permute(1, 2, 0).numpy()
        
        axes[1, i].imshow(aug_np)
        axes[1, i].set_title("Augmented\n(Jitter+Rotate+Erase)", fontsize=12, color='red')
        axes[1, i].axis('off')

    plt.suptitle("Robust Training: Data Augmentation Pipeline", fontsize=20, weight='bold')
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=150)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    visualize()
