import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms as T
import sys

# ================= CONFIGURATION =================
# Path to your dataset
SEARCH_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version") 

BATCH_SIZE = 32
EPOCHS = 10  # Increased for better convergence
LEARNING_RATE = 5e-5
NUM_CLASSES = 2
# =================================================

# 1. STRICT GPU ENFORCEMENT
if not torch.cuda.is_available():
    print("CRITICAL ERROR: No GPU detected!")
    print("This script requires a CUDA-capable GPU (like your RTX 4060) to run efficiently.")
    print("Aborting to prevent slow CPU training.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

# 2. Auto-Find Images (Binary Logic)
print(f"\nScanning {SEARCH_DIR}...")
file_list = []

for root, dirs, files in os.walk(SEARCH_DIR):
    current_path = Path(root)
    folder_name = current_path.name.lower()
    
    if folder_name in ['train', 'test', 'val']:
        for class_name in ['fire', 'nofire']:
            target_path = current_path / class_name
            if target_path.exists():
                found_files = list(target_path.glob('*.jpg')) + \
                              list(target_path.glob('*.jpeg')) + \
                              list(target_path.glob('*.png'))
                
                for f in found_files:
                    file_list.append({
                        'path': str(f),
                        'split': folder_name,
                        'label': 1 if class_name == 'fire' else 0
                    })

df = pd.DataFrame(file_list)
if len(df) == 0:
    print("Error: No images found! Did you run reset_dataset.py?")
    sys.exit(1)

print(f"Total Images: {len(df)}")
print(df.groupby(['split', 'label']).size())

# 3. Data Augmentation & Dataset
model_id = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_id)

# Define Augmentations
# We apply these ONLY to the training set
train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)), # Cut out small rectangles
    # Note: We need to normalize using ViT mean/std, but processor does this usually.
    # However, mixing T.Compose with processor is tricky.
    # Strategy: We will use the processor for resizing and normalization, 
    # but apply geometric/color transforms on the PIL image BEFORE processor.
])

class WildfireDataset(Dataset):
    def __init__(self, dataframe, processor, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.augment = augment
        
        # Augmentations that work on PIL images
        self.pil_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        
        # Random Erasing works on Tensor, so we handle it separately if needed,
        # but for simplicity with ViTProcessor, we'll stick to PIL transforms 
        # or apply Erasing on the pixel_values tensor.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['path']).convert("RGB")
            
            # Apply Augmentations
            if self.augment:
                image = self.pil_transforms(image)
            
            # Use Processor (Resizing + Normalization)
            encoding = self.processor(images=image, return_tensors="pt")
            pixel_values = encoding['pixel_values'].squeeze(0)
            
            # Apply Random Erasing (works on Tensor)
            if self.augment:
                if torch.rand(1) < 0.2: # 20% chance
                    eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15))
                    pixel_values = eraser(pixel_values)

            return pixel_values, torch.tensor(row['label'])
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(0)

train_ds = WildfireDataset(df[df['split']=='train'], processor, augment=True) # Augmentation ON
val_ds = WildfireDataset(df[df['split']=='val'], processor, augment=False)   # Augmentation OFF

# Pin Memory for faster GPU transfer
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 4. Initialize Model
print("\nLoading Binary ViT Model...")
model = ViTForImageClassification.from_pretrained(
    model_id,
    num_labels=2,
    id2label={0: 'No Fire', 1: 'Fire'},
    label2id={'No Fire': 0, 'Fire': 1}
)
model.to(device)

# 5. Training Loop with Scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # Smooth decay
criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
output_dir = "./vit_wildfire_binary_robust"

print(f"\nStarting Robust Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = 0.0
    for images, labels in loop:
        # Non-blocking transfer
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    # Step Scheduler
    scheduler.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"   Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
    
    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"   New Best Model Saved! ({val_acc:.2f}%)")

print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Model saved to {output_dir}")
