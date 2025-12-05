import os
import torch
import pandas as pd
from pathlib import Path
import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.optim import AdamW
from tqdm import tqdm

import argparse

# ================= CONFIGURATION =================
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'severity'], help='Training mode')
parser.add_argument('--model_type', type=str, default='vit', choices=['vit', 'resnet'], help='Model architecture')
args = parser.parse_args()

if args.mode == 'binary':
    SEARCH_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
    NUM_CLASSES = 2
    ID2LABEL = {0: 'No Fire', 1: 'Fire'}
    LABEL2ID = {'No Fire': 0, 'Fire': 1}
    CLASS_NAMES = ['fire', 'nofire']
else:
    SEARCH_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_severity")
    NUM_CLASSES = 3
    ID2LABEL = {0: 'Low', 1: 'Medium', 2: 'High'}
    LABEL2ID = {'Low': 0, 'Medium': 1, 'High': 2}
    CLASS_NAMES = ['low', 'medium', 'high']

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5
# =================================================


# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Auto-Find Images
print(f"\nScanning {SEARCH_DIR}...")

file_list = []

for root, dirs, files in os.walk(SEARCH_DIR):
    current_path = Path(root)
    folder_name = current_path.name.lower()
    
    if folder_name in ['train', 'test', 'val']:
        # Look specifically for class folders
        for class_name in CLASS_NAMES:
            target_path = current_path / class_name

            if target_path.exists():
                found_files = list(target_path.glob('*.jpg')) + \
                              list(target_path.glob('*.jpeg')) + \
                              list(target_path.glob('*.png'))
                
                for f in found_files:
                    file_list.append({
                        'path': str(f),
                        'split': folder_name,
                        'label': LABEL2ID.get(class_name.title(), LABEL2ID.get(class_name, 0))

                    })

df = pd.DataFrame(file_list)
if len(df) == 0:
    print("❌ No images found! Did you run reset_dataset.py?")
    exit()

print(f"Total Images: {len(df)}")
print(df.groupby(['split', 'label']).size())

# 3. Dataset Class
if args.model_type == 'vit':
    model_id = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_id)
else:
    # ResNet
    model_id = 'microsoft/resnet-18'
    # Use ViT processor for resizing/norm as it's standard 224x224
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

class WildfireDataset(Dataset):

    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['path']).convert("RGB")
            encoding = self.processor(images=image, return_tensors="pt")
            return encoding['pixel_values'].squeeze(0), torch.tensor(row['label'])
        except:
            return torch.zeros((3, 224, 224)), torch.tensor(0)

train_ds = WildfireDataset(df[df['split']=='train'], processor)
val_ds = WildfireDataset(df[df['split']=='val'], processor)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 4. Initialize Model
print(f"\nLoading {args.model_type.upper()} Model for {args.mode} task...")

if args.model_type == 'vit':
    model = ViTForImageClassification.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
else:
    from transformers import ResNetForImageClassification
    model = ResNetForImageClassification.from_pretrained(
        model_id,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
model.to(device)

# 5. Training Loop
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

print(f"\nStarting Training...")

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).logits, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"   Epoch {epoch+1} Val Acc: {100 * correct / total:.2f}%")

# Save
output_dir = f"./{args.model_type}_wildfire_{args.mode}"
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")