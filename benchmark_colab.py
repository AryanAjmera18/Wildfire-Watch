import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn

# --- CONFIGURATION ---
TEST_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_severity_4class\test")
RESNET_PATH = Path(r"C:\NYU\ACV\resnet18_wildfire_severity_colab.pth")
VIT_PATH = Path(r"C:\NYU\ACV\vit_wildfire_severity_model")

# Class Mapping (Must match training script)
# LABEL_MAP = {'nofire': 0, 'severity_low': 1, 'severity_medium': 2, 'severity_high': 3}
# But our folders are named: 'nofire', 'severity_low', 'severity_medium', 'severity_high'
CLASS_NAMES = ['nofire', 'severity_low', 'severity_medium', 'severity_high']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class WildfireDataset(Dataset):
    def __init__(self, root_dir, transform, class_names):
        self.transform = transform
        self.file_list = []
        self.label2id = {name: i for i, name in enumerate(class_names)}
        
        for class_name in class_names:
            target_path = root_dir / class_name
            if target_path.exists():
                files = list(target_path.glob('*.jpg')) + list(target_path.glob('*.png'))
                for f in files:
                    self.file_list.append({'path': f, 'label': self.label2id[class_name]})
            else:
                print(f"Warning: Class folder {class_name} not found in {root_dir}")
                    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        try:
            image = Image.open(item['path']).convert("RGB")
            if self.transform:
                return self.transform(image), torch.tensor(item['label'])
            return image, torch.tensor(item['label']) # Should not happen if transform provided
        except:
            return torch.zeros((3, 224, 224)), torch.tensor(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, model_name, test_loader):
    print(f"Evaluating {model_name}...")
    model.eval()
    
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            
            start_time = time.time()
            if 'ViT' in model_name:
                outputs = model(images).logits
            else:
                outputs = model(images)
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000
            inference_times.append(batch_time / images.size(0))
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_inference_time = np.mean(inference_times)
    params = count_parameters(model)
    
    return {
        'Model': model_name,
        'Accuracy': acc,
        'F1 Score (Macro)': f1,
        'Inference Time (ms/img)': avg_inference_time,
        'Parameters (M)': params / 1e6
    }

def main():
    results = []
    
    # --- 1. Load ResNet-18 ---
    if RESNET_PATH.exists():
        print(f"Loading ResNet-18 from {RESNET_PATH}...")
        resnet = models.resnet18(weights=None) # No weights needed, loading state_dict
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 4)
        resnet.load_state_dict(torch.load(RESNET_PATH, map_location=device))
        resnet.to(device)
        
        # ResNet Transform
        resnet_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # Matches training script
        ])
        
        test_ds_resnet = WildfireDataset(TEST_DIR, resnet_tf, CLASS_NAMES)
        test_loader_resnet = DataLoader(test_ds_resnet, batch_size=32, shuffle=False, num_workers=0)
        
        res = evaluate_model(resnet, 'ResNet-18 (Colab)', test_loader_resnet)
        results.append(res)
    else:
        print(f"ResNet model not found at {RESNET_PATH}")

    # --- 2. Load ViT ---
    if VIT_PATH.exists():
        print(f"Loading ViT from {VIT_PATH}...")
        try:
            vit = ViTForImageClassification.from_pretrained(VIT_PATH)
            vit.to(device)
            
            try:
                processor = ViTImageProcessor.from_pretrained(VIT_PATH)
            except:
                print("Warning: Local processor not found. Using default 'google/vit-base-patch16-224-in21k'.")
                processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            
            # ViT Transform (Using Processor)
            # We need a custom transform wrapper because Dataset expects a callable
            class ViTTransform:
                def __init__(self, processor):
                    self.processor = processor
                def __call__(self, img):
                    return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            
            vit_tf = ViTTransform(processor)
            
            test_ds_vit = WildfireDataset(TEST_DIR, vit_tf, CLASS_NAMES)
            test_loader_vit = DataLoader(test_ds_vit, batch_size=32, shuffle=False, num_workers=0)
            
            res = evaluate_model(vit, 'ViT (Colab)', test_loader_vit)
            results.append(res)
        except Exception as e:
            print(f"Failed to load ViT: {e}")
    else:
        print(f"ViT model not found at {VIT_PATH}")

    if results:
        df = pd.DataFrame(results)
        print("\n=== Colab Benchmark Report ===")
        print(df.to_string(index=False))
        df.to_csv("benchmark_colab_report.csv", index=False)
        print("\nSaved to benchmark_colab_report.csv")
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()
