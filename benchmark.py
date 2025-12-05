import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import ViTForImageClassification, ResNetForImageClassification, ViTImageProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WildfireDataset(Dataset):
    def __init__(self, search_dir, processor, class_names):
        self.processor = processor
        self.file_list = []
        self.label2id = {name: i for i, name in enumerate(class_names)}
        
        for class_name in class_names:
            target_path = search_dir / class_name
            if target_path.exists():
                files = list(target_path.glob('*.jpg')) + list(target_path.glob('*.png'))
                for f in files:
                    self.file_list.append({'path': f, 'label': self.label2id[class_name]})
                    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]
        try:
            image = Image.open(item['path']).convert("RGB")
            encoding = self.processor(images=image, return_tensors="pt")
            return encoding['pixel_values'].squeeze(0), torch.tensor(item['label'])
        except:
            return torch.zeros((3, 224, 224)), torch.tensor(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_path, model_type, test_dir, class_names):
    print(f"Evaluating {model_type} from {model_path}...")
    
    try:
        if model_type == 'vit':
            model = ViTForImageClassification.from_pretrained(model_path)
            processor = ViTImageProcessor.from_pretrained(model_path)
        else:
            model = ResNetForImageClassification.from_pretrained(model_path)
            # ResNet uses same processor in our training script
            processor = ViTImageProcessor.from_pretrained(model_path)
            
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # Load Test Data
    test_ds = WildfireDataset(test_dir, processor, class_names)
    if len(test_ds) == 0:
        print("No test images found.")
        return None
        
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            # Calculate time per image in ms
            batch_time = (end_time - start_time) * 1000
            inference_times.append(batch_time / images.size(0))
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_inference_time = np.mean(inference_times)
    params = count_parameters(model)
    
    return {
        'Model': model_type.upper(),
        'Accuracy': acc,
        'F1 Score (Macro)': f1,
        'Inference Time (ms/img)': avg_inference_time,
        'Parameters (M)': params / 1e6
    }

def run_benchmark():
    results = []
    
    # 1. Evaluate ViT Severity
    vit_path = "./vit_wildfire_severity"
    test_dir = Path(r"C:\NYU\ACV\the_wildfire_dataset_severity\test")
    class_names = ['low', 'medium', 'high']
    
    if Path(vit_path).exists():
        res = evaluate_model(vit_path, 'vit', test_dir, class_names)
        if res:
            results.append(res)
            
    # 2. Evaluate ResNet Severity (if exists)
    resnet_path = "./resnet_wildfire_severity"
    if Path(resnet_path).exists():
        res = evaluate_model(resnet_path, 'resnet', test_dir, class_names)
        if res:
            results.append(res)
            
    if not results:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results)
    print("\n=== Benchmark Report ===")
    print(df.to_string(index=False))
    df.to_csv("benchmark_report.csv", index=False)
    print("\nSaved to benchmark_report.csv")

if __name__ == "__main__":
    run_benchmark()
