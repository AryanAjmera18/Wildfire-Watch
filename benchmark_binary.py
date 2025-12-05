import torch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from PIL import Image
import os

# Configuration
MODEL_PATH = "./vit_wildfire_binary_robust"
DATASET_DIR = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
OUTPUT_FILE = "benchmark_binary_report.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_model():
    print(f"Benchmarking Binary Model from {MODEL_PATH}...")
    
    # Load Model
    try:
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Find Test Images
    test_images = []
    labels = []
    
    # 0: No Fire, 1: Fire
    for class_name, label in [('nofire', 0), ('fire', 1)]:
        folder = DATASET_DIR / 'test' / class_name
        if folder.exists():
            files = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            test_images.extend(files)
            labels.extend([label] * len(files))
            
    if not test_images:
        print("No test images found!")
        return

    print(f"Found {len(test_images)} test images.")

    # Inference Loop
    predictions = []
    latencies = []
    
    print("Running Inference...")
    with torch.no_grad():
        for img_path in tqdm(test_images):
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                
                start_time = time.time()
                outputs = model(**inputs)
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000) # ms
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                predictions.append(-1) # Error flag

    # Metrics
    # Filter out errors
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    y_true = [labels[i] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    avg_latency = np.mean(latencies)
    params = count_parameters(model)
    
    print(f"\nResults:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Params: {params:.2f} M")
    
    # Save to CSV
    df = pd.DataFrame([{
        'Model': 'ViT Binary (Robust)',
        'Accuracy': acc,
        'F1 Score (Macro)': f1,
        'Inference Time (ms/img)': avg_latency,
        'Parameters (M)': params
    }])
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    benchmark_model()
