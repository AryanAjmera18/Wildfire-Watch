import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import sys

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reusing components from explain_wildfire.py ---
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result
# ---------------------------------------------------

def perturb_image(image_tensor, heatmap, mask_percentile):
    """
    Masks the top mask_percentile% of pixels in the image based on the heatmap.
    """
    # Flatten heatmap to find threshold
    heatmap_flat = heatmap.flatten()
    threshold = np.percentile(heatmap_flat, 100 - mask_percentile)
    
    # Create mask (1 where heatmap >= threshold, 0 otherwise)
    mask = heatmap >= threshold
    
    # Expand mask to match image channels [3, H, W]
    mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
    
    # Apply mask (set masked pixels to 0)
    # image_tensor is [1, 3, H, W], we need [3, H, W]
    perturbed_image = image_tensor.clone().squeeze(0).cpu().numpy()
    perturbed_image[mask] = 0 # Mask with black
    
    return torch.tensor(perturbed_image).unsqueeze(0).to(device)

def evaluate_faithfulness(num_samples=50):
    model_path = "./vit_wildfire_binary"
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        processor = ViTImageProcessor.from_pretrained(model_path)
        wrapped_model = ModelWrapper(model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Setup Grad-CAM
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Find 'Fire' images
    search_dir = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
    print(f"Scanning {search_dir} for Fire images...")
    
    fire_images = []
    # Look in val/test folders for 'fire' class
    for split in ['val', 'test']:
        fire_dir = search_dir / split / 'fire'
        if fire_dir.exists():
            fire_images.extend(list(fire_dir.glob('*.jpg')) + list(fire_dir.glob('*.png')))
    
    if not fire_images:
        print("❌ No Fire images found!")
        return

    # Select random subset
    if len(fire_images) > num_samples:
        selected_images = np.random.choice(fire_images, num_samples, replace=False)
    else:
        selected_images = fire_images
    
    print(f"Evaluating on {len(selected_images)} images...")

    results = []
    
    for img_path in tqdm(selected_images):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            input_tensor = inputs['pixel_values'].to(device)
            
            # 1. Get Original Score
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                orig_score = probs[0, 1].item() # Score for 'Fire' class (index 1)
            
            # Skip if model doesn't predict Fire confidently (e.g. < 0.5)
            if orig_score < 0.5:
                continue

            # 2. Generate Heatmap
            targets = [ClassifierOutputTarget(1)] # Target 'Fire' class
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            heatmap = grayscale_cam[0, :]
            
            # 3. Perturb and Re-evaluate
            row = {'image': str(img_path), 'original_score': orig_score}
            
            for pct in [10, 20, 30]:
                perturbed_input = perturb_image(input_tensor, heatmap, pct)
                
                with torch.no_grad():
                    p_outputs = model(perturbed_input)
                    p_probs = torch.nn.functional.softmax(p_outputs.logits, dim=-1)
                    p_score = p_probs[0, 1].item()
                
                drop = (orig_score - p_score) / orig_score
                row[f'score_drop_{pct}'] = drop
                row[f'masked_score_{pct}'] = p_score
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save Results
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No valid results generated.")
        return

    df.to_csv("faithfulness_results.csv", index=False)
    print(f"\nResults saved to faithfulness_results.csv")
    
    # Print Summary Table
    print("\n=== Faithfulness Test Results (Average % Drop in Confidence) ===")
    print(f"{'Mask %':<10} | {'Avg Drop':<10} | {'Interpretation'}")
    print("-" * 45)
    for pct in [10, 20, 30]:
        avg_drop = df[f'score_drop_{pct}'].mean() * 100
        print(f"{pct:<10} | {avg_drop:.2f}%     | {'Good' if avg_drop > 0 else 'Bad'}")
    print("-" * 45)
    print("Interpretation: A positive drop means the model relies on the highlighted regions.")

if __name__ == "__main__":
    evaluate_faithfulness()
