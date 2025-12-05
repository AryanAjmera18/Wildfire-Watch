import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
import os
import random
import sys
import traceback

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reshape_transform(tensor, height=14, width=14):
    # Result of ViT is [batch, seq_len, hidden_dim]
    # seq_len is 197 (1 cls + 196 patches) for 224x224 input with patch size 16
    # We discard the CLS token (index 0) and reshape the rest
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Bring the channels to the first dimension,
    # like in CNNs: [batch, channels, height, width]
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def run_explainability():
    model_path = "./vit_wildfire_binary"
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Please run train_wildfire.py first.")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        processor = ViTImageProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Find a random image from the test set
    # Using the same search logic as train_wildfire.py but looking for any image
    search_dir = Path(r"C:\NYU\ACV\the_wildfire_dataset_2n_version")
    print(f"Scanning {search_dir} for images...")
    
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(search_dir.rglob(ext)))
    
    if not test_images:
        print("❌ No images found to explain!")
        return
        
    image_path = random.choice(test_images)
    print(f"Explaining image: {image_path}")
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Failed to open image: {e}")
        return

    # Resize for visualization (Grad-CAM utils expect float32 in [0, 1])
    image_resized = image.resize((224, 224))
    rgb_img = np.float32(image_resized) / 255
    
    # Preprocess for model
    inputs = processor(images=image, return_tensors="pt")
    input_tensor = inputs['pixel_values'].to(device)
    
    # Target the last layer of the last block in the encoder
    # For ViT-Base, this is usually model.vit.encoder.layer[-1].layernorm_before
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]

    # Initialize Grad-CAM
    try:
        # Wrap model to return logits
        wrapped_model = ModelWrapper(model)
        cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    except Exception as e:
        print(f"❌ Failed to initialize GradCAM: {e}")
        return

    # Run inference to get prediction
    outputs = model(input_tensor)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    print(f"Predicted class: {predicted_label}")
    
    # Generate CAM for the predicted class
    targets = [ClassifierOutputTarget(predicted_class_idx)]

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Create side-by-side comparison
        # rgb_img is float32 [0, 1], convert to uint8 [0, 255]
        original_uint8 = np.uint8(255 * rgb_img)
        
        # Concatenate horizontally
        combined_img = np.hstack((original_uint8, visualization))
        
        # Save result
        save_path = "gradcam_result.jpg"
        # Convert RGB to BGR for cv2 saving
        cv2.imwrite(save_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM visualization to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Failed to generate or save CAM: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_explainability()
