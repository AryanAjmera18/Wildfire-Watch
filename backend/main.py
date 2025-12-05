from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from transformers import ViTForImageClassification, ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

app = FastAPI(title="Wildfire Watch API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files (for images like cluster_grid.jpg)
app.mount("/static", StaticFiles(directory=".."), name="static")

# ... (global state)

# --- STARTUP ---
@app.on_event("startup")
async def load_models():
    print("Loading Models...")
    try:
        # Load Severity Model
        severity_path = "vit_wildfire_severity_model"
        models['severity'] = ViTForImageClassification.from_pretrained(severity_path)
        try:
            processors['severity'] = ViTImageProcessor.from_pretrained(severity_path)
        except:
            print("Warning: Could not load local processor for severity model. Using default.")
            processors['severity'] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            
        labels_map['severity'] = ['No Fire', 'Low Severity', 'Medium Severity', 'High Severity']
        
        # Load Binary Model
        binary_path = "vit_wildfire_binary_robust"
        models['binary'] = ViTForImageClassification.from_pretrained(binary_path)
        processors['binary'] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        labels_map['binary'] = ['No Fire', 'Fire']
        
        print("Models Loaded Successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

# ... (utils)

# --- ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "active", "models_loaded": list(models.keys())}

@app.get("/stats")
def get_stats():
    stats = {}
    
    # Faithfulness
    if os.path.exists("../faithfulness_results.csv"):
        df = pd.read_csv("../faithfulness_results.csv")
        if 'score_drop_10' in df.columns:
            stats['faithfulness'] = [
                {'name': '10%', 'drop': df['score_drop_10'].mean() * 100},
                {'name': '20%', 'drop': df['score_drop_20'].mean() * 100},
                {'name': '30%', 'drop': df['score_drop_30'].mean() * 100},
            ]
            
    # Benchmarks
    if os.path.exists("../benchmark_colab_report.csv"):
        df = pd.read_csv("../benchmark_colab_report.csv")
        stats['benchmark'] = df[['Model', 'Accuracy', 'F1 Score (Macro)']].to_dict(orient='records')
        
    return stats

@app.post("/predict")
# ... (rest of predict)
async def predict(file: UploadFile = File(...), mode: str = Form("binary")):
    if mode not in models:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose 'binary' or 'severity'.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    model = models[mode]
    processor = processors[mode]
    labels = labels_map[mode]
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    
    label = labels[pred_idx]
    confidence = probs[0][pred_idx].item() * 100
    
    return {
        "label": label, 
        "confidence": confidence,
        "model_used": mode,
        "raw_probs": probs[0].tolist()
    }

@app.post("/explain")
async def explain(file: UploadFile = File(...), mode: str = Form("binary")):
    if mode not in models:
        raise HTTPException(status_code=400, detail="Invalid mode.")
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    model = models[mode]
    processor = processors[mode]
    
    # Grad-CAM Logic
    inputs = processor(images=image, return_tensors="pt")
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x).logits

    wrapped_model = ModelWrapper(model)
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]

    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=inputs['pixel_values'], targets=None)[0, :]
    
    img_np = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    vis_img = Image.fromarray(visualization)
    
    return {
        "original": img_to_base64(image.resize((224, 224))),
        "gradcam": img_to_base64(vis_img),
        "mode": mode
    }

@app.post("/report")
async def generate_report(request: dict):
    # Expects JSON body: { "model_choice": "...", "label": "...", "confidence": ... }
    # Using dict for simplicity, or define Pydantic model
    
    try:
        # Load API Key from file or env
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key and os.path.exists("../groq.txt"):
            with open("../groq.txt", "r") as f:
                api_key = f.read().strip()
                
        if not api_key:
            raise HTTPException(status_code=500, detail="Groq API Key not found.")
            
        client = Groq(api_key=api_key)
        
        mode = request.get("model_choice", "binary")
        label = request.get("label", "Unknown")
        confidence = request.get("confidence", 0.0)
        
        if mode == "severity":
            task_desc = "Assess urgency based on severity and recommend actions."
        else:
            task_desc = "Confirm presence/absence of fire and recommend verification."
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prompt = f"""
        You are an expert wildfire analyst. A drone has processed an image.
        
        **Metadata:**
        - **Date/Time**: {timestamp}
        - **Location**: Unknown (Drone Telemetry Unavailable)
        
        **Detection Data:**
        - **Mode**: {mode}
        - **Result**: {label}
        - **Confidence**: {confidence:.2f}%
        
        **Task:**
        Write a concise, professional incident report (max 150 words).
        1. Start with the Date/Time provided.
        2. {task_desc}
        3. Recommend immediate next steps.
        4. Mention the confidence level.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        
        return {"report": completion.choices[0].message.content}
        
    except Exception as e:
        print(f"Groq Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
