import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from pathlib import Path
import os

# Try importing Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Wildfire Watch | AI Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #FF4B4B; text-align: center; font-weight: 800;}
    .sub-header {font-size: 1.5rem; color: #333; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; color: #000000;}
    .stButton>button {width: 100%; font-weight: bold;}
    .report-box {background-color: #e8f4f8; padding: 20px; border-radius: 10px; border: 1px solid #b3d7ff; color: #000000;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/color/96/000000/fire-element.png", width=80)
st.sidebar.title("🔥 Wildfire Watch")
st.sidebar.markdown("**Group 8**\n\nAryan Ajmera\nRushabh Bhatt\nSoham\nDhrity")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Project Overview", "🧠 Methodology & Rigor", "📈 Performance Hub", "🤖 Live Demo & LLM"])

# --- DATA LOADING ---
@st.cache_data
def load_data():
    faithfulness = None
    if Path("faithfulness_results.csv").exists():
        try:
            faithfulness = pd.read_csv("faithfulness_results.csv")
            if 'score_drop_10' not in faithfulness.columns:
                 pass # Handle headerless if needed
        except: pass
        
    bench_local = None
    if Path("benchmark_report.csv").exists():
        bench_local = pd.read_csv("benchmark_report.csv")
        
    bench_colab = None
    if Path("benchmark_colab_report.csv").exists():
        bench_colab = pd.read_csv("benchmark_colab_report.csv")
        
    return faithfulness, bench_local, bench_colab

faithfulness_df, bench_local_df, bench_colab_df = load_data()

# --- 1. PROJECT OVERVIEW ---
if page == "🏠 Project Overview":
    st.markdown('<p class="main-header">Wildfire Smoke Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">From "Black Box" to Explainable, Robust AI</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 The Mission")
        st.info("""
        **Early detection saves lives.** 
        
        We built an AI system to detect wildfire smoke from aerial imagery. 
        But accuracy isn't enough—we needed **rigor**, **explainability**, and **robustness** for real-world deployment.
        """)
        
        st.markdown("### 🚀 The 'Pivot to Rigor' Journey")
        st.markdown("""
        1.  **Robust Explainability**: We proved our model isn't cheating. It looks at the smoke.
        2.  **Semi-Supervised Labeling**: We used AI to label AI, organizing data into severity levels.
        3.  **Comparative Benchmarking**: ViT vs ResNet-18.
        4.  **Robust Training**: Solved the "Smoke vs Cloud" problem with augmentation.
        """)

    with col2:
        if Path("gradcam_result.jpg").exists():
            st.image("gradcam_result.jpg", caption="Our Explainable AI in Action", use_container_width=True)
        else:
            st.warning("Run explain_wildfire.py to generate the cover image!")

# --- 2. METHODOLOGY & RIGOR ---
elif page == "🧠 Methodology & Rigor":
    st.title("🧠 Methodology & Rigor")
    
    tab1, tab2 = st.tabs(["Explainability (Perturbation)", "Semi-Supervised Labeling"])
    
    with tab1:
        st.header("How do we know the model is right?")
        st.markdown("""
        We use **Perturbation Testing** to quantitatively measure "Faithfulness".
        
        **The Logic:**
        1.  **Ask**: "Where are you looking?" (Grad-CAM heatmap).
        2.  **Perturb**: Mask the top X% of those pixels (turn them black).
        3.  **Measure**: Does the confidence drop?
        
        If the confidence **drops significantly**, the model was truly relying on those pixels. If it stays high, the explanation was fake.
        """)
        
        if faithfulness_df is not None:
            drops = []
            for pct in [10, 20, 30]:
                col_name = f'score_drop_{pct}'
                if col_name in faithfulness_df.columns:
                    avg = faithfulness_df[col_name].mean() * 100
                    drops.append({'Mask %': f'{pct}%', 'Confidence Drop': avg})
            
            if drops:
                drop_df = pd.DataFrame(drops)
                fig = px.bar(drop_df, x='Mask %', y='Confidence Drop', 
                             title="Faithfulness Test Results (Higher Drop = Better)",
                             color='Confidence Drop', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"**Result:** Masking just 10% of the 'hot' pixels causes a ~{drops[0]['Confidence Drop']:.1f}% drop in confidence. Our explanations are highly faithful!")
        else:
            st.warning("No faithfulness results found. Run `evaluate_explainability.py`.")

    with tab2:
        st.header("Cluster-Assist Labeling")
        st.markdown("""
        Labeling thousands of images by hand is slow. We used **Unsupervised Learning (K-Means)** to group images into 50 micro-clusters based on their visual features.
        
        This allowed us to label **entire clusters** at once (e.g., "Cluster 12 is High Severity").
        """)
        
        if Path("cluster_grid.jpg").exists():
            st.image("cluster_grid.jpg", caption="50 Micro-Clusters of Smoke", use_container_width=True)

# --- 3. PERFORMANCE HUB ---
elif page == "📈 Performance Hub":
    st.title("📈 Performance Analysis")
    
    # --- Training Curves ---
    st.subheader("1. Training Dynamics (Severity Task)")
    st.write("Training progress on Google Colab for the 4-class severity task.")
    
    epochs = [1, 2, 3, 4, 5]
    vit_acc = [84.86, 73.20, 79.40, 78.91, 80.89]
    resnet_acc = [70.22, 74.69, 74.44, 77.42, 77.42]
    
    fig_train = go.Figure()
    fig_train.add_trace(go.Scatter(x=epochs, y=vit_acc, mode='lines+markers', name='ViT (Colab)', line=dict(color='#FF4B4B', width=3)))
    fig_train.add_trace(go.Scatter(x=epochs, y=resnet_acc, mode='lines+markers', name='ResNet-18 (Colab)', line=dict(color='#1f77b4', width=3)))
    fig_train.update_layout(xaxis_title="Epoch", yaxis_title="Validation Accuracy (%)", template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_train, use_container_width=True)
    
    # --- Benchmark Comparison ---
    st.subheader("2. Model Benchmarking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Local Benchmark (3-Class)")
        if bench_local_df is not None:
            st.dataframe(bench_local_df[['Model', 'Accuracy', 'F1 Score (Macro)', 'Inference Time (ms/img)']].style.highlight_max(axis=0))
        else:
            st.warning("Run `benchmark.py` first.")
            
    with col2:
        st.markdown("#### Colab Benchmark (4-Class)")
        if bench_colab_df is not None:
            st.dataframe(bench_colab_df[['Model', 'Accuracy', 'F1 Score (Macro)', 'Inference Time (ms/img)']].style.highlight_max(axis=0))
        else:
            st.warning("Run `benchmark_colab.py` first.")

    # Combined Chart
    if bench_colab_df is not None:
        st.subheader("ViT vs ResNet-18 (4-Class Accuracy)")
        fig_bench = px.bar(bench_colab_df, x='Model', y='Accuracy', color='Model', text_auto='.2%', title="Final Test Set Accuracy")
        st.plotly_chart(fig_bench, use_container_width=True)

# --- 4. LIVE DEMO & LLM ---
elif page == "🤖 Live Demo & LLM":
    st.title("🤖 Live Model Serving & AI Assistant")
    
    # Groq API Key Input
    with st.sidebar:
        st.header("🔑 Groq API Key")
        
        # Auto-load from file if exists
        default_key = ""
        if Path("groq.txt").exists():
            try:
                default_key = Path("groq.txt").read_text().strip()
            except: pass
            
        groq_api_key = st.text_input("Enter Groq API Key", value=default_key, type="password")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader("Choose a wildfire image...", type=["jpg", "png", "jpeg"])
        
    with col2:
        # Model Loading (Cached)
        @st.cache_resource
        def load_model():
            model_path = "vit_wildfire_severity_model"
            try:
                model = ViTForImageClassification.from_pretrained(model_path)
                try:
                    processor = ViTImageProcessor.from_pretrained(model_path)
                except:
                    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
                return model, processor
            except Exception as e:
                return None, None

        model, processor = load_model()
        
        if uploaded_file is not None and model is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Inference
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
            # Labels
            labels = ['No Fire', 'Low Severity', 'Medium Severity', 'High Severity']
            pred_label = labels[pred_idx]
            confidence = probs[0][pred_idx].item() * 100
            
            # Display Prediction
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction: <span style="color:#FF4B4B">{pred_label}</span></h3>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Grad-CAM
            with st.expander("🔍 See Explanation (Grad-CAM)", expanded=True):
                try:
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
                    
                    c1, c2 = st.columns(2)
                    c1.image(image, caption="Original", use_container_width=True)
                    c2.image(visualization, caption="Grad-CAM Attention", use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Grad-CAM Error: {e}")
            
            # Groq LLM Integration
            st.markdown("### 🧠 AI Analysis (Groq)")
            if not GROQ_AVAILABLE:
                st.warning("Groq library not installed. Run `pip install groq`.")
            elif not groq_api_key:
                st.info("Enter your Groq API Key in the sidebar to generate an AI report.")
            else:
                if st.button("Generate Incident Report"):
                    with st.spinner("Consulting Groq LLM..."):
                        try:
                            client = Groq(api_key=groq_api_key)
                            
                            prompt = f"""
                            You are an expert wildfire analyst. A drone has detected a potential wildfire.
                            
                            **Detection Data:**
                            - **Severity Classification**: {pred_label}
                            - **Model Confidence**: {confidence:.2f}%
                            
                            **Task:**
                            Write a concise, professional incident report (max 150 words).
                            1. Assess the urgency based on the severity.
                            2. Recommend immediate actions for the response team (e.g., deploy water bombers, send ground crew, monitor).
                            3. Mention the confidence level to indicate certainty.
                            """
                            
                            completion = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7,
                                max_tokens=200,
                            )
                            
                            report = completion.choices[0].message.content
                            
                            st.markdown(f"""
                            <div class="report-box">
                                <h4>📋 Incident Report</h4>
                                {report}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Groq Error: {e}")
