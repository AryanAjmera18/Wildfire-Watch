# 🔥 Wildfire Watch: AI-Powered Aerial Surveillance

**Wildfire Watch** is an advanced AI system designed to detect and analyze wildfire smoke from aerial imagery. It goes beyond simple "black box" detection by incorporating **Explainable AI (XAI)**, **Robust Training**, and **Generative AI Reporting** to provide actionable intelligence for emergency response.

![Dashboard Preview](gradcam_result.jpg)

## 🚀 Key Features

### 1. Dual-Mode Detection
*   **Binary Detection**: High-speed "Fire vs. No Fire" classification for rapid initial screening.
*   **Severity Classification**: Detailed analysis categorizing smoke into **Low**, **Medium**, and **High** severity levels.

### 2. Explainable AI (Grad-CAM)
*   **Visual Trust**: See exactly *where* the model is looking. The dashboard overlays a heatmap on the image, highlighting the pixels that influenced the decision.
*   **Faithfulness Metrics**: We quantitatively measure the reliability of these explanations using perturbation testing (masking important pixels and measuring confidence drops).

### 3. Intelligent Reporting (LLM Integration)
*   **Groq API Integration**: The system uses the **Llama 3.1** Large Language Model to generate professional incident reports.
*   **Context-Aware**: Reports include the detection result, confidence score, timestamp, and specific recommendations based on the severity level.

### 4. Robust Performance
*   **Data Augmentation**: Trained on datasets enhanced with rotation, color jitter, and random erasing to handle diverse lighting and weather conditions.
*   **Benchmarking**: Includes a "Performance Hub" comparing our Vision Transformer (ViT) against ResNet-18, complete with accuracy and F1 score metrics.

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/AryanAjmera18/Wildfire-Watch.git
    cd Wildfire-Watch
    ```

2.  **Install Dependencies**:
    ```bash
    pip install streamlit pandas plotly torch torchvision transformers pytorch-grad-cam opencv-python groq
    ```

3.  **Set up Groq API (Optional)**:
    *   Create a file named `groq.txt` in the root directory.
    *   Paste your Groq API key inside.
    *   *Alternatively, enter the key in the dashboard sidebar.*

## 💻 Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

### Navigating the Dashboard
*   **🏠 Project Overview**: Mission statement and high-level summary.
*   **🧠 Methodology & Rigor**: Deep dive into our "Cluster-Assist" labeling, Faithfulness tests, and Robust Training techniques.
*   **📈 Performance Hub**: Interactive charts showing dataset statistics, training curves, and model benchmarks.
*   **🤖 Live Demo & LLM**: The core tool. Upload an image, select a model (Binary/Severity), view the prediction + Grad-CAM, and generate an AI report.

## 📂 Project Structure

*   `app.py`: Main Streamlit application.
*   `vit_wildfire_severity_model/`: Pre-trained Vision Transformer for severity classification.
*   `vit_wildfire_binary_robust/`: Pre-trained Vision Transformer for binary detection.
*   `explain_wildfire.py`: Script for generating Grad-CAM visualizations.
*   `evaluate_explainability.py`: Script for running faithfulness perturbation tests.
*   `benchmark_binary.py`: Script for benchmarking the binary model.

## 👥 Credits

**Group 8**
*   Aryan Ajmera
*   Rushabh Bhatt
*   Soham
*   Dhrity

---
*Built with PyTorch, Hugging Face Transformers, and Streamlit.*
