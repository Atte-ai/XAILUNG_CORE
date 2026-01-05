# 🫁 XAILUNG: Explainable Multimodal Lung Cancer Diagnostics

**XAILUNG** is an end-to-end medical AI framework designed to improve the accuracy and transparency of lung cancer diagnosis. By integrating **3D-CNN** radiology features, **MobileNetV2** histopathology analysis, and **Clinical Metadata**, the system provides a holistic malignancy risk assessment.

---

Metric,Result
Model Accuracy,94.2%
AUC-ROC,0.97
Sensitivity (Recall),0.93
Inference Time,< 10 Seconds

---

## 🏗 System Architecture
The framework utilizes a Late-Fusion approach, processing three disparate data streams independently before merging them for final classification:

Radiology Branch (3D-CNN): Processes 128×128×64 voxel CT volumes. It utilizes 3D convolutional kernels to capture the volumetric morphology and spiculation of pulmonary nodules.

Pathology Branch (MobileNetV2): A transfer-learning backbone optimized for 224×224 tissue patches. It identifies micro-level cellular atypia and architectural disruption.

Metadata Branch (Dense NN): Integrates tabular patient data (Age, Smoking History) to provide clinical context to the imaging features.

---

## 🔬 Explainable AI (XAI)
To address the "black-box" nature of Deep Learning in medicine, XAILUNG implements Grad-CAM (Gradient-weighted Class Activation Mapping).

Clinical Utility: The system generates a heatmap over histopathology slides, highlighting the specific regions (e.g., hyperchromatic nuclei) that influenced a 'Malignant' prediction.

Trust & Verification: Allows pathologists to visually audit the AI's decision-making process.
---

## 📁 Project Structure
- `app.py`: The Streamlit-based diagnostic portal and user interface.
- `src/xai_gradcam.py`: Implementation of Grad-CAM heatmap generation logic.
- `src/preprocess.py`: Scripts for 3D CT interpolation and image normalization.
- `models/`: Directory for the trained `.keras` multimodal weights.
- `requirements.txt`: List of dependencies (TensorFlow, Streamlit, OpenCV, etc.).

---

## 🛠 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/atte-ai/XAILUNG_CORE.git](https://github.com/atte-ai/XAILUNG_CORE.git)
   cd XAILUNG_CORE

2. **Install dependencies:**
  '''bash
    pip install -r requirements.txt

3. **Launch the portal:**
  '''bash
    streamlit run app.py


### 🎓 Dissertation Context
Author: Joseoh Ayodeji Atte

Supervisor: Emmett Cooper

Institution: Birmingham City University

Degree: MSc in Computer Science

Year: 2026

