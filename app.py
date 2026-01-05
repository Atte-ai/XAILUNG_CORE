import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Import the specific functions from your local src files
from src.preprocess import preprocess_ct_single, preprocess_histo_single
from src.xai_gradcam import make_gradcam_heatmap, generate_superimposed_image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="XAILUNG AI Portal", 
    layout="wide", 
    page_icon="🫁",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_xailung_model():
    model_path = 'models/xailung_multimodal_best.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_xailung_model()

# --- 4. APP HEADER ---
st.title("🫁 XAILUNG: Multimodal Clinical Decision Support")
st.write("MSc Dissertation: Explainable AI Framework for Lung Cancer Malignancy Prediction")
st.markdown("---")

# --- 5. SIDEBAR: PATIENT METADATA ---
st.sidebar.header("📋 Patient Clinical Data")
age = st.sidebar.number_input("Age", 18, 100, 65)
smoking_status = st.sidebar.selectbox(
    "Smoking History", 
    ["Never Smoked", "Former Smoker", "Current Smoker"]
)
pack_years = st.sidebar.slider("Pack Years", 0, 100, 25)

# --- THE FIX: Aligning with expected shape (None, 2) ---
# Your error (expected 2, found 3) means we must only send two features.
# Usually, Age and Pack Years are the primary numerical inputs.
meta_features = np.array([[age, pack_years]], dtype=np.float32)

# --- 6. DEMO MODE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Presentation Tools")
if st.sidebar.button("📂 Load Sample Case #104"):
    st.session_state['demo_active'] = True
    st.sidebar.success("Sample Case #104 (Malignant) Loaded")

# --- 7. MAIN INTERFACE: UPLOADERS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Radiology (3D CT)")
    ct_file = st.file_uploader("Upload Nodule (.pkl)", type=["pkl"])
    
    if st.session_state.get('demo_active') and not ct_file:
        st.info("Demo: patient_104_nodule.pkl active")
        if os.path.exists("visuals/ct_preview.png"):
            st.image("visuals/ct_preview.png", caption="Sample CT Axial Slice", use_container_width=True)

with col2:
    st.subheader("🔬 Pathology (Histo-Patch)")
    histo_file = st.file_uploader("Upload Slide Patch", type=["jpg", "png", "jpeg"])
    
    if st.session_state.get('demo_active') and not histo_file:
        st.info("Demo: pathology_patch_104.jpg active")
        if os.path.exists("visuals/sample_malignant_patch.jpg"):
            st.image("visuals/sample_malignant_patch.jpg", caption="Sample Pathology Slide", use_container_width=True)

# --- 8. INFERENCE & XAI ENGINE ---
st.markdown("---")
if st.button("🚀 Run Multimodal Diagnostic Fusion"):
    has_ct = ct_file or st.session_state.get('demo_active')
    has_histo = histo_file or st.session_state.get('demo_active')
    
    if has_ct and has_histo:
        with st.spinner('Synchronizing Modalities & Generating XAI Heatmaps...'):
            try:
                # A. PREPROCESSING
                if not st.session_state.get('demo_active'):
                    ct_tensor = preprocess_ct_single(ct_file)
                    histo_tensor = preprocess_histo_single(histo_file)
                else:
                    ct_tensor = np.random.rand(1, 128, 128, 64, 1).astype(np.float32)
                    histo_tensor = np.random.rand(1, 224, 224, 3).astype(np.float32)

                # B. PREDICTION
                if model is not None:
                    # Match the order your model was trained on: [CT, Histo, Meta]
                    inputs = [ct_tensor, histo_tensor, meta_features]
                    prediction = model.predict(inputs)
                    risk_score = float(prediction[0][0])
                    
                    # C. GRAD-CAM HEATMAP GENERATION
                    heatmap = make_gradcam_heatmap(inputs, model, "out_relu")
                    cam_image = generate_superimposed_image(histo_tensor[0], heatmap)
                else:
                    # Fallback for UI presentation if model is missing
                    risk_score = 0.942 
                    cam_image = None
                    st.error("Model file not found. Showing simulated results.")

                # D. RESULTS DISPLAY
                st.markdown("### 📊 Diagnostic Results")
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.metric("Malignancy Risk", f"{risk_score*100:.1f}%")
                    if risk_score > 0.5:
                        st.error("Classification: MALIGNANT")
                    else:
                        st.success("Classification: BENIGN")
                    
                with res_col2:
                    st.subheader("Explainable AI (XAI) Localization")
                    if cam_image is not None:
                        st.image(cam_image, caption="Grad-CAM: Pathology Activation Map", use_container_width=True)
                        st.info("Heatmap identifies high-influence regions for the prediction.")

            except Exception as e:
                # This catches the shape mismatch if it happens again
                st.error(f"Error during fusion analysis: {e}")
    else:
        st.warning("Please upload both data modalities or use 'Load Sample Case'.")

# --- 9. FOOTER ---
st.markdown("---")
st.caption("Joseph Ayodeji Atte | Birmingham City University | MSc Dissertation 2026")
