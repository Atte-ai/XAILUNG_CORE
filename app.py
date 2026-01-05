import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os
from scipy.ndimage import zoom

# --- 1. IMPORTING YOUR XAI LOGIC ---
try:
    from src.xai_gradcam import make_gradcam_heatmap
except ImportError:
    st.error("Could not find 'make_gradcam_heatmap' in src/xai_gradcam.py.")

# Helper to overlay heatmap on the original image
def display_gradcam(img_array, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    img = img_array[0]
    if img.max() <= 1.0:
        img = img * 255
    
    jet = cv2.resize(jet, (int(img.shape[1]), int(img.shape[0])))
    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# --- 2. PREPROCESSING ---
def process_ct_pkl(uploaded_file):
    data = pickle.load(uploaded_file)
    img = data.get('image') or data.get('data') or list(data.values())[0]
    img = img.astype(np.float32)
    factors = (128/img.shape[0], 128/img.shape[1], 64/img.shape[2])
    img = zoom(img, factors, order=1)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return np.expand_dims(np.expand_dims(img, axis=-1), axis=0)

def process_histo_img(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_final = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(img_final, axis=0)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="XAILUNG Diagnostic Portal", layout="wide")
st.title("🫁 XAILUNG: Multimodal Diagnostic Framework")

st.sidebar.header("3. Clinical Metadata")
age = st.sidebar.slider("Patient Age", 18, 95, 60)
smoke = st.sidebar.number_input("Smoking History (Pack Years)", 0, 150, 20)

col1, col2 = st.columns(2)
with col1:
    st.header("1. Radiology (CT)")
    ct_file = st.file_uploader("Upload .pkl Scan", type=['pkl'])
with col2:
    st.header("2. Pathology (Histo)")
    histo_file = st.file_uploader("Upload Slide Image", type=['jpg', 'png', 'jpeg'])

# --- 4. EXECUTION ---
if st.button("🚀 Run Full Diagnostic Pipeline"):
    if ct_file and histo_file:
        try:
            model_path = 'models/xailung_multimodal_best.keras'
            model = tf.keras.models.load_model(model_path)

            with st.status("Executing Multimodal Pipeline...", expanded=True) as s:
                st.write("Preprocessing Radiology Volume...")
                p_ct = process_ct_pkl(ct_file)
                
                st.write("Preprocessing Histopathology Slide...")
                p_histo = process_histo_img(histo_file)
                
                st.write("Vectorizing Clinical Metadata...")
                meta_vector = np.array([[age, smoke]], dtype=np.float32)
                
                st.write("Performing Multimodal Fusion Inference...")
                prediction = model.predict([p_ct, p_histo, meta_vector])
                risk_score = prediction[0][0]
                
                st.write("Generating XAI Heatmap...")
                
                # AUTO-DETECT LAST CONV LAYER
                target_layer = None
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        target_layer = layer.name
                        break
                
                # FIX: Passing arguments POSITIONALLY to avoid naming conflicts
                # Logic: (input_data, trained_model, layer_name)
                raw_heatmap = make_gradcam_heatmap(p_histo, model, target_layer)
                
                xai_overlay = display_gradcam(p_histo, raw_heatmap)
                s.update(label="Analysis Complete!", state="complete")

            # --- 5. RESULTS ---
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.subheader("Diagnostic Result")
                st.metric("Malignancy Risk", f"{risk_score*100:.2f}%")
                if risk_score > 0.5:
                    st.error("Result: MALIGNANT")
                else:
                    st.success("Result: BENIGN")
                st.caption("Status: Real Model Active")

            with res_col2:
                st.subheader("Explainable AI (XAI) Output")
                st.image(xai_overlay, caption=f"Grad-CAM Heatmap (Layer: {target_layer})", use_container_width=True)
                st.info("Red zones highlight the specific tissue features your model used to reach its conclusion.")

        except Exception as e:
            st.error(f"Pipeline Error: {e}")
    else:
        st.warning("Please upload both imaging files to proceed.")