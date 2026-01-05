import os
import numpy as np
import pickle
import cv2
from scipy.ndimage import zoom
from PIL import Image

# --- CONFIGURATION ---
RAW_CT_DIR = "data/raw/ct"
RAW_HISTO_DIR = "data/raw/histo"
PROCESSED_CT_DIR = "data/processed/ct"
PROCESSED_HISTO_DIR = "data/processed/histo"

for d in [PROCESSED_CT_DIR, PROCESSED_HISTO_DIR]:
    os.makedirs(d, exist_ok=True)

# --- 1. RADIOLOGY PREPROCESSING (CT) ---

def process_ct(path_or_file):
    """Core logic for 3D CT processing: Interpolation & Normalization."""
    # Check if input is a path (string) or a file-like object (Streamlit upload)
    if isinstance(path_or_file, str):
        with open(path_or_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = pickle.load(path_or_file)
        
    img = data.get('image') or data.get('data') or list(data.values())[0]
    img = img.astype(np.float32)
    
    # Target Shape: (128, 128, 64)
    factors = (128/img.shape[0], 128/img.shape[1], 64/img.shape[2])
    img = zoom(img, factors, order=1)
    
    # Min-Max Normalization
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return img

def preprocess_ct_single(uploaded_file):
    """Wrapper for Streamlit app to return a 5D Tensor (Batch, D, H, W, C)"""
    img = process_ct(uploaded_file)
    img = np.expand_dims(img, axis=-1) # Add Channel
    img = np.expand_dims(img, axis=0)  # Add Batch
    return img

# --- 2. PATHOLOGY PREPROCESSING (HISTO) ---

def process_histo(path_or_file):
    """Core logic for Histopathology: Color conversion, Resize, Normalization."""
    if isinstance(path_or_file, str):
        img = cv2.imread(path_or_file)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # For Streamlit uploads
        img = Image.open(path_or_file).convert('RGB')
        img = np.array(img)
        
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_histo_single(uploaded_file):
    """Wrapper for Streamlit app to return a 4D Tensor (Batch, H, W, C)"""
    img = process_histo(uploaded_file)
    img = np.expand_dims(img, axis=0) # Add Batch
    return img

# --- 3. BATCH EXECUTION LOGIC ---
# This part only runs if you execute this script directly (python src/preprocess.py)
if __name__ == "__main__":
    print("🚀 Starting Batch Preprocessing...")

    # Process Histo (Recursive search)
    histo_count = 0
    for root, dirs, files in os.walk(RAW_HISTO_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                raw_path = os.path.join(root, f)
                unique_name = f"{os.path.basename(root)}_{f.rsplit('.', 1)[0]}.npy"
                out_path = os.path.join(PROCESSED_HISTO_DIR, unique_name)
                
                if not os.path.exists(out_path):
                    try:
                        arr = process_histo(raw_path)
                        if arr is not None:
                            np.save(out_path, arr)
                            histo_count += 1
                    except Exception as e:
                        print(f"Error: {f}: {e}")
    print(f"✅ Histo Complete: {histo_count} images.")

    # Process CT
    ct_count = 0
    ct_files = [f for f in os.listdir(RAW_CT_DIR) if f.endswith('.pkl')]
    for f in ct_files:
        out_path = os.path.join(PROCESSED_CT_DIR, f.replace('.pkl', '.npy'))
        if not os.path.exists(out_path):
            try:
                arr = process_ct(os.path.join(RAW_CT_DIR, f))
                np.save(out_path, arr)
                ct_count += 1
            except Exception as e:
                print(f"Error: CT {f}: {e}")
    print(f"✅ CT Complete: {ct_count} scans.")