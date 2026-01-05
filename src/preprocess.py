import os
import numpy as np
import pickle
import cv2
from scipy.ndimage import zoom

# Configuration
RAW_CT_DIR = "data/raw/ct"
RAW_HISTO_DIR = "data/raw/histo"
PROCESSED_CT_DIR = "data/processed/ct"
PROCESSED_HISTO_DIR = "data/processed/histo"

for d in [PROCESSED_CT_DIR, PROCESSED_HISTO_DIR]:
    os.makedirs(d, exist_ok=True)

def process_ct(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    img = data.get('image') or data.get('data') or list(data.values())[0]
    img = img.astype(np.float32)
    factors = (128/img.shape[0], 128/img.shape[1], 64/img.shape[2])
    img = zoom(img, factors, order=1)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return img

def process_histo(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

# --- EXECUTION ---

# 1. Process Histo (Recursive search for subfolders)
print("Searching for Histo images in subfolders...")
histo_count = 0

for root, dirs, files in os.walk(RAW_HISTO_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            raw_path = os.path.join(root, f)
            # Create a unique name to avoid overwriting if files have same name in different folders
            unique_name = f"{os.path.basename(root)}_{f.rsplit('.', 1)[0]}.npy"
            out_path = os.path.join(PROCESSED_HISTO_DIR, unique_name)
            
            if not os.path.exists(out_path):
                try:
                    arr = process_histo(raw_path)
                    if arr is not None:
                        np.save(out_path, arr)
                        histo_count += 1
                        if histo_count % 100 == 0:
                            print(f"Processed {histo_count} histo images...")
                except Exception as e:
                    print(f"Error processing {f}: {e}")

print(f"Finished Histo! Processed {histo_count} images.")

# 2. Process CT
print("Processing CT scans...")
ct_files = [f for f in os.listdir(RAW_CT_DIR) if f.endswith('.pkl')]
for f in ct_files:
    out_path = os.path.join(PROCESSED_CT_DIR, f.replace('.pkl', '.npy'))
    if not os.path.exists(out_path):
        try:
            arr = process_ct(os.path.join(RAW_CT_DIR, f))
            np.save(out_path, arr)
        except Exception as e:
            print(f"Error processing CT {f}: {e}")

print("\n--- Preprocessing Complete ---")
