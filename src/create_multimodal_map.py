import os
import pandas as pd
import re
import random

# Configuration - Paths to your PROCESSED data
CT_DIR = "data/processed/ct"
HISTO_DIR = "data/processed/histo"
OUTPUT_DIR = "data/metadata"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "multimodal_master.csv")

def extract_uid(filename):
    """Extracts the long DICOM-style UID from the filename."""
    match = re.search(r'(\d+\.\d+\.\d+\.\d+\.\d+\.\d+\.\d+)', filename)
    return match.group(1) if match else "unknown"

# 1. Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Get list of all processed files
ct_files = [f for f in os.listdir(CT_DIR) if f.endswith('.npy')]
histo_files = [f for f in os.listdir(HISTO_DIR) if f.endswith('.npy')]

# 3. Categorize Histo files by class (based on folder names from your photo)
# lung_aca = Adenocarcinoma (Malignant)
# lung_scc = Squamous Cell Carcinoma (Malignant)
# lung_n = Normal (Benign)
histo_mal = [f for f in histo_files if "lung_aca" in f.lower() or "lung_scc" in f.lower()]
histo_ben = [f for f in histo_files if "lung_n" in f.lower()]

print(f"Found {len(ct_files)} CT scans.")
print(f"Found {len(histo_mal)} Malignant Histo slides and {len(histo_ben)} Benign Histo slides.")

# 4. Create the Paired Map
paired_data = []

for ct_f in ct_files:
    patient_id = extract_uid(ct_f)
    is_malignant = ct_f.startswith("MAL")
    
    # Select a matching Histo image based on the CT label
    if is_malignant:
        if not histo_mal: continue
        selected_histo = random.choice(histo_mal)
        label = 1
    else:
        if not histo_ben: continue
        selected_histo = random.choice(histo_ben)
        label = 0
        
    paired_data.append({
        "patient_id": patient_id,
        "ct_path": os.path.join(CT_DIR, ct_f),
        "histo_path": os.path.join(HISTO_DIR, selected_histo),
        "label": label,
        # Placeholder metadata for the 3rd input head
        "age": random.randint(45, 80), 
        "smoking_history": random.choice([0, 1]) 
    })

# 5. Save to CSV
df = pd.DataFrame(paired_data)
df.to_csv(OUTPUT_FILE, index=False)

print("-" * 30)
print(f"SUCCESS: Created {OUTPUT_FILE}")
print(f"Total Paired Samples: {len(df)}")
print(f"Malignant: {len(df[df['label']==1])} | Benign: {len(df[df['label']==0])}")