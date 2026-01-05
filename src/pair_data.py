import os
import pickle

# --- CONFIG ---
BASE_DIR = '/Users/banky/Desktop/XAILUNG_CORE'
HISTO_BASE = os.path.join(BASE_DIR, 'data/raw/histo')
CT_DIR = os.path.join(BASE_DIR, 'data/raw/ct')

def create_manifest():
    # 1. Collect Histo Paths based on your folders
    # Malignant: Combining lung_aca and lung_scc
    h_aca = [os.path.join(HISTO_BASE, 'lung_aca', f) for f in os.listdir(os.path.join(HISTO_BASE, 'lung_aca')) if f.endswith('.jpeg')]
    h_scc = [os.path.join(HISTO_BASE, 'lung_scc', f) for f in os.listdir(os.path.join(HISTO_BASE, 'lung_scc')) if f.endswith('.jpeg')]
    histo_m = sorted(h_aca + h_scc)
    
    # Benign: lung_n
    histo_b = sorted([os.path.join(HISTO_BASE, 'lung_n', f) for f in os.listdir(os.path.join(HISTO_BASE, 'lung_n')) if f.endswith('.jpeg')])
    
    # 2. Collect CT Paths
    ct_files = os.listdir(CT_DIR)
    ct_m = sorted([os.path.join(CT_DIR, f) for f in ct_files if f.startswith('MAL')])
    ct_b = sorted([os.path.join(CT_DIR, f) for f in ct_files if f.startswith('BEN')])

    # 3. Pair them 1-to-1
    m_count = min(len(histo_m), len(ct_m))
    b_count = min(len(histo_b), len(ct_b))
    
    paired_data = []
    
    # Label 1: Malignant
    for i in range(m_count):
        paired_data.append((histo_m[i], ct_m[i], 1))
        
    # Label 0: Benign
    for i in range(b_count):
        paired_data.append((histo_b[i], ct_b[i], 0))

    print(f"✅ Paired {m_count} Malignant samples (ACA + SCC).")
    print(f"✅ Paired {b_count} Benign samples (N).")
    print(f"📊 Total Dataset Size: {len(paired_data)}")

    # 4. Save Manifest
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    manifest_path = os.path.join(BASE_DIR, 'data/manifest.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(paired_data, f)
    print(f"🚀 Manifest saved to {manifest_path}")

if __name__ == "__main__":
    create_manifest()