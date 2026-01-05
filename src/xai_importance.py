import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load Model and Data
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv").head(50)

def get_modality_importance():
    # Prepare original inputs
    ct = np.array([np.load(p)[..., np.newaxis] for p in df['ct_path']])
    histo = np.array([np.load(p) for p in df['histo_path']])
    meta = df[['age', 'smoking_history']].values
    
    # Baseline Accuracy (using all data)
    baseline_preds = model.predict([ct, histo, meta])
    
    # 2. Permutation: "Kill" one modality and see how much the prediction changes
    # Test CT importance
    ct_removed = model.predict([np.zeros_like(ct), histo, meta])
    ct_drop = np.mean(np.abs(baseline_preds - ct_removed))
    
    # Test Histo importance
    histo_removed = model.predict([ct, np.zeros_like(histo), meta])
    histo_drop = np.mean(np.abs(baseline_preds - histo_removed))
    
    # Test Meta importance
    meta_removed = model.predict([ct, histo, np.zeros_like(meta)])
    meta_drop = np.mean(np.abs(baseline_preds - meta_removed))
    
    return {"Radiology (CT)": ct_drop, "Pathology (Histo)": histo_drop, "Clinical (Meta)": meta_drop}

# 3. Plotting
importance = get_modality_importance()
plt.figure(figsize=(10, 6))
plt.bar(importance.keys(), importance.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title("XAILUNG: Global Modality Importance", fontweight='bold')
plt.ylabel("Mean Decrease in Prediction Confidence")
plt.savefig('visuals/modality_importance.png')
print("Global importance chart saved to visuals/modality_importance.png")