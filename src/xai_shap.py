import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Model and a small batch of data
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv").head(20)

# 2. Function to test "Importance" by zeroing out branches
def get_importance():
    ct_inputs = np.array([np.load(p)[..., np.newaxis] for p in df['ct_path']])
    histo_inputs = np.array([np.load(p) for p in df['histo_path']])
    meta_inputs = df[['age', 'smoking_history']].values
    
    # Base prediction (All data)
    base_preds = model.predict([ct_inputs, histo_inputs, meta_inputs])
    
    # 3. Test CT Importance (Zero out CT)
    no_ct = model.predict([np.zeros_like(ct_inputs), histo_inputs, meta_inputs])
    ct_impact = np.mean(np.abs(base_preds - no_ct))
    
    # 4. Test Histo Importance (Zero out Histo)
    no_histo = model.predict([ct_inputs, np.zeros_like(histo_inputs), meta_inputs])
    histo_impact = np.mean(np.abs(base_preds - no_histo))
    
    # 5. Test Metadata Importance (Zero out Metadata)
    no_meta = model.predict([ct_inputs, histo_inputs, np.zeros_like(meta_inputs)])
    meta_impact = np.mean(np.abs(base_preds - no_meta))
    
    return {"Radiology (CT)": ct_impact, "Pathology (Histo)": histo_impact, "Clinical (Meta)": meta_impact}

# 6. Plot the results
importance = get_importance()
plt.figure(figsize=(10, 6))
plt.bar(importance.keys(), importance.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Global Feature Importance (Modality Contribution)")
plt.ylabel("Mean Impact on Prediction Confidence")
plt.savefig('visuals/feature_importance_shap.png')
print("Feature importance chart saved to visuals/feature_importance_shap.png")