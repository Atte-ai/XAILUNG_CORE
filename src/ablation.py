import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load Model and Data
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv")

def run_ablation():
    ct = np.array([np.load(p)[..., np.newaxis] for p in df['ct_path']])
    histo = np.array([np.load(p) for p in df['histo_path']])
    meta = df[['age', 'smoking_history']].values
    y_true = df['label'].values
    
    # Version A: Full Multimodal (XAILUNG)
    preds_full = model.predict([ct, histo, meta])
    acc_full = np.mean((preds_full > 0.5).flatten() == y_true)
    
    # Version B: Image Only (Zero out clinical metadata)
    preds_no_meta = model.predict([ct, histo, np.zeros_like(meta)])
    acc_no_meta = np.mean((preds_no_meta > 0.5).flatten() == y_true)
    
    return acc_full, acc_no_meta

# 2. Visualize
full, no_meta = run_ablation()
labels = ['XAILUNG (Full)', 'Images Only']
values = [full, no_meta]

plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['#27ae60', '#95a5a6'])
plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Ablation Study: Value of Multimodal Fusion')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')

plt.savefig('visuals/ablation_study.png')
print("Ablation study complete!")