import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. Load Model and Data
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv")

print("Generating ROC Curve data...")
ct_inputs, histo_inputs, meta_inputs, y_true = [], [], [], []

# Collect all samples
for _, row in df.iterrows():
    ct_inputs.append(np.load(row['ct_path'])[..., np.newaxis])
    histo_inputs.append(np.load(row['histo_path']))
    meta_inputs.append([row['age'], row['smoking_history']])
    y_true.append(row['label'])

# 2. Get Probability Predictions
# We need the raw decimals (0.0 to 1.0), not just the 0/1 classes
y_probs = model.predict([np.array(ct_inputs), np.array(histo_inputs), np.array(meta_inputs)])

# 3. Calculate ROC
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# 4. Plot
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC): XAILUNG Multimodal')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('visuals/roc_curve.png')
print(f"ROC Curve saved with AUC: {roc_auc:.4f}")