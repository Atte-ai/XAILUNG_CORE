import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load Model and Data
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv")

print("Evaluating all samples...")
ct_inputs, histo_inputs, meta_inputs, y_true = [], [], [], []

# We'll test on a subset of 100 for speed, or remove .head(100) for all
for _, row in df.head(100).iterrows():
    ct_inputs.append(np.load(row['ct_path'])[..., np.newaxis])
    histo_inputs.append(np.load(row['histo_path']))
    meta_inputs.append([row['age'], row['smoking_history']])
    y_true.append(row['label'])

# 2. Get Predictions
preds = model.predict([np.array(ct_inputs), np.array(histo_inputs), np.array(meta_inputs)])
y_pred = (preds > 0.5).astype(int)

# 3. Create Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'], 
            yticklabels=['Benign', 'Malignant'])
plt.title('XAILUNG Final Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('visuals/confusion_matrix.png')

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred))