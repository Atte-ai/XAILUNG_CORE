import tensorflow as tf
import numpy as np
import pandas as pd

# 1. Load the Model
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')
df = pd.read_csv("data/metadata/multimodal_master.csv")

# 2. Pick one sample (e.g., the first one)
sample = df.iloc[0]

# 3. Load the data for that sample
ct_data = np.load(sample['ct_path'])[..., np.newaxis] # Add channel dim
histo_data = np.load(sample['histo_path'])
meta_data = np.array([[sample['age'], sample['smoking_history']]])

# 4. Predict
prediction = model.predict({
    "ct_input": np.expand_dims(ct_data, axis=0),
    "histo_input": np.expand_dims(histo_data, axis=0),
    "meta_input": meta_data
})

# 5. Output a Professional Summary for your Dissertation
print("\n" + "="*40)
print("       XAILUNG DIAGNOSTIC REPORT       ")
print("="*40)
print(f"Patient ID:      {sample['patient_id']}")
print(f"Clinical Age:    {sample['age']}")
print(f"Smoking History: {sample['smoking_history']} pack-years")
print("-" * 40)
print(f"Malignancy Risk: {prediction[0][0]*100:.2f}%")
print(f"Final Diagnosis: {'MALIGNANT' if prediction[0][0] > 0.5 else 'BENIGN'}")
print("="*40)