import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. SETUP & DATA LOAD
os.makedirs("models", exist_ok=True)
df = pd.read_csv("data/metadata/multimodal_master.csv")

def data_generator(df, batch_size=2): # Reduced batch size for stability
    while True:
        df = df.sample(frac=1).reset_index(drop=True) 
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            ct_imgs, histo_imgs, meta_data, labels = [], [], [], []
            for _, row in batch.iterrows():
                try:
                    ct_val = np.load(row['ct_path'])
                    if ct_val.ndim == 3: ct_val = ct_val[..., np.newaxis]
                    ct_imgs.append(ct_val)
                    histo_imgs.append(np.load(row['histo_path']))
                    meta_data.append([row['age'], row['smoking_history']])
                    labels.append(row['label'])
                except: continue
            
            if not ct_imgs: continue
            yield ({"ct_input": np.array(ct_imgs), "histo_input": np.array(histo_imgs), 
                    "meta_input": np.array(meta_data)}, np.array(labels))

# 2. MODEL BUILDING
ct_input = layers.Input(shape=(128, 128, 64, 1), name="ct_input")
x = layers.Conv3D(16, 3, activation='relu')(ct_input)
x = layers.MaxPooling3D()(x)
x = layers.Conv3D(32, 3, activation='relu')(x)
x = layers.GlobalAveragePooling3D()(x)

histo_input = layers.Input(shape=(224, 224, 3), name="histo_input")
base_model = tf.keras.applications.MobileNetV2(input_tensor=histo_input, include_top=False, weights='imagenet')
base_model.trainable = False 
y = layers.GlobalAveragePooling2D()(base_model.output)

meta_input = layers.Input(shape=(2,), name="meta_input")
z = layers.Dense(16, activation='relu')(meta_input)

combined = layers.concatenate([x, y, z])
combined = layers.Dense(64, activation='relu')(combined)
output = layers.Dense(1, activation='sigmoid', name="prediction")(combined)

model = models.Model(inputs=[ct_input, histo_input, meta_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. FREQUENT SAVING CALLBACK
# This saves the model every 100 batches so you don't lose progress
step_save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/xailung_step_checkpoint.keras",
    save_weights_only=False,
    save_freq=100 
)

# Standard end-of-epoch checkpoint
epoch_save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/xailung_multimodal_best.keras",
    monitor='accuracy',
    save_best_only=True
)

print("\n--- Starting Crash-Proof Training ---")
batch_size = 2
model.fit(
    data_generator(df, batch_size=batch_size), 
    steps_per_epoch=len(df) // batch_size, 
    epochs=10, 
    callbacks=[step_save_callback, epoch_save_callback]
)