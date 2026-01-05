import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import pickle
from scipy.ndimage import zoom

# --- 1. ROBUST FILE LOADING ---
def load_medical_image(path):
    """Loads .pkl, .nii, or .npy and forces shape (128, 128, 64, 1)"""
    try:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                img = data.get('image') or data.get('data') or data.get('voxels') or data.get('vol')
                if img is None:
                    for val in data.values():
                        if isinstance(val, np.ndarray):
                            img = val
                            break
            else:
                img = data
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            img = nib.load(path).get_fdata()
        else:
            img = np.load(path)

        if img is None:
            raise ValueError("Could not find array data in file.")

        img = img.astype(np.float32)

        # RESIZE: Forces matching (128, 128, 64)
        if img.shape != (128, 128, 64):
            factors = (128/img.shape[0], 128/img.shape[1], 64/img.shape[2])
            img = zoom(img, factors, order=1) 

        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=-1)
        
        # Min-Max Normalization
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return img

    except Exception as e:
        raise RuntimeError(f"Failed to process {os.path.basename(path)}: {e}")

# --- 2. DATA GENERATOR ---
def data_generator(image_folder, metadata_dict):
    file_list = [f for f in os.listdir(image_folder) if f.lower().endswith(('.nii', '.nii.gz', '.npy', '.pkl'))]
    
    while True:
        np.random.shuffle(file_list)
        for fname in file_list:
            img_path = os.path.join(image_folder, fname)
            try:
                image_data = load_medical_image(img_path)
                info = metadata_dict.get(fname, {'meta': np.zeros(10), 'label': 0})
                meta_data = np.array(info['meta'], dtype=np.float32)
                label = np.array([info['label']], dtype=np.int32)
                yield (image_data, meta_data), label
            except Exception as e:
                print(f"Error: {e}")
                continue

# --- 3. PIPELINE ---
def get_datasets(image_folder, metadata_dict, batch_size=2, total_samples=100):
    output_signature = (
        (
            tf.TensorSpec(shape=(128, 128, 64, 1), dtype=tf.float32), 
            tf.TensorSpec(shape=(10,), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32)
    )

    full_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(image_folder, metadata_dict),
        output_signature=output_signature
    )

    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    train_ds = full_dataset.take(train_size).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = full_dataset.skip(train_size).take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, train_size, val_size

# --- 4. MODEL ---
def build_model():
    img_in = tf.keras.Input(shape=(128, 128, 64, 1), name="img_input")
    x = tf.keras.layers.Conv3D(32, 3, activation="relu", padding="same")(img_in)
    x = tf.keras.layers.MaxPooling3D()(x)
    x = tf.keras.layers.Conv3D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    meta_in = tf.keras.Input(shape=(10,), name="meta_input")
    y = tf.keras.layers.Dense(16, activation="relu")(meta_in)

    combined = tf.keras.layers.Concatenate()([x, y])
    z = tf.keras.layers.Dense(32, activation="relu")(combined)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    model = tf.keras.Model(inputs=[img_in, meta_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- 5. RUN ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "ct")
    
    if not os.path.exists(DATA_PATH):
        print(f"Directory not found: {DATA_PATH}")
    else:
        all_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(('.nii', '.nii.gz', '.npy', '.pkl'))]
        total_files = len(all_files)

        if total_files == 0:
            print("No valid scan files found.")
        else:
            print(f"Found {total_files} files. Initializing...")
            metadata_dict = {f: {"meta": np.random.rand(10), "label": 0} for f in all_files}
            
            batch_size = 2
            train_ds, val_ds, n_train, n_val = get_datasets(DATA_PATH, metadata_dict, batch_size, total_files)

            model = build_model()
            
            spe = max(1, n_train // batch_size)
            vs = max(1, n_val // batch_size)

            # --- TRAINING ---
            print("Training started...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10,
                steps_per_epoch=spe,
                validation_steps=vs
            )

            # --- SAVING ---
            model_path = os.path.join(os.path.dirname(BASE_DIR), "xailung_ct_model.keras")
            model.save(model_path)
            print(f"Model saved successfully at: {model_path}")