import tensorflow as tf
from tensorflow.keras import layers, models

def build_fusion_model():
    # --- 1. HISTOPATHOLOGY BRANCH (2D) ---
    # Standard input for ResNet50 is 224x224x3
    histo_input = layers.Input(shape=(224, 224, 3), name='histo_input')
    
    # We use a pre-trained ResNet50 but remove the final 'Top' classification layers
    base_resnet = tf.keras.applications.ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_tensor=histo_input
    )
    
    # Freeze the base layers for now so we don't destroy pre-trained weights
    base_resnet.trainable = False 
    
    h = layers.GlobalAveragePooling2D()(base_resnet.output)
    h = layers.Dense(512, activation='relu', name='histo_feature_vector')(h)

    # --- 2. CT VOLUMETRIC BRANCH (3D) ---
    # Input is a 64x64x64 cube with 1 channel (grayscale intensity)
    ct_input = layers.Input(shape=(64, 64, 64, 1), name='ct_input')
    
    c = layers.Conv3D(32, kernel_size=3, activation='relu')(ct_input)
    c = layers.MaxPooling3D(pool_size=2)(c)
    c = layers.Conv3D(64, kernel_size=3, activation='relu')(c)
    c = layers.MaxPooling3D(pool_size=2)(c)
    c = layers.GlobalAveragePooling3D()(c)
    c = layers.Dense(512, activation='relu', name='ct_feature_vector')(c)

    # --- 3. LATE FUSION ---
    # Concatenate Histo features (512) and CT features (512) = 1024 vector
    fused = layers.Concatenate()([h, c])
    
    # Add final decision-making layers
    x = layers.Dense(256, activation='relu')(fused)
    x = layers.Dropout(0.3)(x) # Prevents overfitting
    
    # Output: 2 units (Benign vs Malignant)
    output = layers.Dense(2, activation='softmax', name='final_output')(x)

    # Create the Multi-Input Model
    model = models.Model(inputs=[histo_input, ct_input], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test if the model builds correctly
    m = build_fusion_model()
    m.summary()
    print("\n✅ Model architecture built and verified successfully!")