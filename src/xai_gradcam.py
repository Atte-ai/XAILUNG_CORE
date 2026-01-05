import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

# --- 1. CORE GRAD-CAM LOGIC ---

def make_gradcam_heatmap(img_inputs, model, last_conv_layer_name="out_relu"):
    """
    Computes the Grad-CAM heatmap for the pathology branch.
    Args:
        img_inputs: List of [ct_tensor, histo_tensor, meta_tensor]
        model: The loaded Keras model
        last_conv_layer_name: The name of the layer to monitor (e.g., 'out_relu')
    """
    # Create a sub-model that outputs the last conv layer and the final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_inputs)
        # Target the malignancy probability channel
        class_channel = preds[:, 0]

    # Calculate gradients of the class with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global Average Pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map channels by the gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def generate_superimposed_image(original_img, heatmap, alpha=0.6):
    """
    Overlays the heatmap on the original image.
    Args:
        original_img: NumPy array of the original histo patch (0-255 or 0-1)
        heatmap: The 2D heatmap array from make_gradcam_heatmap
    """
    # Ensure original image is 0-255 uint8
    if original_img.max() <= 1.0:
        img = (original_img * 255).astype(np.uint8)
    else:
        img = original_img.astype(np.uint8)

    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB colormap
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Combine original and heatmap
    superimposed_img = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)
    return superimposed_img


# --- 2. BATCH EXECUTION LOGIC ---
# This part only runs if you run this file directly: python src/xai_gradcam.py
if __name__ == "__main__":
    import pandas as pd
    
    print("🧪 Running XAI Validation Script...")
    
    if not os.path.exists('visuals'):
        os.makedirs('visuals')

    # Load resources
    model_path = 'models/xailung_multimodal_best.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        last_conv_layer_name = "out_relu" 

        # Load a sample from metadata
        try:
            df = pd.read_csv("data/metadata/multimodal_master.csv")
            sample = df[df['label'] == 1].iloc[0] 

            # Prepare inputs
            ct = np.expand_dims(np.load(sample['ct_path'])[..., np.newaxis], axis=0)
            histo = np.expand_dims(np.load(sample['histo_path']), axis=0)
            meta = np.array([[sample['age'], sample['smoking_history']]])

            # Generate
            heatmap = make_gradcam_heatmap([ct, histo, meta], model, last_conv_layer_name)
            vis_img = generate_superimposed_image(histo[0], heatmap)

            # Save
            plt.imsave('visuals/gradcam_histo_final.png', vis_img)
            print("✅ Success! Visualization saved to visuals/gradcam_histo_final.png")
            
        except Exception as e:
            print(f"❌ Error during sample processing: {e}")
    else:
        print(f"❌ Model not found at {model_path}")