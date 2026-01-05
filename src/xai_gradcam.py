import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1. Setup
if not os.path.exists('visuals'):
    os.makedirs('visuals')

model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')

# From your error log, the last relevant layer for the Histo branch is 'out_relu'
last_conv_layer_name = "out_relu" 

def make_gradcam_heatmap(img_inputs, model, last_conv_layer_name):
    # Create a model that maps the inputs to the activations of the last conv layer
    # and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_inputs)
        # We target the 'Malignant' probability (index 0 usually, but let's be safe)
        class_channel = preds[:, 0]

    # Gradient of the output neuron wrt the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of intensity of gradients over the feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by 'how important this channel is'
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 2. Select a Malignant Sample
df = pd.read_csv("data/metadata/multimodal_master.csv")
sample = df[df['label'] == 1].iloc[0] 

# Prepare inputs (using the names from your layer list)
ct = np.expand_dims(np.load(sample['ct_path'])[..., np.newaxis], axis=0)
histo = np.expand_dims(np.load(sample['histo_path']), axis=0)
meta = np.array([[sample['age'], sample['smoking_history']]])

# 3. Generate and Save
print("Generating Grad-CAM for Pathology branch...")
heatmap = make_gradcam_heatmap([ct, histo, meta], model, last_conv_layer_name)

# Rescale heatmap to 0-255 and apply colormap
img = (histo[0] * 255).astype(np.uint8) # Original image
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

# Superimpose the heatmap on original image
superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# Save result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Histopathology")
plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title("Grad-CAM Explainability")
plt.savefig('visuals/gradcam_histo_final.png')

print("Success! Grad-CAM visualization saved to visuals/gradcam_histo_final.png")