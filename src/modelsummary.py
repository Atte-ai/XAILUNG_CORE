import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('models/xailung_multimodal_best.keras')

# Print the full list of layers
model.summary()
for layer in model.layers:
    # Check if the layer is a convolution layer
    if 'conv' in layer.name.lower():
        print(f"Layer Name: {layer.name}")
