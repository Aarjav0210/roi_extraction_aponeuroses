import tensorflow as tf

model = tf.keras.load_model('../saved_model_2')


def _preprocess(image):
    image = np.asarray(image, dtype=np.float32) / 255.0  # Convert to float32 and normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def zs_n2n(image):
    image = _preprocess(image)
    denoised_image = model(image, training=False)
    pred = predictions[0, ..., 0]
    return denoised_image.astype(np.uint8)