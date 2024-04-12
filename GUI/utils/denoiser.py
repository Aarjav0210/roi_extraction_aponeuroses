import cv2
import numpy as np
from skimage.restoration import estimate_sigma, denoise_nl_means
import tensorflow as tf
from tensorflow.keras.models import load_model


def layered_average_pooling(im, layers=[5, 3]):
    steps = layers
    for step in steps:
        temp = np.zeros((im.shape[0] // step, im.shape[1] // step))
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                temp[i, j] = np.mean(im[i*step:(i+1)*step, j*step:(j+1)*step])
        im = temp.copy()

    return im


def bilateral_filter(im):
    im_bilateral = cv2.bilateralFilter(im.astype(np.uint8), 9, 75, 75)
    return im_bilateral


def gaussian_smooth(im, k=3):
    im_blurred = cv2.GaussianBlur(im, (k, k), 0)
    return im_blurred


def nlm_denoise(noisy_im):
    sigma_est = np.mean(estimate_sigma(noisy_im))
    denoised_image = denoise_nl_means(noisy_im, h=1.15 * sigma_est, fast_mode=True,
                                      patch_size=5, patch_distance=3)
    
    denoised_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return denoised_image


def zsnoise2noise(noisy_im):
    image = np.asarray(noisy_im, dtype=np.float32) / 255.0  # Convert to float32 and normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    model_path = './utils/saved_model_1'
    

    tfsmlayer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    
    inputs = tf.keras.Input(shape=(None, None, 1))  # Update the shape as per your requirement
    outputs = tfsmlayer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    denoised_image = model(image, training=False)

    # Extract the tensor from the dictionary using the correct key
    output_tensor = denoised_image['output_1']

    # Now you can proceed with your original processing, assuming output_tensor is the correct tensor
    denoised_image = output_tensor[0, ..., 0].numpy()


    # denoised_image = denoised_image[0, ..., 0].numpy()
    return denoised_image

def denoise_image(image, technique):
    if technique == 'Bilateral':
        denoised_image = bilateral_filter(image)
    elif technique == 'Gaussian':
        denoised_image = gaussian_smooth(image)
    elif technique == 'NLM':
        denoised_image = nlm_denoise(image)
    elif technique == 'Average Pooling':
        denoised_image = layered_average_pooling(image)
    elif technique == 'Noise2Noise':
        denoised_image = zsnoise2noise(image)
    else:
        raise ValueError('Invalid denoising technique.')
    return denoised_image