import cv2
import numpy as np
from skimage.restoration import estimate_sigma, denoise_nl_means

def denoise_image(noisy_im):
    # Apply Non-Local Means Denoising
    sigma_est = np.mean(estimate_sigma(noisy_im))
    denoised_image = denoise_nl_means(noisy_im, h=1.15 * sigma_est, fast_mode=True,
                                      patch_size=5, patch_distance=3)
    # denoised_image = denoise_nl_means(im, h=1.15 * sigma_est, sigma=sigma_est fast_mode=True,
    
    # Normalize the image
    denoised_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return denoised_image