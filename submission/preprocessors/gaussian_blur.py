import cv2

def gaussian_blur(im, k=3):
    # Apply Gaussian blurring to the image
    im_blurred = cv2.GaussianBlur(im, (k, k), 0)
    return im_blurred