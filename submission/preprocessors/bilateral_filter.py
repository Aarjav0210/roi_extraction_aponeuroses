import cv2

def bilateral_filter(im):
    # Apply bilateral filtering to the image
    im_bilateral = cv2.bilateralFilter(im, 9, 75, 75)
    return im_bilateral