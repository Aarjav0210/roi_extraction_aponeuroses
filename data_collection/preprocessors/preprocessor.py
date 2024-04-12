from preprocessors.noise_reduction import denoise_image
from preprocessors.bilateral_filter import bilateral_filter
from preprocessors.gaussian_blur import gaussian_blur
from preprocessors.average_blur import layered_average_blur
# from binary_threshold import binary_threshold

def preprocess_image(im, technique, method=None):
    if technique == 'denoise':
        return denoise_image(im)
    elif technique == 'bilateral':
        return bilateral_filter(im)
    elif technique == 'gaussian':
        return gaussian_blur(im)
    # elif technique == 'binary':
    #     return binary_threshold(im, method)
    elif technique == 'layered':
        return layered_average_blur(im)
    else:
        raise ValueError('Invalid preprocessing technique')