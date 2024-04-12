import numpy as np

def layered_min_blur(im):
    steps = [5, 3]
    for step in steps:
        temp = np.zeros((im.shape[0] // step, im.shape[1] // step))
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                temp[i, j] = np.min(im[i*step:(i+1)*step, j*step:(j+1)*step])
        im = temp.copy()

    return im