import cv2

# Edge detection
def edge_detection(im):
    edges = cv2.Canny(im.astype('uint8'), 50, 150)
    return edges