from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def edge_detection(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    image = image.astype(float)
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]])
    kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]])

    edgeX = convolve2d(image, kernel_x, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(image, kernel_y, mode='same', boundary='fill', fillvalue=0)
    edgeMAG = edgeX**2 + edgeY**2

    return edgeMAG
