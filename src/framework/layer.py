import numpy as np 
import cv2
from skimage.measure import block_reduce

def myResize(x, H, W):
    new_x = np.zeros((x.shape[0], H, W, x.shape[3]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[3]):
            new_x[i,:,:,j] = cv2.resize(x[i,:,:,j], (W,H), interpolation=cv2.INTER_CUBIC)
    return new_x

def MaxPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.max)

