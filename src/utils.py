# utility
from division import div
from sklearn.feature_selection import f_classif
import numpy as np
import imageio 
from sklearn.decomposition import PCA

def compute_IoU(result_img, mask_img):
    import sys
    eps = sys.float_info.epsilon #2.220446049250313e-16
    result_img = result_img > 0.5
    mask_img = mask_img > 0.5
    I = result_img & mask_img
    U = result_img | mask_img
    # Another way is to set deno=1 where it is 0
    return div(1.0 * np.sum(I), (np.sum(U)+eps))

def compute_mIoU(result_imgs, mask_imgs):
    # Assume result_imgs and mask_imgs are lists
    return np.mean(np.asarray([compute_IoU(result_imgs[i], mask_imgs[i]) for i in range(len(mask_imgs))]))

# Load image
def loadImage(train_img_addr):
    return np.array(imageio.imread(train_img_addr), dtype=np.float64)
