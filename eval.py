###
### This file contains evaluation metrics to assess the similarity between 
### two images.
### Last updated: 2022/05/04 9:15 AM
### 

# Import libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as pSNR


def mean_squared_error(y_true, y_pred):
    """
    Calculates the Mean Squared error of a set of images
    Input shape  : (frames, x, y)
    Output shape : (frames)
    """
    diff_sqr = np.square(y_true - y_pred)
    mse      = np.mean(diff_sqr, axis=(1,2))
    return mse

def struct_simil_index(y_true, y_pred):
    """
    Calculates the Structural Similarity Index Measurement of a set of images
    Input shape  : (frames, x, y)
    Output shape : (frames)
    """
    ssims = [ssim(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    return np.array(ssims)

def peak_SNR(y_true, y_pred):
    """
    Calculates the Peak Signal-to-Noise Ratio (pSNR) of a set of images
    Input shape  : (frames, x, y)
    Output shape : (frames)
    """
    img_pSNR = [pSNR(y_true[i], y_pred[i]) for i in range(y_true.shape[0])] 
    return np.array(img_pSNR)


def show_img(inp, ref, pred):
    plt.figure(figsize=(15, 15))
    # Show input image, ground truth and predicted image
    display_list = [inp, ref, pred]
    title = ['Input Image', 'Reference Image', 'Predicted Image']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(np.swapaxes(display_list[i], 0,1), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.colorbar()
    plt.show()