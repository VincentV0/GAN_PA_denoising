###
### 
### 
### Last updated: 2022/05/05 3:50 AM
###


# Import external libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import tensorflow as tf
import time

# Import from other files
from training import model_train
from pix2pix import Generator, generator_loss
from data_utils import Dataset
from eval import mean_squared_error, peak_SNR, show_img, struct_simil_index

# Constants
PATCH_SIZE       = (128,128)
MODEL_INPUT_SIZE = (128,896,1)
IMAGE_SIZE       = (128,896)
SAVE_PATH_MODELS = 'saved_models'
SAVE_PATH_CSV    = os.path.join('logs','csv', datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + '.csv')

# Data loading
train = Dataset('data/RFdata_train.mat','RF_train_single', 'RF_train_avg', PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 
val   = Dataset('data/RFdata_val.mat'  ,'RF_val_single'  , 'RF_val_avg'  , PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 
tes   = Dataset('data/RFdata_test.mat'  ,'RF_test_single'  , 'RF_test_avg'  , PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 

#G = Generator(PATCH_SIZE, 64)
Gen = tf.keras.models.load_model('saved_models/Generator_20220506-12_45_53')


# Predict patches and restore patches in original image
#mr = []
#for data_item in train.patches:
#    start_time = time.time()
#    pred = Gen(data_item[np.newaxis])
#    print(f"Time elapsed: {(time.time()-start_time):.3} secs")
#    mr.append(pred[0])

#train.model_results = np.array(mr)
val.model_results  = np.array(Gen(val.patches)) # predict model results; use generator to predict result of patches
#test.model_results   = np.array(Gen(val))

#y_pred_train = train.revert_patching()
y_pred_val   = val.revert_patching()

for ix in range(20):
   # Show input image, ground truth and predicted image
    show_img(val.data[ix], val.data_ref[ix], y_pred_val[ix])