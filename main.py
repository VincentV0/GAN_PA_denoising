###
### This file contains the main pipeline for training the GAN model.
### Functions are used from 'training.py', 'data_utils.py' and 'eval.py'.
### Last updated: 2022/05/04 9:15 AM
###


# Import external libraries
import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd

# Import from other files
from training import model_train
from data_utils import Dataset
from eval import mean_squared_error, peak_SNR, show_img, struct_simil_index

# Set constants
PATCH_SIZE       = (128,128)
MODEL_INPUT_SIZE = (128,128,1)
IMAGE_SIZE       = (128,896)
SAVE_PATH_MODELS = 'saved_models'
SAVE_PATH_CSV    = os.path.join('logs','csv', datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + '.csv')

train = Dataset('data/RFdata_train.mat','RF_train_single', 'RF_train_avg', PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 
val   = Dataset('data/RFdata_val.mat'  ,'RF_val_single'  , 'RF_val_avg'  , PATCH_SIZE, combine_n_frames=1, normalize=True, svd_denoise=['ref']) 

# Train the model
Gen, Disc, losses = model_train(train.patches, train.patches_ref, val.patches, val.patches_ref, epochs=500, img_shape=MODEL_INPUT_SIZE, batch_size=5)

# Save the fully trained models
Gen.save(os.path.join(SAVE_PATH_MODELS, f'Generator_{datetime.now().strftime("%Y%m%d-%H_%M_%S")}'))

# Write losses to CSV-file
pd.DataFrame(losses).to_csv(SAVE_PATH_CSV)

# Predict patches and restore patches in original image
val.model_results = np.array(Gen(val.patches)) # predict model results; use generator to predict result of patches
y_pred = val.revert_patching()

# Show input image, ground truth and predicted image
show_img(val.data[0], val.data_ref[0], y_pred[0])



### Metrics
y_true   = val.data_ref.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1])
# Calculate Mean Squared Error:
print(f"MSE (avg over all frames)  : {np.mean(mean_squared_error(y_true, y_pred))}")
# Calculate SSIM
print(f"SSIM (avg over all frames) : {np.mean(struct_simil_index(y_true, y_pred))}")
# Calculate pSNR
print(f"pSNR (avg over all frames) : {np.mean(peak_SNR(y_true, y_pred))}")
