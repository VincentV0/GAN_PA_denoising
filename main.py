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
from eval import mean_squared_error, peak_SNR, struct_simil_index

# Set constants
PATCH_SIZE       = (128,128)
MODEL_INPUT_SIZE = (128,128,1)
IMAGE_SIZE       = (256,256)
SAVE_PATH_MODELS = 'saved_models'
SAVE_PATH_CSV    = os.path.join('logs','csv', datetime.now().strftime("%Y.%m.%d-%H.%M.%S") + '.csv')

d = Dataset('pseudo8_realnoise.mat','x', PATCH_SIZE, combine_n_frames=1, normalize=True) 

# Train the model
Gen, Disc, losses = model_train(d.patches, d.patches_ref, d.patches, d.patches_ref, epochs=150, img_shape=MODEL_INPUT_SIZE, batch_size=10) # <<<<<<<< change training data / validata

# Save the fully trained models
Gen.save(os.path.join(SAVE_PATH_MODELS, f'Generator_{datetime.now().strftime("%Y%m%d-%H_%M_%S")}'))

# Write losses to CSV-file
pd.DataFrame(losses).to_csv(SAVE_PATH_CSV)

# Predict patches andd restore patches in original image
d.model_results = np.array(Gen(d.patches)) # predict model results; use generator to predict result of patches <<<<<<<<<<<<< has to be validata 
y_pred = d.revert_patching()

plt.figure(figsize=(15, 15))
# Show input image, ground truth and predicted image
display_list = [d.data[0,0], d.data_ref[0,0], y_pred[0]]
title = ['Input Image', 'Reference Image', 'Predicted Image']
for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()
plt.show()


### Metrics
y_true   = d.data_ref.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1]) # <<<<<<<<<<<<<<<<< has to be validata
# Calculate Mean Squared Error:
print(f"MSE (avg over all frames)  : {np.mean(mean_squared_error(y_true, y_pred))}")
# Calculate SSIM
print(f"SSIM (avg over all frames) : {np.mean(struct_simil_index(y_true, y_pred))}")
# Calculate pSNR
print(f"pSNR (avg over all frames) : {np.mean(peak_SNR(y_true, y_pred))}")
