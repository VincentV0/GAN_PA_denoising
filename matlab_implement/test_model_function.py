### 
### Last updated: 2022/05/11 10:55 AM
###


# Import external libraries
import os
import numpy as np
import tensorflow as tf

# Constants
PATCH_SIZE       = (128,128)
MODEL_INPUT_SIZE = (128,896,1)
IMAGE_SIZE       = (128,896)

# Load model
def load_test_model_matlab_call(save_path_model):
    global Gen
    Gen = tf.keras.models.load_model(save_path_model)

def test_model_matlab_call(frame):
    pred = Gen(frame)
    return np.array(pred)


# # Predict patches and restore patches in original image
# mr = []
# for data_item in train.data:
#     start_time = time.time()
#     pred = Gen(data_item[np.newaxis])
#     print(f"Time elapsed: {(time.time()-start_time):.3} secs")
#     mr.append(pred)
# train.model_results = np.array(mr)
# #val.model_results   = np.array(Gen(val.data)) # predict model results; use generator to predict result of patches
# #test.model_results   = np.array(Gen(val))

# #y_pred_train = train.revert_patching()
# #y_pred_val   = val.revert_patching()

# #for ix in range(20):
#     # Show input image, ground truth and predicted image
# #    show_img(train.data[ix], train.data_ref[ix], train.model_results[ix])