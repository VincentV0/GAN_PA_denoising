from data_utils import *


PATCH_SIZE = (32,32)
IMAGE_SIZE = (384, 1792)

x, _ = load_data('pseudo.mat')

x_avg = np.array([combine_frames(x[i], 2) for i in range(x.shape[0])])
x_ref = create_reference_data(x)
x_patch = np.array([create_patches(x_avg[i], PATCH_SIZE) for i in range(x_avg.shape[0])])
x_ref_patch = np.array([create_patches(x_ref[i, np.newaxis], PATCH_SIZE) for i in range(x_ref.shape[0])])

# Throw it through the model
# ...
model_results = x_patch # we have no model yet

# Restore patches
predicted_denoised = np.array([revert_patching(model_results[i], IMAGE_SIZE) for i in range(model_results.shape[0])])

### export functions -> graphs and such
### other projects for inspiration
### metrics