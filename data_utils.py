import numpy as np
import h5py
from scipy.io import loadmat

def load_mat_as_np(filename, variable_name=""):
    """
    Reads a .mat file and writes it to a NumPy Array.
    The .mat file should have the following structure:
     - < update when data is available >
     - < >
    """

    try: # In case it is an hdf5-based .mat file
        f = h5py.File(filename, 'r')
        data = f.get(variable_name)
        data = np.array(data) # For converting to a NumPy array
    except OSError: # In case it is a MatLab v7 based file
        mat = loadmat(filename)
        data = np.array(mat[variable_name])
    else:
        raise ImportError('File could not be loaded.')
    data = np.transpose(data, axes=[3,2,1,0])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                # make sure that the format becomes (spot, frame, element, timepoint)

    return data


def combine_frames(data_array, n_average=1):
    """
    Use this function to average multiple frames in order to increase SNR. 
    n_average is the number of frames that should be averaged every time.
    """

    assert int(n_average) == n_average and n_average > 1, "n_average should be an integer > 1"
    assert len(data_array.shape) == 3, f"data_array should have three dimensions, has {len(data_array.shape)}"
    # data array: (frames, x, y)
    
    avg_frames = [np.average(data_array[i_frame:(i_frame+n_average)], axis=0) for i_frame in range(0, data_array.shape[0], n_average)]
    return np.array(avg_frames)
    

def create_reference_data(data_array):
    """
    Create the reference dataset. This means that for every spot, all frames
    should be averaged, thus yielding a single image per spot.
    """

    return np.average(data_array, axis=1) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axis is still correct


def create_patches(data, patch_size=(128,128)):
    """
    Generate patches from the data, to simplify the training algorithm.
    """
    assert data.shape[1] % patch_size[0] == 0 or data.shape[2] % patch_size[1] == 0, "data shape divided by patch size should yield an integer"
    
    # assume an image shape of (frame, x, y)
    patches_x = data.shape[1] // patch_size[0]
    patches_y = data.shape[2] // patch_size[1]

    # Extract patches from the image and reshape, such that 
    patches     = np.array([data[:,        patch_size[0]*x:patch_size[0]*(x+1),     patch_size[1]*y:patch_size[1]*(y+1)] for x in range(patches_x) for y in range(patches_y)])
    patches     = patches.reshape(-1, patch_size[0], patch_size[1])
    
    return patches



def revert_patching(patches, data_size=(128,128)):
    """
    Reverts patches that have been created using the `create_patches' function.
    """
    assert data_size[0] % patches.shape[1] == 0 or data_size[1] % patches.shape[2] == 0, "data shape divided by patch size should yield an integer"

    # assume an image shape of (frame, x, y)
    patches_x = data_size[0] // patches.shape[1]
    patches_y = data_size[1] // patches.shape[2]
    patches_tot = patches_x*patches_y

    # Calculate number of images
    nr_of_images     = patches.shape[0] // patches_tot
    data             = np.zeros((nr_of_images, data_size[0], data_size[1]))
   
    # Put the data in the right format
    patches     = patches.reshape(patches_x*patches_y, nr_of_images, patches.shape[1], patches.shape[2])
    patches     = patches.swapaxes(0,1) 


    # Loop over all frames
    for im in zip(range(nr_of_images)):
        
        # Loop over all patches
        for i, frame in enumerate(patches[im]):
            
            # Determine where the patch is supposed to be
            order = np.arange(patches_x * patches_y).reshape(patches_x,patches_y)
            x,y = np.where(order==i)
            
            # Change type of 'patch-coordinates'
            x,y = int(x), int(y)
            
            # Put the patches back into the image
            data[im, patches.shape[2]*x:patches.shape[2]*(x+1), patches.shape[3]*y:patches.shape[3]*(y+1)] = frame

    return data


def load_data(filename):
    """
    Load the data from the .mat file and do some processing steps.
    Still to be updated!
    """
    # Load the data from the .mat file
    data_all = load_mat_as_np(filename, "x")
    
    # Normalize data (Gaussian normalization)
    data_all_norm = (data_all - np.mean(data_all))/(np.std(data_all))

    # Create reference data based on the full dataset
    data_ref = create_reference_data(data_all_norm)

    # split into train/val/test!
    
    return data_all, data_ref

