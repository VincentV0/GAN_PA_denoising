###
### This file contains the Dataset-class, which can be used to load the 
### MatLab data files into Python and perform pre-processing steps.
### Last updated: 2022/05/04 9:15 AM 
###

# Import libraries
import numpy as np
import h5py
from scipy.io import loadmat


### Define the Dataset class
class Dataset:
    def __init__(self, filename, single_variable_name="", ref_variable_name="", patch_size=None, combine_n_frames=1, 
                    normalize=True, svd_denoise=[]):
        # Set variables in `self`
        self.filename         = filename
        self.single_var_name  = single_variable_name
        self.ref_var_name     = ref_variable_name
        self.patch_size       = patch_size
        self.combine_n_frames = combine_n_frames
        self.normalize        = normalize
        self.svd_denoise      = svd_denoise  # should contain 'ref' and/or 'single'
        self.model_results    = None # reserve variable for later

        assert int(self.combine_n_frames) == self.combine_n_frames and self.combine_n_frames >= 1, "combine_n_frames should be an integer >= 1"

        # Load data and set data shape
        self.data, self.data_ref = self.load_data()

        # Combine multiple frames (only if > 1)
        if combine_n_frames > 1:
            raise NotImplementedError('Combining frames has been disabled after change in data types')
            #self.data = self.combine_frames()
        
        # Add channel axis to be able to run this model
        self.data = self.data[..., np.newaxis]
        self.data_ref = self.data_ref[..., np.newaxis]

        # Create patches
        if self.patch_size != None:
            print(self.data.shape)
            # assume an image shape of (frame, x, y)
            self.patches_x = self.data.shape[1] // self.patch_size[0]
            self.patches_y = self.data.shape[2] // self.patch_size[1]

            # calculate number of patches per image
            self.patches_per_img = self.patches_x*self.patches_y
            assert self.data.shape[1] % self.patch_size[0] == 0 or self.data.shape[2] % self.patch_size[1] == 0, "data shape divided by patch size should yield an integer"

            # calculate patches
            self.patches     = self.create_patches(self.data)
            self.patches_ref = self.create_patches(self.data_ref)

            # Add channel axis to be able to run this model
            self.patches     = self.patches[..., np.newaxis]
            self.patches_ref = self.patches_ref[..., np.newaxis]


    def load_data(self):
        """
        Load the data from the .mat file and do some processing steps.
        Still to be updated!
        """
        # Load the data from the .mat file
        data, data_ref = self.load_mat_as_np()
        
        # Normalize data (Gaussian normalization per frame)
        if self.normalize:
            xmin = np.min(data, axis=(1,2))[..., np.newaxis, np.newaxis]
            xmax = np.max(data, axis=(1,2))[..., np.newaxis, np.newaxis]
            data = (data-xmin) / (xmax-xmin)

            xmin = np.min(data_ref, axis=(1,2))[..., np.newaxis, np.newaxis]
            xmax = np.max(data_ref, axis=(1,2))[..., np.newaxis, np.newaxis]
            data_ref = (data_ref-xmin) / (xmax-xmin)
        
        if 'ref' in self.svd_denoise:
            data_ref = np.array([self.svd_denoise_image(img) for img in data_ref])
        if 'single' in self.svd_denoise:
            data     = np.array([self.svd_denoise_image(img) for img in data])

        return data, data_ref


    def load_mat_as_np(self):
        """
        Reads a .mat file and writes it to a NumPy Array.
        The .mat file should have the following structure:
        - < update when data is available >
        - < >
        """

        try: # In case it is an hdf5-based .mat file
            f        = h5py.File(self.filename, 'r')
            data     = f.get(self.single_var_name)
            data_ref = f.get(self.ref_var_name)
            # Convert to NumPy array
            data     = np.array(data)
            data_ref = np.array(data_ref)

        except OSError: # In case it is a MatLab v7 based file
            mat      = loadmat(self.filename)
            data     = np.array(mat[self.single_var_name])
            data_ref = np.array(mat[self.ref_var_name])
        
        data = np.transpose(data, axes=[1,2,0])   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< check whether axes are still correct; 
                                                  # make sure that the format becomes (frame, element, timepoint)
        data_ref = np.transpose(data_ref, axes=[1,2,0])

        self.data_size = (data.shape[1], data.shape[2])
        return data, data_ref


    def combine_frames(self):
        """
        Use this function to average multiple frames in order to increase SNR. 
        n_average is the number of frames that should be averaged every time.
        """

        assert len(self.data.shape) == 4, f"data should have four dimensions, has {len(self.data.shape)}"
        # data array: (frames, x, y)

        avg_frames = [np.average(self.data[:, i_frame:(i_frame+self.combine_n_frames)], axis=1) for i_frame in range(0, self.data.shape[1], self.combine_n_frames)]
        return np.array(avg_frames).swapaxes(0,1)
        

    def create_reference_data(self, data_array):
        """
        Create the reference dataset. This means that for every spot, all frames
        should be averaged, thus yielding a single image per spot.
        """
        # Average the frames
        data_ref = np.average(data_array, axis=1)

        # Determine the number of copies that should be made
        nr_of_copies = data_array.shape[1] // self.combine_n_frames
        
        # Copy those
        data_ref = data_ref[:, np.newaxis]
        data_ref_copied = data_ref.repeat(nr_of_copies, axis=1)

        return data_ref_copied


    def create_patches(self, data):
        """
        Generate patches from the data, to simplify the training algorithm.
        """
        # Extract patches from the image and reshape 
        patches     = np.array([data[:, self.patch_size[0]*x:self.patch_size[0]*(x+1), self.patch_size[1]*y:self.patch_size[1]*(y+1)] for x in range(self.patches_x) for y in range(self.patches_y)])
        patches     = patches.reshape(-1, self.patch_size[0], self.patch_size[1])
        
        return patches



    def revert_patching(self):
        """
        Reverts patches that have been created using the `create_patches' function.
        """
        assert self.data_size[0] % self.patches.shape[1] == 0 or self.data_size[1] % self.patches.shape[2] == 0, "data shape divided by patch size should yield an integer"


        # Calculate number of images
        nr_of_images     = self.patches.shape[0] // self.patches_per_img
        data_to_return   = np.zeros((nr_of_images, self.data_size[0], self.data_size[1]))
    
        # Put the data in the right format
        patches     = self.model_results.reshape(self.patches_per_img, nr_of_images, self.patches.shape[1], self.patches.shape[2])
        patches     = patches.swapaxes(0,1)


        # Loop over all frames
        for im in zip(range(nr_of_images)):
            
            # Loop over all patches
            for i, frame in enumerate(patches[im]):
                
                # Determine where the patch is supposed to be
                order = np.arange(self.patches_per_img).reshape(self.patches_x,self.patches_y)
                x,y = np.where(order==i)
                
                # Change type of 'patch-coordinates'
                x,y = int(x), int(y)
                
                # Put the patches back into the image
                data_to_return[im, patches.shape[2]*x:patches.shape[2]*(x+1), patches.shape[3]*y:patches.shape[3]*(y+1)] = frame

        return data_to_return

    def svd_denoise_image(self, img, k=15):
        # Calculate U (u), Σ (s) and V (vh)
        u, s, vh = np.linalg.svd(img, full_matrices=False)
        # Remove all but the k highest sigma values 
        ind = np.argpartition(s, -k)[-k:] # k highest values for sigma
        s_cleaned = np.zeros(s.shape)
        s_cleaned[ind] = s[ind]
        #s_cleaned = np.array([si if si > 250000 else 0 for si in s])
        # Calculate A' = U * Σ (cleaned) * V
        img_denoised = np.array(np.dot(u * s_cleaned, vh))
        return img_denoised


