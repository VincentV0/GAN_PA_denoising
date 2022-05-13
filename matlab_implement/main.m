%% Initialization
clear all; close all; %#ok<CLALL> 

% Set Python environment (only once)
if isempty(pyenv().Version)
    pyenv('Version','C:\Users\vince\Miniconda3\envs\internship\pythonw.exe')
end

% add path to path
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% Load model
save_path_model = '../saved_models/Generator_20220509-14_48_25';
py.test_model_function.load_test_model_matlab_call(save_path_model)

%% Load data and set constants
load('../data/RFdata_val.mat')
IMAGE_SIZE = [128 896];
PATCH_SIZE = [128 128];
USE_PATCHES = 0;

RF_single = permute(RF_train_single, [2 3 1]);
RF_ref    = permute(RF_train_avg,    [2 3 1]);

RF_single = normalize_img(RF_single);
RF_ref    = normalize_img(RF_ref);

NR_IMGS = size(RF_single, 1);

%% Generate image from model
for i=1:NR_IMGS
    img_input     = RF_single(i, :, :);
    patches_input = patch_extract(img_input, PATCH_SIZE);

    tic
    if USE_PATCHES == 1
        x = py.test_model_function.test_model_matlab_call(py.numpy.array(patches_input));
        op = double(x);
        img_output = patch_reconstruct(op, IMAGE_SIZE);
    else
        x  = py.test_model_function.test_model_matlab_call(py.numpy.array(img_input));
        img_output = squeeze(double(x));
    end
    toc

    %% Show results
    subplot(1,3,1);
    colormap gray
    x_plot = permute(squeeze(RF_ref(i,:,:)),[2 1]);
    imshow(x_plot)
    title('Reference')
    
    subplot(1,3,2);
    imshow(permute(squeeze(img_input),[2 1]))
    title('Model input')
    
    subplot(1,3,3); 
    imshow(permute(squeeze(img_output),[2 1]))
    title('Predicted')
    w = waitforbuttonpress;
end