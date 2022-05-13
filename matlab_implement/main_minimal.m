clear all; close all
%% Initialize Python environment
tic
save_path_model = '../saved_models/Generator_20220509-14_48_25';
initialize_python(save_path_model)
toc

%% Load data and set constants
load('../data/RFdata_train.mat')
IMAGE_SIZE = [128 896];
PATCH_SIZE = [128 128];
USE_PATCHES = 0;
MAKE_PLOT   = 0;

RF_single = permute(RF_train_single, [2 3 1]);
RF_ref    = permute(RF_train_avg,    [2 3 1]);

RF_single = normalize_img(RF_single);
RF_ref    = normalize_img(RF_ref);

NR_IMGS = size(RF_single, 1);
%%
for i=1:70
%i = randi(NR_IMGS);
tic
pred = predict_python(RF_single(i,:,:));
toc

if MAKE_PLOT == 1
    subplot(1,3,1);
    colormap gray
    x_plot = permute(squeeze(RF_ref(i,:,:)),[2 1]);
    imshow(x_plot)
    title('Reference')
    
    subplot(1,3,2);
    imshow(permute(squeeze(RF_single(i,:,:)),[2 1]))
    title('Model input')
    
    subplot(1,3,3); 
    imshow(permute(squeeze(pred),[2 1]))
    title('Predicted')
    w = waitforbuttonpress;
end
end