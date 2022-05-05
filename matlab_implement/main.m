clear all; close all;                                       %#ok<CLALL> 
%% Set parameters:
PATCH_SIZE   = [128 128];   % patch size
IMAGE_SIZE   = [896 128];   % size of the original image
COMBINE_IMGS = 1;           % must be an integer

% Set model path
MODEL_PATH = '../saved_models/';

% Set model name (name of folder the model is stored into)
MODEL_FILENAME = [ MODEL_PATH 'Generator_20220505-09_21_54'];

%% Set up CuDNN support and import model (~ 1 min)
disp('Initializing GPU...')
envCfg = coder.gpuEnvConfig('host');
envCfg.DeepLibTarget = 'cudnn';
envCfg.DeepCodegen = 1;
envCfg.Quiet = true;
coder.checkGpuInstall(envCfg);

% Import trained model
model = importTensorFlowNetwork(MODEL_FILENAME, ...
                                "TargetNetwork", "dlnetwork");

%% Data acquisition

% Acquire single image (Verasonix implementation here! For now, use the
% pseudo-data)
load('../data/RFdata_val.mat')
while true
i2 = randi(9);
img = reshape(RF_val_single(:, i2, :), 1, IMAGE_SIZE(1), IMAGE_SIZE(2));

%% Image processing
% at this point, the data should have a shape of (frames, imagesize_x, imagesize_y)

%% Pre-process data for model input
tic % timer starting here

% Combine frames if necessary
img = mean(img, 1);

% Normalize data (data shape at this point should be (1, x, y)); only a
% single frame as input to the model!
img = normalize_img(img);

% Extract patches from image
patches = patch_extract(img, PATCH_SIZE);

% Re-structure the data to fit into the model
dl_patches = dlarray(patches, "BSSC");

% Let the model predict
output = extractdata(model.predict(dl_patches));
output = permute(output, [4 1 2 3]);
output = reshape(output, size(patches));

% Reconstruct patches
img_output = patch_reconstruct(output, IMAGE_SIZE);
toc % timer stopping here

%% Show results
subplot(1,3,1);
x_plot = squeeze(RF_val_avg(:,i2,:));
imshow(x_plot)
title('Reference')

subplot(1,3,2);
imshow(squeeze(img))
title('Model input')

subplot(1,3,3); 
imshow(squeeze(img_output))
title('Predicted')
w = waitforbuttonpress;
end