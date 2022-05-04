function [img_output] = predict(dl_model, img_input)
%PREDICT Function is used to generate the denoised image
%   Use the deep learning model and an image as input.
%   Image dimensions: (dim1xdim2x1) (patch-based evaluation)

img_output = dl_model.predict(img_input);
end

