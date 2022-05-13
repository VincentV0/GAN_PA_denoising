function [img_output] = predict_python_patches(img_input, image_size, patch_size)
%PREDICT_PYTHON_PATCHES Summary of this function goes here
%   Detailed explanation goes here
patches_input = patch_extract(img_input, patch_size);
x = py.test_model_function.test_model_matlab_call(py.numpy.array(patches_input));
op = double(x);
img_output = patch_reconstruct(op, image_size);
end

