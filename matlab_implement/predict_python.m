function [img_output] = predict_python(img_input)
%PREDICT_PYTHON Summary of this function goes here
%   Detailed explanation goes here

% Generate image from model

x  = py.test_model_function.test_model_matlab_call(py.numpy.array(img_input));
img_output = squeeze(double(x));
end

