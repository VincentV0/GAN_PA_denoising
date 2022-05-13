function initialize_python(save_path_model)
%INITIALIZE_PYTHON Summary of this function goes here
%   Detailed explanation goes here

% Set Python environment
if isempty(pyenv().Version)
    pyenv('Version','C:\Users\vince\Miniconda3\envs\internship\pythonw.exe')
end

% add current path to python path
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% Load model
py.test_model_function.load_test_model_matlab_call(save_path_model)

end

