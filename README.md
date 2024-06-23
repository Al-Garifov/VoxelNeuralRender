# Voxel Neural Render
Implementation of Ray Tracing Engine via Machine Learning techniques, mainly Convolutional Neural Networks.
___
### Structure
Configuration (such as pointing to data roots) could be done by setting global variables in `__init__.py` files.

#### Dataset
Provides functions to work with "raw" data generated from CG software with usage of industrial stardard formats i.e. OpenVDB. 

While **recommended way to interact with data is via prepared ".npz" archives** native to NumPy and understandable by majority of Machine Learning frameworks such as PyTorch mainly used in this project.

Though data is big enough not to fit into RAM of most computers, custom Dataset class can be found in `dataset/prepared.py` capable of loading data by chunks of 100 samples. 

Training data is OpenVDB representation of rigid objects in camera space (in distance from 1 and 1.8 units from camera and 256x256x256 resolution) and rendered images of the objects with static direct light and ground. 
Both VDBs and images utilize only one channel for simplification. 
Multiplying channel number is one of the project's goals if it succeeds with easier tasks.

#### Model
Is intended to consist of three parts: 
- 3D convolution starting from raw data (may be with use of Fourier Feature Mapping)
- resample
- 2D deconvolution to final render prediction.

I own 3060 RTX 12GB so it is a limiting factor to be able to train model in adequate time using this GPU.

___
More docs are coming when more code is written.
___

Feel free to contact me via [al.garifov@gmail.com](mailto:al.garifov@gmail.com) or [LinkedIn Alexey Garifov](https://www.linkedin.com/in/alexey-garifov/) if you have any questions.

