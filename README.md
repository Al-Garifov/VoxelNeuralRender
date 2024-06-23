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

Render is done using clear and reflective material to check if it is possible for neural network to predict reflections and shadows.

![image](https://github.com/Al-Garifov/VoxelNeuralRender/assets/113169696/40b0751c-404b-4c5f-9989-aae2235dea91)


#### Model
Is intended to consist of three parts: 
- 3D convolution starting from raw data (may be with use of Fourier Feature Mapping)
- resample
- 2D deconvolution to final render prediction.

I own 3060 RTX 12GB so it is a limiting factor to be able to train model in adequate time using this GPU.

Architecture v001 (U-Net style connections should be added in next versions):

![image](https://github.com/Al-Garifov/VoxelNeuralRender/assets/113169696/a2f58e27-c694-4dc5-b372-4defabde6f7d)

```==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ConvDeconv                               [1, 1, 256, 256]          --
├─Conv3d: 1-1                            [1, 64, 128, 128, 128]    4,160
├─Conv3d: 1-2                            [1, 64, 128, 128, 128]    4,160
├─Conv3d: 1-3                            [1, 128, 64, 64, 64]      524,416
├─Conv3d: 1-4                            [1, 128, 64, 64, 64]      16,512
├─Conv3d: 1-5                            [1, 512, 32, 32, 32]      4,194,816
├─Conv3d: 1-6                            [1, 512, 32, 32, 32]      262,656
├─Conv3d: 1-7                            [1, 1024, 16, 16, 16]     33,555,456
├─Conv3d: 1-8                            [1, 1024, 16, 16, 16]     1,049,600
├─ConvTranspose2d: 1-9                   [1, 512, 128, 128]        8,389,120
├─ConvTranspose2d: 1-10                  [1, 512, 128, 128]        262,656
├─ConvTranspose2d: 1-11                  [1, 1, 256, 256]          8,193
==========================================================================================
Total params: 48,271,745
Trainable params: 48,271,745
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 589.34
==========================================================================================
Input size (MB): 67.11
Forward/backward pass size (MB): 3154.64
Params size (MB): 193.09
Estimated Total Size (MB): 3414.84
==========================================================================================
```

___
More docs are coming when more code is written.
___

Feel free to contact me via [al.garifov@gmail.com](mailto:al.garifov@gmail.com) or [LinkedIn Alexey Garifov](https://www.linkedin.com/in/alexey-garifov/) if you have any questions.

