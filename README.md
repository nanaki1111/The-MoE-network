# The-MoE-network
seismic data denoising
Unsupervised 2D signal denoising based on a Mixture-of-Experts (MoE) model. The model consists of three autoencoder-style expert networks and a softmax gating network with entropy regularization, which encourages balanced participation of the experts. It is suitable for denoising 2D signals with random noise, such as seismic data.

## Project Structure

MoE/
├── MoE.py                 # Main script: model building, training, inference and evaluation
├── utils.py               # Patching and reconstruction utilities
├── sigmoid.mat            # Example data
└── tools/
    ├── Step1_Patching_3D.m  # MATLAB script for 3D data patching
    └── yc_patch3d.m         # MATLAB 3D patching function

## Method Overview

1. **Patching**: The 2D data is split into overlapping 48×48 patches with a shift of 1, which serve as training samples.
2. **Expert Networks**: Three identical fully connected autoencoders (128→64→input) learn different denoising patterns.
3. **Gating Network**: A softmax gate dynamically computes the weights of the three experts for each patch, with entropy regularization to prevent collapse onto a single expert.
4. **Training**: The noisy data itself is used as both input and target (unsupervised), trained with MSE loss and the Adam optimizer.
5. **Evaluation**: The full image is reconstructed, and SNR and SSIM are computed. The noisy, denoised, and residual results are visualized.

## Requirements

- Python 3.9+
- TensorFlow 2.x
- numpy
- scipy
- matplotlib

## 3D Data (MATLAB Tools)

The `tools/` directory provides patching tools for 3D data. See `Step1_Patching_3D.m` for usage; the default window is 15×15×15 with a shift of 1. Reference:

> Yang, Liuqing, et al. "Unsupervised 3-D random noise attenuation using deep skip autoencoder." *IEEE Transactions on Geoscience and Remote Sensing* 60 (2021): 1–16.
