# Image Classification and Influence Functions with PyTorch

## About
This is our project for SC1015 (Introduction to Data Science and Artifical Intelligence) which focuses on classifying images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). 

Firstly, we classify the images dataset into different labels, via a [convolutional neural network](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/convolutionalneuralnetwork.ipynb). Next, we analyse the influence of individual images in the image folders, and find the most influential image within it, using the [an influence function](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/influencefunctions.ipynb), with reference to PyTorch.

## Contributors

## Problem Definition
How can we effectively categorize the influence of the CIFAR-10 dataset on machine learning research and applications?
## Models Used
Convolutional Neural Network
## Issues Faced and Reference Fixes
1. Error: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  - Solution: https://github.com/nimarb/pytorch_influence_functions/issues/34

2. Error: Index error within Influence Function 
  - Solution: Update calc_influence_function
    - default value for test_start_index is 'None' instead of False
    - fixes bug when test_start_index is set to 0
    - [Forked changes](https://github.com/expectopatronum/pytorch_influence_functions/commit/ecce2d27e3d46b3125bb3dd963beebd7a5407959)

## Conclusion
