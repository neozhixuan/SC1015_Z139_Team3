# SC1015 Project: CIFAR-10 Image Classification and Analysis using Convolutional Neural Networks and Influence Functions

## About
This project aims to demonstrate our understanding of Data Science and Artificial Intelligence concepts in the context of image classification and analysis, focusing on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

In this repository, you will find three Jupyter notebooks, EDA.ipynb, convolutionalneuralnetwork.ipynb, and influencefunctions.ipynb, which contain the following components:

   1) **[exploratoryanalysis.ipynb](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/exploratoryanalysis.ipynb)** : This notebook contains an Exploratory Data Analysis (EDA) of the CIFAR-10 dataset. It involves a thorough examination of the dataset, its structure, and its properties. This EDA helps us gain insights into the dataset and informs our decisions on preprocessing and model selection for the image classification task.

   2) **[convolutionalneuralnetwork.ipynb](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/convolutionalneuralnetwork.ipynb)** : This notebook demonstrates the process of classifying images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in PyTorch. The CNN architecture is designed to learn and identify the features of the images effectively, resulting in accurate classification of the images into their respective labels. The notebook also includes data preprocessing, model training, evaluation, and visualization of the results. The accuracy scores of the model are presented and discussed in detail.

  3) **[influencefunctions.ipynb](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/influencefunctions.ipynb)** : This notebook focuses on the analysis of individual images and their impact on the classification performance of the trained model. By implementing influence functions in PyTorch, with reference to the [influence function implementation](https://github.com/nimarb/pytorch_influence_functions), we investigate the contribution of each image in the dataset, identifying the most influential images. This analysis helps us understand the model's behavior and provides insights into potential improvements in the classification performance.The results of this analysis are saved as [influence_results_0_1.json](https://github.com/neozhixuan/SC1015_Z139_Team3/blob/main/results/influence_results_0_1.json), which can be found in the dataset folder of the repository.





## Contributors
   - [Neo Zhi Xuan](https://github.com/neozhixuan)
     - Exploratory Data Analysis
   - [Ng Zhuo Quan](https://github.com/blanknew)
     - Convolutional Neural Network
   - [Darius](https://github.com/Unknownplaylist)
     - Influence Functions & Assist in Model
 
    
## Problem Definition
The objective of this project is to classify images from the CIFAR-10 dataset into one of 10 possible classes. The CIFAR-10 dataset is a widely-used benchmark dataset for image classification tasks and consists of 60,000 32x32 color images. These images are divided into 10 classes, with 6,000 images per class. The 10 classes are:
   - Airplane
   - Automobile
   - Bird
   - Cat
   - Deer
   - Dog
   - Frog
   - Horse
   - Ship
   - Truck
   
Our goal is to develop a machine learning model, specifically a convolutional neural network (CNN), capable of accurately identifying the class of each image. Additionally, we aim to analyze the influence of individual images on the model's classification performance to gain insights into the model's behavior and identify areas for potential improvement.

The problem can be formalized as a multi-class classification task, where the input is a 32x32 color image, and the output is a class label from the 10 possible classes. The performance of the model can be assessed using metrics such as accuracy, precision, recall, and F1-score.

## Exploratory Analysis of Images
To begin, we analysed the images and its distribution of pixels. The results were:
- Mean: tensor([0.4914, 0.4822, 0.4465])
- Std: tensor([0.2023, 0.1994, 0.2010])

We normalised it using PyTorch, getting these results:
- Mean: tensor([-1.5628e-06, -2.0851e-04,  1.5382e-04])
- Std: tensor([1.0000, 1.0001, 0.9998])

<p float="left">
  <img src="https://user-images.githubusercontent.com/79783660/232287332-32155a44-a93b-4d8c-84da-6c0240526ad7.png" height="300" width="auto">
  <img src="https://user-images.githubusercontent.com/79783660/232287348-68f26f09-4edd-4a6e-af94-e98ab85edf87.png" height="300" width="auto">
</p>
After which, we analysed the color distribution and pixel distributions to detect biases in data. We plot the data using subplots and histograms, and the results were relatively centered at the mean (0), meaning that the data is well balanced and normalised. 

![image](https://user-images.githubusercontent.com/79783660/232287391-630bfc05-f2b0-428e-bafb-fdc638f5d633.png)
Finally, we analysed the PCA distribution to detect whether the images are distinct and classifiable. The plot shows distinct groups of images where similar classes are grouped closely to each other, thus the normalisation has made it simpler for our CNN to classify the images.

To conclude, the post-normalisation data has helped us normalise our images and ensure that it is well prepared for classification.

## Model Architecture
In this section, we present the model architecture used for classifying images from the CIFAR-10 dataset. The model is a convolutional neural network (CNN) and is implemented using PyTorch. Below is the code defining the Net class that describes the structure of the CNN:

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
The model architecture consists of the following layers:

  1) Convolutional layer conv1 with 3 input channels, 6 output channels, and a 5x5 kernel size.
  2) Max pooling layer pool with a 2x2 window size.
  3) Convolutional layer conv2 with 6 input channels, 16 output channels, and a 5x5 kernel size.
  4) Fully connected layer fc1 with an input size of 16 * 5 * 5 and an output size of 120.
  5) Fully connected layer fc2 with an input size of 120 and an output size of 84.
  6) Fully connected layer fc3 with an input size of 84 and an output size of 10.
  
The forward function defines the forward pass of the model, where the input tensor x is transformed by the various layers of the network, ultimately producing an output tensor with 10 channels, corresponding to the 10 classes in the CIFAR-10 dataset.

## Issues Faced and Reference Fixes
1. Error: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
   - Solution: https://github.com/nimarb/pytorch_influence_functions/issues/34

2. Error: Index error within Influence Function 
   - Solution: Update calc_influence_function
     - default value for test_start_index is 'None' instead of False
     - fixes bug when test_start_index is set to 0
     - [Forked changes](https://github.com/expectopatronum/pytorch_influence_functions/commit/ecce2d27e3d46b3125bb3dd963beebd7a5407959)
    
## Key Takeaways from the CIFAR-10 Image Classification Project
Dataset understanding: Gained insights into the CIFAR-10 dataset through exploratory data analysis, informing preprocessing and model selection decisions.

   - **Convolutional Neural Networks:** Acquired hands-on experience in designing, implementing, and evaluating a CNN using PyTorch for image classification.
   - **Performance metrics:** Learned to assess model performance using metrics such as accuracy, precision, recall, and F1-score, enabling quantitative evaluation and identification of improvement areas.
   - **Influence functions:** Explored the concept of influence functions for analyzing individual image contributions to the model's performance, providing insights into model behavior.
   - **Interpreting results:** Understood the characteristics of influential images and their impact on the model, guiding potential refinements and data collection efforts.
   - **Iterative improvement:** Emphasized the importance of continuous model refinement based on performance evaluation and influence analysis in data science and artificial intelligence projects.

## Conclusion

Our analysis using influence functions in the influencefunctions.ipynb notebook revealed some interesting insights about the most influential images in the CIFAR-10 dataset. For the 'plane' class, we observed that the images with harmful influence predominantly resemble planes, while the helpful ones do not seem to have a strong resemblance to planes at all.

![image](https://user-images.githubusercontent.com/44828267/230854030-67094293-cb80-4a4b-9659-50790e1cdb28.png)

This observation suggests that the model may struggle to differentiate between certain planes and other classes, leading to misclassifications. The helpful images, which do not resemble planes, may aid the model in identifying the distinguishing features of other classes and consequently improve classification accuracy.

These findings emphasize the importance of examining individual image contributions to better understand the model's behavior and identify potential areas for improvement. By addressing such issues, we can further enhance the model's performance and achieve a more robust image classification system.

## References
- https://www.cs.toronto.edu/~kriz/cifar.html
- https://pytorch.org/
- https://scikit-learn.org/stable/
- https://github.com/nimarb/pytorch_influence_functions
- https://github.com/nimarb/pytorch_influence_functions/issues/34
- https://github.com/expectopatronum/pytorch_influence_functions/commit/ecce2d27e3d46b3125bb3dd963beebd7a5407959
- https://slidesgo.com/theme/computer-science-mathematics-major-for-college-software-media-applications#search-computer+science&position-4&results-34&rs=search
