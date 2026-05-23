# KMNIST Image Classification with PyTorch

This project demonstrates how to classify KMNIST (Kuzushiji-MNIST) images using various neural network architectures implemented in PyTorch. It explores different models, from simple linear models to convolutional neural networks (CNNs), to compare their performance in image classification.

## Table of Contents
- [Project Description](#project-description)
- [Setup](#setup)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)

## Project Description
The goal of this project is to build and train PyTorch models to accurately classify images from the KMNIST dataset, which consists of 10 different classes of Kuzushiji (Japanese cursive hiragana characters). We implement and evaluate three different model architectures to understand the impact of linearity, non-linearity, and convolutional layers on classification performance.

## Setup
To run this notebook, you will need the following Python libraries. They can be installed using pip:

```bash
pip install torch torchvision matplotlib tqdm torchmetrics mlxtend pandas numpy
```

The `helper_functions.py` file is automatically downloaded if not present, which provides utilities like `accuracy_fn`.

## Dataset
The KMNIST dataset is used, which is a drop-in replacement for the original MNIST dataset but features 10 classes of Kuzushiji characters. The dataset is loaded using `torchvision.datasets.KMNIST`.

- **Training Data**: 60,000 images
- **Testing Data**: 10,000 images
- **Image Size**: 1x28x28 (1 color channel, 28x28 pixels)
- **Classes**: 10 classes (`o`, `ki`, `su`, `tsu`, `na`, `ha`, `ma`, `ya`, `re`, `wo`)

DataLoaders are set up with a batch size of 32 for efficient training.

## Models
Three different models are implemented and evaluated:

1.  **KMNISTModelV0 (Linear Model)**:
    - A simple linear model with a `Flatten` layer followed by two `Linear` layers.
    - Architecture: `Flatten -> Linear -> Linear`

2.  **KMNISTModelV1 (Non-Linear Model)**:
    - A linear model enhanced with `ReLU` activation functions between linear layers.
    - Architecture: `Flatten -> Linear -> ReLU -> Linear -> ReLU`

3.  **KMNISTModelV2 (Convolutional Neural Network - CNN)**:
    - A more complex model utilizing convolutional layers, `ReLU` activations, and `MaxPool2d` layers.
    - Architecture: Two blocks of `Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d`, followed by a `Flatten` and `Linear` classifier.

All models are trained using `CrossEntropyLoss` as the loss function and `SGD` (Stochastic Gradient Descent) as the optimizer.

## Results
The models were trained for 3 epochs each, and their performance (loss and accuracy) on the test dataset, along with training times, are summarized below:

| model_name    | model_loss | model_acc | training_time |
| :------------ | :--------- | :-------- | :------------ |
| KMNISTModelV0 | 1.023566   | 69.488818 | 28.067011     |
| KMNISTModelV1 | 0.919101   | 72.404153 | 32.510032     |
| KMNISTModelV2 | 0.402964   | 88.588259 | 39.150853     |

As observed, the Convolutional Neural Network (KMNISTModelV2) achieves significantly higher accuracy with lower loss compared to the linear and non-linear models, demonstrating the effectiveness of CNNs for image classification tasks. The training time is slightly higher for the CNN due to its increased complexity.

## Usage
To use this project:

1.  Run all cells in the Jupyter/Colab notebook.
2.  Explore the data loading, preprocessing, model definitions, training loops, and evaluation metrics.
3.  Modify model architectures, hyperparameters (e.g., learning rate, epochs, hidden units), or data transformations to experiment with different settings.
4.  The trained `KMNISTModelV2` is saved to `models/Pytorch_Japan_KMNIST.pth` and can be reloaded for inference.
