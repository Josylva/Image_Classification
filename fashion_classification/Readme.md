```markdown
# Fashion MNIST Classification with PyTorch

This project demonstrates the classification of Fashion MNIST dataset images using various PyTorch models. The goal is to build and compare the performance of a simple linear model, a linear model with non-linear activation functions, and a Convolutional Neural Network (CNN).

## Project Structure

The notebook walks through the following steps:

1.  **Data Loading and Exploration**: Loading the Fashion MNIST dataset using `torchvision.datasets` and exploring its characteristics, such as image shape, number of samples, and class names.
2.  **Data Preparation**: Setting up `DataLoader` instances for batching the training and testing data.
3.  **Model Building**: Implementing three different models:
    *   **`FashionMNISTModelV0`**: A simple linear model.
    *   **`FashionMNISTModelV1`**: A linear model incorporating ReLU non-linearity.
    *   **`FashionMNISTModelV2`**: A Convolutional Neural Network (CNN) based on the TinyVGG architecture.
4.  **Model Training and Evaluation**: Training each model for 3 epochs and evaluating its performance (loss and accuracy) on the test set.
5.  **Prediction and Visualization**: Making predictions on random test samples and visualizing the predictions along with the ground truth labels.
6.  **Confusion Matrix**: Generating and plotting a confusion matrix for the best-performing model.
7.  **Model Saving and Loading**: Demonstrating how to save and load a trained PyTorch model's `state_dict`.

## Models Implemented

### `FashionMNISTModelV0` (Linear Model)

A basic neural network with one hidden linear layer. It takes the flattened image pixels as input and outputs logits for each of the 10 classes.

-   **Architecture**: `Flatten` -> `Linear` -> `Linear`
-   **Activation**: None (linear layers only)

### `FashionMNISTModelV1` (Linear Model with Non-Linearity)

An improved linear model that introduces ReLU activation functions between the linear layers, allowing the model to learn more complex patterns.

-   **Architecture**: `Flatten` -> `Linear` -> `ReLU` -> `Linear` -> `ReLU`
-   **Activation**: ReLU

### `FashionMNISTModelV2` (Convolutional Neural Network - TinyVGG)

A more sophisticated model utilizing convolutional layers, ReLU activations, and max-pooling layers. This architecture is designed to capture spatial hierarchies in the image data.

-   **Architecture**:
    *   Block 1: `Conv2d` -> `ReLU` -> `Conv2d` -> `ReLU` -> `MaxPool2d`
    *   Block 2: `Conv2d` -> `ReLU` -> `Conv2d` -> `ReLU` -> `MaxPool2d`
    *   Classifier: `Flatten` -> `Linear`

## Results Comparison

The following table summarizes the performance of the three models on the Fashion MNIST test set after 3 epochs of training:

| model_name          | model_loss | model_acc | training_time |
|:--------------------|:-----------|:----------|:--------------|
| FashionMNISTModelV0 | 0.476639   | 83.426518 | 34.607019     |
| FashionMNISTModelV1 | 0.685001   | 75.019968 | 32.135674     |
| FashionMNISTModelV2 | 0.326308   | 88.019169 | 39.060669     |

As observed, `FashionMNISTModelV2` (the CNN) achieved the best accuracy and lowest loss, demonstrating the effectiveness of convolutional layers for image classification tasks. Although it took slightly longer to train, its superior performance makes it the preferred model for this task.

## Dependencies

This project relies on the following Python libraries:

*   `torch`
*   `torchvision`
*   `matplotlib`
*   `pandas`
*   `tqdm`
*   `requests`
*   `mlxtend`
*   `torchmetrics`

These can typically be installed via `pip`:

```bash
pip install torch torchvision matplotlib pandas tqdm requests mlxtend torchmetrics
```

Note: `helper_functions.py` is downloaded during the notebook execution.

## How to Run

1.  Open the `.ipynb` file in a Jupyter environment (e.g., Google Colab).
2.  Run all cells sequentially.
3.  Ensure a GPU is available and enabled in your runtime settings for faster training of `FashionMNISTModelV1` and `FashionMNISTModelV2`.
```
