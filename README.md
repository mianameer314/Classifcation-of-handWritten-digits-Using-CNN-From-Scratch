# Custom Convolutional Neural Network (CNN) from Scratch

This project implements a Convolutional Neural Network (CNN) from scratch using NumPy. The model is designed to classify handwritten digits from the `digits` dataset provided by `sklearn`. It includes custom implementations of convolutional layers, activation functions, pooling, and softmax layers.

## Features
- **Custom CNN Components**:
  - Convolutional Layer
  - ReLU Activation
  - Max Pooling
  - Softmax Layer
- **Dataset Preprocessing**:
  - Normalization of pixel values
  - One-hot encoding of labels
  - Reshaping for CNN compatibility
- **Training**:
  - Mini-batch gradient descent
  - Adjustable learning rate and batch size
  - Early stopping based on target accuracy
- **Evaluation**:
  - Test set accuracy calculation

## Project Structure
- **Notebook**: `Custom-CNN-from-Scratch.ipynb`
  - Contains the implementation, training, and evaluation of the CNN.
- **Dataset**: `digits` dataset from `sklearn`.

## Steps to Run
1. Clone the repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install numpy scikit-learn
   ```
3. Open the notebook `Custom-CNN-from-Scratch.ipynb` in VSCode or Jupyter Notebook.
4. Run the cells sequentially to:
   - Import libraries
   - Preprocess the dataset
   - Define CNN components
   - Train the model
   - Evaluate the model

## Results
- The model achieves a target accuracy of 90% on the training set.
- Test accuracy is calculated after training.

## Custom CNN Components
### 1. Convolutional Layer
Extracts features from the input using learnable filters.

### 2. ReLU Activation
Applies non-linearity to the feature maps.

### 3. Max Pooling
Reduces the spatial dimensions of the feature maps.

### 4. Softmax Layer
Outputs probabilities for each class.

## Training Parameters
- **Epochs**: 100
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Target Accuracy**: 90%

## Evaluation
The model is evaluated on the test set to calculate its accuracy. The test accuracy is printed at the end of the notebook.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
