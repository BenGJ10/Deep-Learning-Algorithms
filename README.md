# Deep Learning Algorithms from Scratch

This repository contains implementations of various deep learning algorithms from scratch, focusing on understanding the underlying mechanics without relying on high-level libraries like TensorFlow or PyTorch. The goal is to provide a clear and educational approach to building neural networks and other deep learning models.


## **Neural Network from Scratch**: <br>
A comprehensive guide to building neural networks, including coding neurons and layers, implementing activation functions, and training models using backpropagation. 
<br>

### Implemented Features

#### 1. Neuron and Layer Fundamentals

- Manual implementation of single neurons with arbitrary numbers of inputs and weights.
- Construction of layers as collections of neurons, including manual and loop-based approaches.
- Explanation and demonstration of the mathematical foundation of neuron output: weighted sum plus bias.

#### 2. Vectorized Operations with NumPy

- Efficient computation of neuron and layer outputs using NumPy's vectorized operations.
- Use of matrix multiplication (`np.dot`) for batch processing and multi-layer networks.
- Introduction to matrix multiplication rules and broadcasting for neural network computations.

#### 3. Dense (Fully Connected) Layer Implementation

- Creation of a reusable `DenseLayer` class supporting arbitrary input and output sizes.
- Initialization of weights and biases with appropriate scaling.
- Forward propagation through dense layers for both single samples and batches.

#### 4. Activation Functions

- Implementation of the ReLU (Rectified Linear Unit) activation function, including both forward and backward passes.
- Implementation of the Softmax activation function for multi-class classification, with numerical stability considerations.
- Explanation of the importance of non-linearity and activation function selection.

#### 5. Loss Functions

- Implementation of the categorical cross-entropy loss function for both sparse and one-hot encoded targets.
- Numerical stability via prediction clipping.
- Calculation of average loss over batches.
- Introduction to the distinction between loss and cost functions.

#### 6. Model Evaluation: Accuracy

- Calculation of classification accuracy for both sparse and one-hot encoded targets.
- Use of `np.argmax` for prediction extraction and comparison with ground truth.

#### 7. Optimization Strategies

- Random search for weights and biases to minimize loss (Strategy 1).
- Iterative random adjustment of weights and biases with loss-based acceptance (Strategy 2).
- Discussion of the limitations of naive optimization and the motivation for gradient-based methods.

#### 8. Backpropagation and Gradient Computation

- Step-by-step derivation and implementation of backpropagation for single neurons and full layers.
- Calculation of gradients with respect to weights, biases, and inputs using matrix operations.
- Implementation of backward methods for both dense layers and activation functions.
- Integration of loss gradients for end-to-end training.

#### 9. Batch Processing and Broadcasting

- Use of NumPy broadcasting for efficient batch operations.
- Explanation and demonstration of axis-based operations (`axis=0`, `axis=1`, `keepdims=True`).
- Handling of batch inputs and outputs throughout the network.


### **Single Perceptron**

- Implementation of a basic single-layer perceptron for binary classification.
- Demonstration of perceptron learning rule and convergence on linearly separable data.
