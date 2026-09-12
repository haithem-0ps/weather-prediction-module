# weather-prediction-module
a weather prediction module made using numpy
## Overview
Instead of jumping straight into high-level frameworks like PyTorch or TensorFlow, I built this project to reverse-engineer how neural networks actually work under the hood. It takes a small toy dataset and trains a multi-layer perceptron using pure NumPy matrix math and gradient descent.

## Features
- **Zero Frameworks:** Uses only NumPy for all matrix multiplications, activations, and weight updates.
- **Manual Backpropagation:** Implements explicit calculations for gradients, deltas, and weight transposes (`W2.T`).
- **Interactive Inference Loop:** Includes a live command-line prompt to test custom input values in real-time.

## Code Structure
- **Dataset Configuration:** Simple input features and target labels.
- **Architecture:** 3 input nodes, 2 hidden nodes, and 1 output node utilizing the sigmoid activation function.
- **Training Loop:** Runs for 10,000 epochs, updating weights and biases via gradient descent.
- **Prediction:** Allows users to input comma-separated values to test the trained model.

## How to Run
1. Make sure you have NumPy installed:
   ```bash
   pip install numpy
