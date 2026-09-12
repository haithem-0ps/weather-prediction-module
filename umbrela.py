import numpy as np

#we just put the data 
X = np.array([
    [0.1, 0.5, 0.2],
    [0.7, 0.1, 0.9],
    [0.3, 0.4, 0.8],
    [0.5, 0.6, 0.1]
])
y = np.array([[0], [1], [1], [0]])

#we do the architecture of the neural network how many neurons in each layer
np.random.seed(42)
input_size, hidden_size, output_size = 3, 2, 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Pure Sigmoid activation for both hidden and output layers
#in numpy there is no relu or sigmoid so we do the alone
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * (1 - sig)

learning_rate = 0.5
epochs = 10000#how many times we want to train the model every time a pass happens it adjust the weights and biases to make the model better

for epoch in range(epochs):
    # Forward Pass this is just math multiplication and adding the bias explantion in phone 6/9/2026
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)       
    z2 = np.dot(a1, W2) + b2
    predictions = sigmoid(z2) 

    output_error = predictions - y
    output_delta = output_error * sigmoid_derivative(predictions)

    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1) 

    # Parameter Updates (Gradient Descent)
    W2 -= learning_rate * np.dot(a1.T, output_delta)
    b2 -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X.T, hidden_delta)
    b1 -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    
while True:
    a=input('please enter the input data for prediction (comma-separated, e.g., 0.1,0.5,0.2): ')
    a=np.array([float(x) for x in a.split(',')])
    z1 = np.dot(a, W1) + b1
    a1 = sigmoid(z1)          
    z2 = np.dot(a1, W2) + b2
    prediction = sigmoid(z2)
    if prediction >= 0.5:
        print("its going to rain")
    else:
        print("its not going to rain")

