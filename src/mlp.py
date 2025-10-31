import numpy as np
from typing import Callable, List, Tuple

RELU = 'relu'
SIGMOID = 'sigmoid'
TANH = 'tanh'


class MLP():

    def __init__(self, 
                 activation_func: str, 
                 num_hidden_layers: int, 
                 hidden_layer_size: int
                 ):
        """
        Initialize MLP with specified architecture.
        
        Args:
            activation_func: Activation function name ('relu', 'sigmoid', 'tanh')
            num_hidden_layers: Number of hidden layers
            hidden_layer_size: Number of units in each hidden layer
        """
        if activation_func.lower() not in [RELU, SIGMOID, TANH]:
            raise ValueError(f"Unsupported activation function: {activation_func}")
        self.activation_func = activation_func.lower()

        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        
        self.weights = []
        self.biases = []
        self.activated = []
        self.z_values = []
        
        self.is_initialized = False



    ################################################
    #                                              #
    #               Private Methods                #
    #                  (Helpers)                   #
    ################################################
    def _initialize_parameters(self, input_size: int, output_size: int):
        """
        Initialize weights and biases using 'He' initialization.
        'He' is supposed to be best for ReLU, but we use it for all activations for simplicity.
        """
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.randn(input_size, self.hidden_layer_size) * np.sqrt(2.0 / input_size)) # He is scaling factor
        self.biases.append(np.zeros((1, self.hidden_layer_size)))
        
        # Hidden layers
        for i in range(self.num_hidden_layers - 1):
            self.weights.append(np.random.randn(self.hidden_layer_size, self.hidden_layer_size) * np.sqrt(2.0 / self.hidden_layer_size)) # He scaling
            self.biases.append(np.zeros((1, self.hidden_layer_size)))
        
        # Last hidden layer to output layer
        self.weights.append(np.random.randn(self.hidden_layer_size, output_size) * np.sqrt(2.0 / self.hidden_layer_size)) # He scaling
        self.biases.append(np.zeros((1, output_size)))
        
        self.is_initialized = True


    def _activation(self, z):
        """Apply activation function."""
        if self.activation_func == RELU:
            return np.maximum(0, z)
        elif self.activation_func == SIGMOID:
            # Clip input to avoid overflow
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_func == TANH:
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")


    def _activation_derivative(self, z):
        """Compute derivative of activation function."""
        if self.activation_func == RELU:
            # Derivative of ReLU is:
            # 0.0 for z <= 0
            # 1.0 for z > 0
            return (z > 0).astype(float)
        elif self.activation_func == SIGMOID:
            s = self._activation(z)
            # Derivative of Sigmoid is : s * (1 - s)
            return s * (1 - s)
        elif self.activation_func == TANH:
            # Derivative of tanh is : 1 - tanh^2(z)
            return 1 - np.tanh(z) ** 2
        else:
            raise ValueError(f"Unknown activation function: {self.activation_func}")


    def _softmax(self, z):
        """
        Compute softmax for output layer.
        Subtracting max for numerical stability.
        (Equivalent to slides but more stable for large values)
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


    def _forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
        
        Returns:
            Output predictions of shape (batch_size, output_size)
        """
        self.activated = [X]
        self.z_values = []
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activated[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z) 
            a = self._activation(z) 
            self.activated.append(a) 
        
        # Output layer with softmax
        z = np.dot(self.activated[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self._softmax(z)
        self.activated.append(output)
        
        return output


    def _backward(self, y):
        """
        Backward pass to compute gradients.
        
        Args:
            y: True labels of shape (batch_size, output_size)
        
        Returns:
            Gradients for weights and biases
        """
        m = y.shape[0]
        grad_weights = []
        grad_biases = []
        
        # Error at output layer
        delta = self.activated[-1] - y
        
        # Backward through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            grad_w = np.dot(self.activated[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            # "Append to front" (going backwards)
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            
            # Propagate gradient to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.z_values[i-1])
        
        return grad_weights, grad_biases


    def _update_parameters(self, grad_weights, grad_biases, learning_rate):
        """Update weights and biases using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i]  -= learning_rate * grad_biases[i]


    ################################################
    #                                              #
    #          Fit and Prediction Methods          #
    #                                              #
    ################################################
    def fit(self, X, y, learning_rate: float = 0.01, epochs: int = 100, batch_size: int = 32, verbose: bool = True):
        """
        Train the MLP on data X with labels y.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Labels (can be integers or one-hot encoded)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
        """
        # Convert labels to one-hot columns if needed
        if len(y.shape) == 1 or y.shape[1] == 1:
            n_classes = len(np.unique(y))
            y_onehot = np.zeros((y.shape[0], n_classes))
            y_onehot[np.arange(y.shape[0]), y.flatten()] = 1
        else:
            y_onehot = y
            n_classes = y.shape[1]
        
        # Initialize parameters if not already done
        if not self.is_initialized:
            self._initialize_parameters(X.shape[1], n_classes)
        
        n_samples = X.shape[0]
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                # Get current mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward, Backward, Update
                self._forward(X_batch)
                grad_weights, grad_biases = self._backward(y_batch)
                self._update_parameters(grad_weights, grad_biases, learning_rate)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                predictions = self._forward(X)
                loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-8), axis=1))
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")


    def predict(self, X):
        """Predict output for input X."""
        probabilities = self._forward(X)
        return np.argmax(probabilities, axis=1)
    