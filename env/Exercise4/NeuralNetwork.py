import numpy as np


# Defines a fully-connected neural network with a controllable number of layers and neurons per layer. Assumes the last
# layer is unactivated
class NeuralNetwork:

    def __init__(self, layer_sizes, learning_rate):
        self.layer_sizes = layer_sizes
        self.connections = self.create_connections()
        self.learning_rate = learning_rate
        self.gradients = []
        self.outputs = []  # Stores the intermediate outputs from each layer
        self.unactivated_outputs = []  # Stores intermediate outputs before the activation function is applied

    # Trains the neural network using the selected optimizer
    def train(self, inputs, targets):
        for input_data, target in zip(inputs, targets):
            self.outputs = []
            self.unactivated_outputs = []
            self.gradients = []
            self.feedforward(input_data)
            self.backprop(target)
            self.optimize()

    # Runs inputs through the network to produce an output
    def feedforward(self, input_data):
        output = input_data
        self.unactivated_outputs.append(output)
        self.outputs.append(output)
        for weight, bias in self.connections[:-1]:
            unactivated_output = np.dot(weight, output) + bias
            output = self.activation(unactivated_output)
            self.unactivated_outputs.append(unactivated_output)
            self.outputs.append(output)
        weight, bias = self.connections[-1]
        output = np.dot(weight, output) + bias
        self.unactivated_outputs.append(output)
        self.outputs.append(output)
        return output

    # Propagates gradients back through the network to obtain the gradient of the cost function w.r.t. each weight.
    # Assumes that a feedforward step has already occurred.
    def backprop(self, target):
        weights = [connection[0] for connection in self.connections]
        grad_wrt_output = self.cost_function_grad(self.outputs[-1], target)
        grad_wrt_bias = grad_wrt_output
        grad_wrt_weight = np.outer(grad_wrt_bias, self.outputs[-2])
        self.gradients.append([grad_wrt_weight, grad_wrt_bias])
        for i in range(2, len(self.layer_sizes)):
            grad_wrt_output *= self.activation_prime(self.unactivated_outputs[-i + 1])
            grad_wrt_output = np.dot(weights[-i + 1].T, grad_wrt_output)
            grad_wrt_bias = grad_wrt_output * self.activation_prime(self.unactivated_outputs[-i])
            grad_wrt_weight = np.outer(grad_wrt_bias, self.outputs[-i - 1])
            self.gradients.append([grad_wrt_weight, grad_wrt_bias])
        self.gradients.reverse()

    # Trains the neural network using stochastic gradient descent
    def optimize(self):
        self.connections = list(np.array(self.connections) - self.learning_rate * np.array(self.gradients))

    @staticmethod
    # Rectified Linear Unit (ReLU) activation function
    def activation(x):
        return np.greater(x, 0) * x

    @staticmethod
    # the derivative of the Rectified Linear Unit activation function
    def activation_prime(x):
        return np.greater(x, 0)

    @staticmethod
    # The gradient of the Mean Squared Error (MSE) cost function w.r.t. the output
    def cost_function_grad(output, target):
        return 2 / len(target) * (output - target)

    # Returns a list of two-element lists containing the weights and biases for the network
    def create_connections(self):
        weights = [np.zeros((self.layer_sizes[i], self.layer_sizes[i - 1])) for i in range(1, len(self.layer_sizes))]
        biases = [np.zeros(size) for size in self.layer_sizes[1:]]
        return [[weight, bias] for weight, bias in zip(weights, biases)]
