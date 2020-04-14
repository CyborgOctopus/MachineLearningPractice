import numpy as np


# A basic neural network (what a shock!)
# The 'layer_sizes' parameter is a list of numbers which give the number of neurons contained in each layer.
# The length of this list defines the total number of layers (including the input and output layers).
# The 'activation_function' parameter defines which activation function is used for the network. The two possible
# choices are 'sigmoid' and 'relu.'
# Guidance from http://neuralnetworksanddeeplearning.com/
class NeuralNetwork:

    def __init__(self, layer_sizes, activation_function='sigmoid'):
        self.activations = {'sigmoid': (self.sigmoid, self.sigmoid_prime), 'relu': (self.relu, self.relu_prime)}
        self.layer_sizes = layer_sizes
        self.weights = self.create_weights()
        self.biases = self.create_biases()
        self.activation_function, self.activation_function_prime = self.activations[activation_function]

    # Runs the network forward to produce an output
    def run(self, input_data):
        output = input_data
        for weight_matrix, bias in zip(self.weights, self.biases):
            output = self.activation_function(np.dot(weight_matrix, output) + bias)
        return output

    # Trains the network on a minibatch of examples. The 'inputs' and 'targets' are lists of training examples and
    # target outputs respectively.
    def train_on_minibatch(self, inputs, targets, learning_rate):
        weight_gradients, bias_gradients = self.backprop(inputs, targets)
        self.weights -= learning_rate * weight_gradients
        for i, bias_gradient_matrix in zip(range(len(self.biases)), bias_gradients):
            for bias_gradient in bias_gradient_matrix.T:
                self.biases[i] -= learning_rate * bias_gradient

    # Returns the weight and bias gradients
    def backprop(self, inputs, targets):
        outputs = [np.array(inputs).T]
        intermediate_outputs = []
        for weight_matrix, bias in zip(self.weights, self.biases):
            bias_matrix = np.array([bias] * len(inputs)).T
            intermediate_outputs.append(np.dot(weight_matrix, outputs[-1]) + bias_matrix)
            outputs.append(self.activation_function(intermediate_outputs[-1]))
        # Derivative of the mean squared error w.r.t. the output: ∂E/∂o = 1/n(o - t)
        error_deriv_output = (outputs[-1] - np.array(targets).T) / len(inputs)
        output_deriv_int_output = self.activation_function_prime(intermediate_outputs[-1])
        error_deriv_int_output = error_deriv_output * output_deriv_int_output
        weight_gradients = [np.dot(error_deriv_int_output, outputs[-2].T)]
        bias_gradients = [error_deriv_int_output]
        for i in range(2, len(outputs)):
            # Derivative of the error w.r.t. the output of the ith layer:
            # ∂E/∂o^i = ∂E/∂(o_int)^(i + 1) * ∂(o_int)^(i + 1)/∂o^i
            error_deriv_output = np.dot(self.weights[-i + 1].T, error_deriv_int_output)
            # ∂o^i/∂(o_int)^i
            output_deriv_int_output = self.activation_function_prime(intermediate_outputs[-i])
            # ∂E/∂(o_int)^i = ∂E/∂o^i * ∂o^i/∂(o_int)^i
            error_deriv_int_output = error_deriv_output * output_deriv_int_output
            # Derivative of the error w.r.t. the weights of the ith layer: ∂E/∂w^i = ∂E/∂(o_int)^i * ∂(o_int)^i/∂w^i
            weight_gradients.append(np.dot(error_deriv_int_output, outputs[-i - 1].T))
            # Deriv. of error w.r.t. biases of ith layer: ∂E/∂b^i = ∂E/∂(o_int)^i * ∂(o_int)^i/∂b^i = ∂E/(∂o_int)^i
            bias_gradients.append(error_deriv_int_output)

        return np.array(weight_gradients[::-1]), np.array(bias_gradients[::-1])

    # Creates a numpy array containing weights for each layer of the neural network
    def create_weights(self):
        weights = []
        for i in range(1, len(self.layer_sizes)):
            layer_weights = []
            for neuron in range(self.layer_sizes[i]):
                neuron_weights = [np.random.normal(0) for j in range(self.layer_sizes[i - 1])]
                layer_weights.append(neuron_weights)
            weights.append(np.array(layer_weights))
        return np.array(weights)

    # Creates a numpy array containing biases for each layer of the network
    def create_biases(self):
        biases = []
        for layer in self.layer_sizes[1:]:
            bias = [np.random.normal(0) for i in range(layer)]
            biases.append(bias)
        return np.array(biases)

    @staticmethod
    # Sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    # The derivative of the sigmoid function
    def sigmoid_prime(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    @staticmethod
    # Rectified Linear Unit (ReLU) function
    def relu(x):
        return np.greater(x, 0) * x

    @staticmethod
    # The derivative of the Rectified Linear Unit (ReLU) function
    def relu_prime(x):
        return np.greater_equal(x, 0)
