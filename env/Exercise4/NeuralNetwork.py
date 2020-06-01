import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


# Defines a fully-connected neural network with a controllable number of layers and neurons per layer. Assumes the last
# layer is unactivated
class NeuralNetwork(nn.Module):

    def __init__(self, layer_sizes, learning_rate):
        super(NeuralNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        self.connections = self.create_connections()
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.parameters(), self.learning_rate)
        self.loss = nn.MSELoss()
        self.device = torch.device('cpu')
        self.to(self.device)

    # Runs inputs through the network to produce an output
    def feedforward(self, input_data):
        output = input_data
        for connection in self.connections[:-1]:
            output = func.relu(connection(output))
        return self.connections[-1](output)

    # Returns a list of the weights and biases between each layer of the network
    def create_connections(self):
        num_layers = len(self.layer_sizes)
        return nn.ModuleList([nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]) for i in range(1, num_layers)])
