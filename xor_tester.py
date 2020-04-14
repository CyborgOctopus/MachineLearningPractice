import random
from NeuralNetwork import NeuralNetwork

# parameters
num_training_epochs = 5000
num_tests = 5000 
learning_rate = 1
network = NeuralNetwork([2, 3, 1])


# trains a neural network to mimic the XOR operation
def main():
    print(evaluate_network())
    for i in range(num_training_epochs):
        a, b = random.randint(0, 1), random.randint(0, 1)
        inputs = [[a, b]]
        targets = [(a != b)]
        network.train_on_minibatch(inputs, targets, learning_rate)
    print(evaluate_network())
    print(network.run([1, 1]))


# Returns the fraction of examples that the network got correct
def evaluate_network():
    num_correct = 0
    for i in range(num_tests):
        a, b = random.randint(0, 1), random.randint(0, 1)
        network_output = network.run([a, b])[0]
        if bool(network_output > 0.5) == (a != b):
            num_correct += 1
    return num_correct / num_tests


if __name__ == '__main__':
    main()
