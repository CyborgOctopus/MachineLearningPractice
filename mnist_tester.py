import random
import numpy as np
from mnist import MNIST
from NeuralNetwork import NeuralNetwork

# parameters
learning_rate = 3
num_training_examples_per_epoch = 100
num_epochs = 100000
num_reports = 20
network = NeuralNetwork([784, 80, 20, 10])

# setup
mnist_data = MNIST('mnist_data')
mnist_data.gz = True
training_images, training_labels = mnist_data.load_training()
testing_images, testing_labels = mnist_data.load_testing()
training_images = np.array(training_images) / 255  # normalization to prevent calculation overflow
testing_images = np.array(testing_images) / 255


# trains a neural network to identify handwritten digits
def main():
    print(evaluate_network())
    for i in range(num_epochs):
        images, labels = get_random_training_examples()
        network.train_on_minibatch(images, [one_hot(label) for label in labels], learning_rate)
        if i % (num_epochs // num_reports) == 0:
            print(evaluate_network())


# pulls a set of randomly chosen training examples from a list
def get_random_training_examples():
    indices = np.random.randint(0, len(training_images) - 1, num_training_examples_per_epoch)
    return [training_images[i] for i in indices], [training_labels[i] for i in indices]


# returns the fraction of digits from the testing set which the network classifies correctly
def evaluate_network():
    num_correct = 0
    for image, label in zip(testing_images, testing_labels):
        if [float(i > 0.5) for i in network.run(image)] == one_hot(label):
            num_correct += 1
    return num_correct / len(testing_labels)


# gives a one-hot representation of a digit (a 10-dimensional vector with a 1 in the place corresponding to the digit
# and zeros everywhere else)
def one_hot(x):
    return [float(i == x) for i in range(10)]


if __name__ == '__main__':
    main()
