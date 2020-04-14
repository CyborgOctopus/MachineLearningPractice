import numpy as np
from RubiksCube import RubiksCube
from NeuralNetwork import NeuralNetwork

# parameters
num_training_epochs = 1000
num_testing_epochs = 100
num_steps_per_epoch = 3
discount_factor = 0.9
learning_rate = 0.1
greedy_choice_prob = 0.9
num_blocks = 26
cube_state_size = num_blocks ** 2
num_actions = 9
num_steps_to_scramble = 1
network = NeuralNetwork([cube_state_size, 30, num_actions])


# trains a neural network to solve a Rubik's cube
def main():
    print(evaluate_network())
    for i in range(num_training_epochs):
        cube = RubiksCube()
        cube = cube.scramble(num_steps_to_scramble)
        cube_state = cube.get_state().reshape(cube_state_size)
        transforms = cube.get_transforms()
        inputs = []
        targets = []
        for j in range(num_steps_per_epoch):
            action = np.random.choice(range(num_actions))
            if np.random.uniform(0, 1) < greedy_choice_prob:
                action = find_best_action(cube)
            inputs.append(cube_state)
            cube *= transforms[action]
            new_cube_state = cube.get_state().reshape(cube_state_size)
            q_values = network.run(cube_state)
            new_q_values = network.run(new_cube_state)
            target = np.concatenate((q_values[0:action], [get_reward(cube) + discount_factor * max(new_q_values)],
                                    q_values[action + 1:]))
            targets.append(target)
            cube_state = new_cube_state
        network.train_on_minibatch(inputs, targets, learning_rate)
    print(evaluate_network())


# evaluates the performance of the neural network by measuring the percentage of Rubik's cubes it solves.
def evaluate_network():
    num_solved = 0
    identity = RubiksCube()
    for i in range(num_testing_epochs):
        cube = RubiksCube()
        cube = cube.scramble(num_steps_to_scramble)
        transforms = cube.get_transforms()
        for j in range(num_steps_per_epoch):
            action = np.random.choice(range(num_actions))
            if np.random.uniform(0, 1) < greedy_choice_prob:
                action = find_best_action(cube)
            cube *= transforms[action]
        if cube == identity:
            num_solved += 1
    return num_solved / num_testing_epochs


# computes the reward for the given Rubik's cube state
def get_reward(cube):
    identity_cube = RubiksCube()
    if cube == identity_cube:
        return 1
    return 0


# finds the best action for the current Rubik's cube state
def find_best_action(cube):
    best_action = 0
    state = cube.get_state().reshape(cube_state_size)
    q_values = network.run(state)
    for i in range(len(q_values)):
        if q_values[i] > q_values[best_action]:
            best_action = i
    return best_action


if __name__ == '__main__':
    main()
