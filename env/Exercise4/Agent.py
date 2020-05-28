import numpy as np


# Defines an agent which can act in environments, controlled by a neural network
class Agent:

    def __init__(self, network, initial_state, num_actions, discount_factor, epsilon_max, epsilon_min,
                 epsilon_num_steps):
        self.network = network
        self.state = initial_state
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_num_steps = epsilon_num_steps

    # Trains the agent using reinforcement learning
    def update(self, action, reward, new_state):
        current_q_values = self.network.feedforward(self.state)
        target_q_values = current_q_values
        best_q_in_new_state = max(self.network.feedforward(new_state))
        target_q_values[action] = reward + self.discount_factor * best_q_in_new_state
        self.network.train([self.state], [target_q_values])

    # Chooses an action according to the epsilon greedy policy
    def action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.best_action(self.state)

    # Finds the action which leads to the greatest reward in a given state, according to the neural network
    def best_action(self, state):
        action_values = self.network.feedforward(state)
        return np.argmax(action_values)

    # Linearly decreases the value of epsilon
    def update_epsilon(self):
        self.epsilon += (self.epsilon_min - self.epsilon_max) / self.epsilon_num_steps
