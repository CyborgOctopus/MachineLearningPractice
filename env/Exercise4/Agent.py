import numpy as np
import torch


# Defines an agent which can act in environments, controlled by a neural network
class Agent:

    def __init__(self, network, num_actions, discount_factor, epsilon_max, epsilon_min,
                 epsilon_dec):
        self.network = network
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_dec = epsilon_dec

    # Trains the agent using reinforcement learning
    def update(self, state, action, reward, new_state):
        self.network.optimizer.zero_grad()
        state = torch.tensor(state, dtype=torch.float).to(self.network.device)
        action = torch.tensor(action).to(self.network.device)
        reward = torch.tensor(reward).to(self.network.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.network.device)
        current_q_value = self.network.feedforward(state)[action]
        best_q_in_new_state = self.network.feedforward(new_state).max()
        target_q_value = reward + self.discount_factor * best_q_in_new_state
        cost = self.network.loss(target_q_value, current_q_value).to(self.network.device)
        cost.backward()
        self.network.optimizer.step()
        self.update_epsilon()

    # Chooses an action according to the epsilon greedy policy
    def action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.best_action(state)

    # Finds the action which leads to the greatest reward in a given state, according to the neural network
    def best_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.network.device)
        action_values = self.network.feedforward(state)
        return torch.argmax(action_values).item()

    # Linearly decreases the value of epsilon
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min
