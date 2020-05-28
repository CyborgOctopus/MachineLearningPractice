import numpy as np


# An agent which can perform Q-learning by updating the values in its Q-table. It uses an epsilon-greedy policy with
# decreasing epsilon.
class QTableLearner:

    def __init__(self, initial_state, num_states, num_actions, learning_rate, discount_factor, epsilon_max, epsilon_min,
                 epsilon_num_steps):
        self.state = initial_state
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = np.zeros((num_states, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_max
        self.epsilon_num_steps = epsilon_num_steps

    # Updates the values of the Q-table, allowing the agent to learn
    def update(self, action, reward, new_state):
        current_q = self.q_table[self.state][action]
        best_action = self.best_action(new_state)
        best_q = self.q_table[new_state][best_action]
        delta_q = self.learning_rate * (reward + self.discount_factor * best_q - current_q)
        self.q_table[self.state][action] += delta_q
        self.state = new_state
        self.update_epsilon()

    # Chooses an action according to the epsilon greedy policy
    def action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.best_action(self.state)

    # Finds the action which leads to the greatest reward in a given state, according to the Q-table.
    def best_action(self, state):
        best_action = 0
        for i in range(self.num_actions):
            if self.q_table[state][i] > self.q_table[state][best_action]:
                best_action = i
        return best_action

    # Linearly decreases the value of epsilon
    def update_epsilon(self):
        self.epsilon += (self.epsilon_min - self.epsilon_max) / self.epsilon_num_steps

