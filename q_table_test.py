# guidance from https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
# helpful in learning what to use for the discount factor
import numpy as np

# parameters
num_training_epochs = 1000
num_testing_epochs = 100
num_steps_per_epoch = 100
discount_factor = 0.95
learning_rate = 0.1
greedy_choice_prob = 0.9

# setup
env = [[0, 1, 1], [0, -1, -1], [-5, 0, 5]]
q_table = np.zeros((9, 5))  # states of the environment are represented by the numbers 0-8 and actions by 0-4
action_table = [-1, 1, 0, -3, 3]


# trains an agent to gain the greatest reward in a simple environment using Q-learning
def main():
    print(evaluate_agent())
    for i in range(num_training_epochs):
        state = 0
        for j in range(num_steps_per_epoch):
            action = np.random.choice(range(len(action_table)))
            if np.random.uniform(0, 1) < greedy_choice_prob:
                action = find_best_action(state)
            new_state, reward = step(state, action)
            current_q = q_table[state][action]
            max_future_q = q_table[new_state][find_best_action(new_state)]
            q_table[state][action] += learning_rate * (reward + discount_factor * max_future_q - current_q)
            state = new_state
    print(evaluate_agent())


# calculates the average reward gained by the agent during repeated run-throughs of the environment
def evaluate_agent():
    total_reward = 0
    for i in range(num_testing_epochs):
        state = 0
        for j in range(num_steps_per_epoch):
            action = np.random.choice(range(len(action_table)))
            if np.random.uniform(0, 1) < greedy_choice_prob:
                action = find_best_action(state)
            state, reward = step(state, action)
            total_reward += reward
    return total_reward / num_testing_epochs


# finds the action estimated to give the greatest reward in a given state of the environment
def find_best_action(state):
    best_action = 0
    for i in range(len(action_table)):
        if q_table[state][i] > q_table[state][best_action]:
            best_action = i
    return best_action


# gets the reward and new state based on the agent's current state and action in the environment
def step(state, action):
    action_table_val = action_table[action]
    new_state = state + action_table_val
    if (state % 3 == 0 and action_table_val == -1) or (state % 3 == 2 and action_table_val == 1):
        new_state = state
    elif (state // 3 == 0 and action_table_val == -3) or (state // 3 == 2 and action_table_val == 3):
        new_state = state
    reward = env[new_state // 3][new_state % 3]
    return new_state, reward


if __name__ == '__main__':
    main()
