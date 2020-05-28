import gym
import matplotlib.pyplot as plt
from Exercise3.QTableLearner import QTableLearner

# Parameters
num_games = int(5e5)
initial_state = 0
num_states = 16
num_actions = 4
learning_rate = 0.0005
discount_factor = 0.9
epsilon_max = 1
epsilon_min = 0.01
epsilon_num_steps = 1e6
num_games_per_batch = 100

# Creates environment and agent
env = gym.make('FrozenLake-v0')
agent = QTableLearner(initial_state, num_states, num_actions, learning_rate, discount_factor, epsilon_max, epsilon_min,
                      epsilon_num_steps)


# Simulates a Q-table agent in the Frozen Lake environment and plots results
def main():
    rewards = []
    win_percentages = []
    for i in range(num_games):
        rewards.append(run_game())
        # Uses the last ten or rewards to calculate the win percentage
        if i % num_games_per_batch == 0:
            win_percentages.append(sum(rewards) / len(rewards) * 100)
            rewards = []
    env.close()
    # Plots the data
    plt.xlabel('Number of ' + str(num_games_per_batch) + '-Game Batches')
    plt.ylabel('Win Percentage')
    plt.plot([i for i in range(num_games // num_games_per_batch)], win_percentages)
    plt.show()


# Runs a single game in the environment and returns the final reward
def run_game():
    observation = 0
    reward = 0
    env.reset()
    done = False
    while not done:
        action = agent.action()
        observation, reward, done = env.step(action)[0:3]
        agent.update(action, reward, observation)
    return reward


if __name__ == '__main__':
    main()