import gym
import matplotlib.pyplot as plt
from Exercise4.NeuralNetwork import NeuralNetwork
from Exercise4.Agent import Agent

# Parameters
num_games = int(1e4)
learning_rate = 0.0001
discount_factor = 0.99
epsilon_max = 1
epsilon_min = 0.01
epsilon_num_steps = 2500
num_observations = 4
num_actions = 2
num_games_per_batch = 100
num_hidden_layer_neurons = 128

# Creates environment, neural network, and agent
env = gym.make('CartPole-v0')
network = NeuralNetwork([num_observations, num_hidden_layer_neurons, num_actions], learning_rate)
agent = Agent(network, env.reset(), num_actions, discount_factor, epsilon_max, epsilon_min, epsilon_num_steps)


# Simulates a neural Q-learning agent in the Cart-pole environment and plots results
def main():
    rewards = []
    win_percentages = []
    for i in range(num_games):
        rewards.append(run_game())
        # Uses the last batch of rewards to calculate the average
        if i % num_games_per_batch == 0:
            win_percentages.append(sum(rewards) / len(rewards))
            rewards = []
    env.close()
    # Plots the data
    plt.xlabel('Number of ' + str(num_games_per_batch) + '-Game Batches')
    plt.ylabel('Avg. Reward / Decreasing Epsilon')
    plt.plot([i for i in range(num_games // num_games_per_batch)], win_percentages)
    plt.show()


# Runs a single game in the environment and returns the final reward
def run_game():
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.action()
        observation, reward, done = env.step(action)[0:3]
        agent.update(action, reward, observation)
        total_reward += reward
    agent.update_epsilon()
    return total_reward


if __name__ == '__main__':
    main()
