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
epsilon_dec = 1e-5
num_observations = 4
num_actions = 2
num_games_to_avg = 100
num_hidden_layer_neurons = 128

# Creates environment, neural network, and agent
env = gym.make('CartPole-v1')
network = NeuralNetwork([num_observations, num_hidden_layer_neurons, num_actions], learning_rate)
agent = Agent(network, num_actions, discount_factor, epsilon_max, epsilon_min, epsilon_dec)


# Simulates a neural Q-learning agent in the Cart-pole environment and plots results
def main():
    rewards = []
    avg_rewards = []
    epsilons = []
    for i in range(num_games):
        if len(rewards) == num_games_to_avg:
            rewards = rewards[1:]
        rewards.append(run_game())
        avg_rewards.append(sum(rewards) / len(rewards))
        epsilons.append(agent.epsilon)
    env.close()
    plot(avg_rewards, epsilons)


# Plots the reward and decreasing epsilon in one graph
def plot(avg_rewards, epsilons):
    color = 'tab:blue'
    fig, reward_axis = plt.subplots()
    reward_axis.set_xlabel('Number of Games')
    reward_axis.set_ylabel('Avg. Reward', color=color)
    reward_axis.plot(range(num_games), avg_rewards, color=color)

    color = 'tab:orange'
    epsilon_axis = reward_axis.twinx()
    epsilon_axis.set_ylabel('Epsilon', color=color)
    epsilon_axis.plot(range(num_games), epsilons, color=color)

    fig.tight_layout()
    plt.show()


# Runs a single game in the environment and returns the final reward
def run_game():
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.action(state)
        new_state, reward, done = env.step(action)[0:3]
        agent.update(state, action, reward, new_state)
        total_reward += reward
        state = new_state
    return total_reward


if __name__ == '__main__':
    main()
