import gym
import matplotlib.pyplot as plt

# Parameters
num_games = 1000
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Creates environment
env = gym.make('FrozenLake-v0')


# Simulates a deterministic agent in the Frozen Lake environment and plots results
def main():
    rewards = []
    win_percentages = []
    for i in range(num_games):
        rewards.append(run_game())
        # Uses the last ten or rewards to calculate the win percentage
        if i % 10 == 0:
            win_percentages.append(sum(rewards) / len(rewards) * 100)
            rewards = []
    env.close()
    # Plots the data
    plt.xlabel('Number of Ten-Game Batches')
    plt.ylabel('Win Percentage')
    plt.plot([i for i in range(num_games // 10)], win_percentages)
    plt.show()


# Runs a single game in the environment and returns the final reward
def run_game():
    observation = 0
    reward = 0
    env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done = env.step(policy[observation])[0:3]
    return reward


# Deterministic policy for the Frozen Lake environment, with observation-action pairs. Observations corresponding to
# a hole or escape are excluded since no more actions can be taken at that point
policy = {0: RIGHT, 1: RIGHT, 2: DOWN, 3: LEFT, 4: DOWN, 6: DOWN, 8: RIGHT, 9: RIGHT, 10: DOWN, 13: RIGHT, 14: RIGHT}

if __name__ == '__main__':
    main()
