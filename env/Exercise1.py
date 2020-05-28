import gym
import matplotlib.pyplot as plt

# Parameters
num_games = 1000

# Creates environment
env = gym.make('FrozenLake-v0')


# Simulates a random agent in the Frozen Lake environment and plots results
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
    reward = 0
    env.reset()
    done = False
    while not done:
        env.render()
        reward, done = env.step(env.action_space.sample())[1:3]
    return reward


if __name__ == '__main__':
    main()
