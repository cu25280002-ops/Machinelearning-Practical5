import numpy as np
import matplotlib.pyplot as plt
episodes = 100
raw_rewards = np.random.randint(10, 100, size=episodes) + np.linspace(0, 100, episodes)
smoothed_rewards = np.convolve(raw_rewards, np.ones(10)/10, mode='valid')
plt.plot(raw_rewards, alpha=0.3, label="Raw Rewards")
plt.plot(smoothed_rewards, label="Moving Average (Window=10)", color="red")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("RL Agent Learning Curve")
plt.legend()
plt.show()