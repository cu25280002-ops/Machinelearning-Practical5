import numpy as np
import matplotlib.pyplot as plt
episodes = 50
q_learning_rewards = np.cumsum(np.random.choice([0, 1], size=episodes, p=[0.6, 0.4]))
dqn_rewards = np.cumsum(np.random.choice([0, 1], size=episodes, p=[0.3, 0.7]))
plt.plot(q_learning_rewards, label="Q-Learning")
plt.plot(dqn_rewards, label="DQN")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.title("Reward Performance Comparison")
plt.legend()
plt.show()