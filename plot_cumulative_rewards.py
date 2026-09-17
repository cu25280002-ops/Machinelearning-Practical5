import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
rewards = []
for episode in range(300):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        q_table[state, action] += 0.8 * (reward + 0.95 * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        total_reward += reward
    rewards.append(total_reward)
plt.plot(np.cumsum(rewards))
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Rewards Curve")
plt.show()
env.close()