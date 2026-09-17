import numpy as np
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1
gamma = 0.99
epsilon = 0.1
state, _ = env.reset()
action = env.action_space.sample()
next_state, reward, terminated, truncated, _ = env.step(action)
q_table[state, action] = q_table[state, action] + alpha * (
    reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
)
print("Q-table update step complete.")
env.close()