import numpy as np
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print("Initial Q-table:\n", q_table)
state, _ = env.reset()
action = 2
next_state, reward, _, _, _ = env.step(action)
q_table[state, action] += 0.1 * (reward + 0.9 * np.max(q_table[next_state]) - q_table[state, action])
print("\nUpdated Q-table:\n", q_table)
env.close()