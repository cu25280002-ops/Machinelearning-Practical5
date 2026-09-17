import numpy as np
import gymnasium as gym
frozen_env = gym.make("FrozenLake-v1", is_slippery=True)
print("FrozenLake State Space:", frozen_env.observation_space.n)
print("FrozenLake Action Space:", frozen_env.action_space.n)
print("\nCustom Grid World:")
print("Deterministic movement, -0.1 reward per step to encourage shortest path.")
frozen_env.close()