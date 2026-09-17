import gymnasium as gym
env = gym.make("FrozenLake-v1")
print("Action Space:", env.action_space)
print("Action Space Size (N):", env.action_space.n)
print("Observation Space:", env.observation_space)
print("Observation Space Size (N):", env.observation_space.n)
env.close()