import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
action = 1
next_state, reward, terminated, truncated, info = env.step(action)
print("Initial State:", state)
print("Action Taken:", action)
print("Next State:", next_state)
print("Reward Received:", reward)
print("Is Terminated:", terminated)
print("Is Truncated:", truncated)
env.close()