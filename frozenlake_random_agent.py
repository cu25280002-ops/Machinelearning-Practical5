import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
state, info = env.reset()
total_reward = 0
for step in range(20):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print("Random Simulation Finished. Total Reward:", total_reward)
env.close()