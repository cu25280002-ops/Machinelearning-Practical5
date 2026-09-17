import gymnasium as gym
env=gym.make("CartPole-v1", render_mode=None)
state, info= env.reset()
for step in range(10):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: Action={action}, Reward={reward}")
    if terminated or truncated:
        break
env.close()