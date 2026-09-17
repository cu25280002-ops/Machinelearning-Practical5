import gymnasium as gym
print("Gymnasium Version",gym.__version__)
env=gym.make("CartPole-v1")
print("Environment created successfully:", env.spec.id)
env.close()