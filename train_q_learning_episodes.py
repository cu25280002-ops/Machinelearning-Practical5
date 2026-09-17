import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.8
gamma = 0.95
epsilon = 0.1
episodes = 500

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

print("Training finished for 500 episodes.")
env.close()