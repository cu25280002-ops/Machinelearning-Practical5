import numpy as np
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])
for episode in range(200):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        q_table[state, action] += 0.8 * (reward + 0.95 * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
print("Final Learned Q-Table:")
print(np.round(q_table, 2))
env.close()