import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

q_changes = []
for episode in range(200):
    state, _ = env.reset()
    done = False
    total_change = 0
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        old_val = q_table[state, action]
        new_val = old_val + 0.5 * (reward + 0.9 * np.max(q_table[next_state]) - old_val)
        q_table[state, action] = new_val
        
        total_change += abs(new_val - old_val)
        state = next_state
        
    q_changes.append(total_change)

plt.plot(q_changes)
plt.xlabel("Episode")
plt.ylabel("Q-Value Variation")
plt.title("Q-Table Convergence Analysis")
plt.show()
env.close()