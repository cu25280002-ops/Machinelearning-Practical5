import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

for _ in range(500):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else env.action_space.sample()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        q_table[state, action] += 0.8 * (reward + 0.95 * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state

successes = 0
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        if reward == 1:
            successes += 1

print(f"Evaluation Success Rate over 100 trials: {successes}%")
env.close()