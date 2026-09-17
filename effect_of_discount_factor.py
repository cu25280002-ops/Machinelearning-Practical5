import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1", is_slippery=False)
gammas = [0.1, 0.5, 0.99]
for gamma in gammas:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    for _ in range(200):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            q_table[state, action] += 0.5 * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    plt.plot(np.cumsum(rewards), label=f"gamma={gamma}")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Discount Factor Impact")
plt.show()
env.close()