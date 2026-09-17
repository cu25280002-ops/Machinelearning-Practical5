import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("FrozenLake-v1", is_slippery=False)
epsilons = [0.01, 0.2, 0.8]
for eps in epsilons:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    for _ in range(200):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = env.action_space.sample() if np.random.rand() < eps else np.argmax(q_table[state])
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            q_table[state, action] += 0.5 * (reward + 0.9 * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state
            ep_reward += reward
        rewards.append(ep_reward)
    plt.plot(np.cumsum(rewards), label=f"eps={eps}")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Epsilon Value Comparison")
plt.show()
env.close()