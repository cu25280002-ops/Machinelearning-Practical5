import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1")
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.fc(x)
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
rewards = []
for episode in range(30):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_r = 0
    while not done:
        action = env.action_space.sample() if random.random() < 0.1 else torch.argmax(model(state)).item()
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_r += reward
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    rewards.append(total_r)
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Reward per Episode")
plt.show()
env.close()