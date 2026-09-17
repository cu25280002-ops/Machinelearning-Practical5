import torch
import torch.nn as nn
import gymnasium as gym
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.fc(x)
model = DQN()
env = gym.make("CartPole-v1")
eval_rewards = []
for episode in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(model(state_t)).item()
        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        total_reward += reward
    eval_rewards.append(total_reward)
print("Evaluation Rewards across 10 Episodes:", eval_rewards)
env.close()