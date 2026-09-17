import torch
import torch.nn as nn
import gymnasium as gym
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x): return self.fc(x)
model = DQN()
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()
env = gym.make("CartPole-v1")
state, _ = env.reset()
state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    action = torch.argmax(model(state_t)).item()
print("Loaded Model Action Output:", action)
env.close()