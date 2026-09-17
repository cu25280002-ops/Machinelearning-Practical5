import torch
import torch.nn as nn
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x): return self.fc(x)
model = DQN()
torch.save(model.state_dict(), "dqn_cartpole.pth")
print("DQN Model saved successfully as 'dqn_cartpole.pth'.")