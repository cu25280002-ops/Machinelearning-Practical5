import torch
import torch.nn as nn
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.network(x)
model = DQN(state_dim=4, action_dim=2)
dummy_input = torch.randn(1, 4)
output = model(dummy_input)
print("DQN Architecture Created:")
print(model)
print("Sample Output (Q-Values):", output)