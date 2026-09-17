import torch
import torch.nn as nn
class SimpleDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x): return self.fc(x)
eval_net = SimpleDQN()
target_net = SimpleDQN()
target_net.load_state_dict(eval_net.state_dict())
print("Target Network weights synchronized with Evaluation Network.")