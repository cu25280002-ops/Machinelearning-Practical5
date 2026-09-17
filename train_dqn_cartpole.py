import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
env = gym.make("CartPole-v1")
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x):
        return self.fc(x)
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
for episode in range(50):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    while not done:
        if random.random() < 0.1:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(model(state)).item()            
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        target = reward + (0.99 * torch.max(model(next_state_t)).item() if not done else 0)
        current = model(state)[0, action]
        loss = criterion(current, torch.tensor(target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state_t
print("DQN Simple Training Done.")
env.close()