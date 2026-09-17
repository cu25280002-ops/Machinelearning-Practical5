import random
from collections import deque
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
memory = ReplayBuffer(100)
memory.push(state=[0, 1], action=1, reward=1.0, next_state=[1, 1], done=False)
print("Replay Memory Size:", len(memory))