import numpy as np
class SimpleGridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = 0
        self.goal = size * size - 1
    def reset(self):
        self.state = 0
        return self.state
    def step(self, action):
        row, col = divmod(self.state, self.size)
        if action == 0 and row > 0: row -= 1
        elif action == 1 and row < self.size - 1: row += 1
        elif action == 2 and col > 0: col -= 1
        elif action == 3 and col < self.size - 1: col += 1
        self.state = row * self.size + col
        done = (self.state == self.goal)
        reward = 1.0 if done else -0.1
        return self.state, reward, done
env = SimpleGridWorld()
print("Initial State:", env.reset())
next_state, reward, done = env.step(1)
print(f"Step Result -> Next State: {next_state}, Reward: {reward}, Done: {done}")