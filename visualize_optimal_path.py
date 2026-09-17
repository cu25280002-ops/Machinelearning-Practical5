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
q_table = np.zeros((16, 4))
for episode in range(300):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state]) if np.random.rand() > 0.1 else np.random.randint(4)
        next_state, reward, done = env.step(action)
        q_table[state, action] += 0.2 * (reward + 0.9 * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
path = []
state = env.reset()
path.append(state)
done = False
while not done and len(path) < 10:
    action = np.argmax(q_table[state])
    state, reward, done = env.step(action)
    path.append(state)
print("Learned Optimal Path through States:", path)