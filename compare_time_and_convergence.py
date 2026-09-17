import time
import numpy as np
start = time.time()
q_data = np.zeros((16, 4))
for _ in range(1000):
    q_data += 0.01
q_time = time.time() - start
start = time.time()
dqn_sim = np.random.randn(1000, 64)
dqn_time = time.time() - start
print(f"Q-Learning Simulated Execution Time: {q_time:.5f} sec")
print(f"DQN Simulated Execution Time:        {dqn_time:.5f} sec")