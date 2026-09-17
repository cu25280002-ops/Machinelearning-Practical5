import numpy as np
def choose_action(state, q_table, epsilon, action_space_size):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(action_space_size)
    else:
        return np.argmax(q_table[state])
q_table_sample = np.array([[0.1, 0.8, 0.3, 0.2]])
action = choose_action(0, q_table_sample, epsilon=0.2, action_space_size=4)
print("Chosen Action via Epsilon-Greedy Policy:", action)