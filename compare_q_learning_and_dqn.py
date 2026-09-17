import pandas as pd
comparison_df = pd.DataFrame({
    'Feature': ['State Space', 'Action Space', 'Function Approximator', 'Memory Usage'],
    'Q-Learning': ['Discrete (Small)', 'Discrete', 'Q-Table Matrix', 'Low'],
    'DQN': ['Continuous / High-Dim', 'Discrete', 'Deep Neural Network', 'High (Replay Memory)']
})
print("--- Q-Learning vs DQN Comparison ---")
print(comparison_df)