import pandas as pd
report = pd.DataFrame({
    'Metric / Property': ['Algorithm Type', 'Suitable Environment', 'Training Speed', 'Sample Efficiency'],
    'Q-Learning': ['Model-Free Value-Based', 'Discrete State/Action', 'Very Fast', 'Moderate'],
    'Deep Q-Network (DQN)': ['Deep Value-Based', 'Continuous State / Discrete Action', 'Slower (Requires GPU)', 'High (via Replay Memory)']
})
print("=====================================================")
print("  REINFORCE LEARNING LAB SHEET-05 COMPARATIVE REPORT ")
print("=====================================================")
print(report)