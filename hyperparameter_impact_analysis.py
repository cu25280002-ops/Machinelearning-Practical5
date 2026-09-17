import pandas as pd
params_df = pd.DataFrame({
    'Hyperparameter': ['Alpha (Learning Rate)', 'Gamma (Discount Factor)', 'Epsilon (Exploration)'],
    'High Value Impact': ['Unstable learning, divergence', 'Focuses on long-term rewards', 'Excessive random actions'],
    'Low Value Impact': ['Slow learning convergence', 'Focuses on immediate rewards', 'Stuck in local optima']
})
print("--- Hyperparameter Impact Analysis ---")
print(params_df)