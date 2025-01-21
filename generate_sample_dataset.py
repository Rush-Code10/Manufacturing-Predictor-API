import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

temperature = np.random.normal(75, 15, n_samples)
run_time = np.random.normal(100, 25, n_samples) 

downtime_prob = (temperature > 85) & (run_time > 120)
downtime = np.random.binomial(1, 0.8, n_samples) * downtime_prob

df = pd.DataFrame({
    'Machine_ID': np.arange(1, n_samples + 1),
    'Temperature': temperature,
    'Run_Time': run_time,
    'Downtime_Flag': downtime.astype(int)
})

df.to_csv('data/sample_data.csv', index=False)