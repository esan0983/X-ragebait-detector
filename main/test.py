import pandas as pd

df = pd.read_parquet("data/raw/ragebait_candidates.parquet", engine="pyarrow")
sampled_df = df.sample(n=300, random_state=42)
sampled_df.to_parquet("data/raw/ragebait_candidates_test.parquet", engine="pyarrow")