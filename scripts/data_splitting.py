"""
src/utils/data_splitting.py
=======================================================
Splits the tweets into pre-initial classifier tweets and inference tweets (post-cleaning, so index 49812 and up) (I know, this hardcoded splitting kinda sucks)
"""

import pandas as pd

def main(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:49812], df.iloc[49812:]

if __name__ == "__main__":
    df = pd.read_parquet("data/raw/ragebait_candidates.parquet")
    jev_input_df, jev_input_df_2 = main(df)
    jev_input_df.to_parquet("data/jev/jev_input_df.parquet")
    jev_input_df_2.to_parquet("data/processed/inference_df.parquet")