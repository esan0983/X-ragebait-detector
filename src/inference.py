"""
src/inference.py
=====================================================
Uses the initial trained RoBERTa model to infer and filter the second batch
"""

import pandas as pd

from src.initial_regression import load_regressor_for_inference

def main(df: pd.DataFrame) -> None:
    predict = load_regressor_for_inference("data/ml_data/initial_classifier")
    texts = df['text'].tolist()

    scores = predict(texts)
    df['ragebait'] = scores

    df = df[df['ragebait'] > 0.5]

    df.to_parquet("data/jev/jev_input_df_2.parquet")
    print("Parquet saved!")

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/inference_df.parquet")
    main(df)