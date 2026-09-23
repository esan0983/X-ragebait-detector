"""
src/inference.py
=====================================================
Uses the initial trained RoBERTa model to infer and filter the second batch.
Note that on my end, I used a Google Colab Notebook.
"""

import pandas as pd
import numpy as np
import torch

from src.initial_regression import load_regressor_for_inference

def main(df: pd.DataFrame) -> None:
    def predict_batched(predict, texts, batch_size=256):
        out = []
        with torch.inference_mode():
            counter = 0
            for i in range(0, len(texts), batch_size):
                out.append(np.asarray(predict(texts[i:i+batch_size])).reshape(-1))
                if i % (batch_size * 50) == 0:
                    torch.cuda.empty_cache()
                counter += batch_size
                print(f"Progress: {counter}/{len(texts)}")
        return np.concatenate(out)

    predict = load_regressor_for_inference("data/ml_data")
    scores = predict_batched(predict, df['text'].tolist(), batch_size=256)
    df['inferred_ragebait'] = scores

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/inference_df.parquet")
    main(df)