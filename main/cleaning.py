import pandas as pd

def clean(df) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['tweet_id'])
    df = df['text']

    discarded_words = ['Clancy']
    regex_pattern = '|'.join(discarded_words)

    df = df[~df['text'].str.contains(regex_pattern, na=False)]

    return df

if __name__ == "__main__":
    ragebait_candidates = pd.read_parquet("data/raw/ragebait_candidates.parquet", engine="pyarrow")

    gpt_input = clean(ragebait_candidates)
    gpt_input.to_parquet("data/gpt_input/gpt_input_1.parquet")