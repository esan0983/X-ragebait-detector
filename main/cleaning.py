import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df['text'] = df['text'].str.strip()
    
    df = df.drop_duplicates(subset=['text'])
    
    df = df[['text']]
    
    return df

if __name__ == "__main__":
    ragebait_candidates = pd.read_parquet("data/raw/ragebait_candidates.parquet", engine="pyarrow") # CHANGE THIS DEPENDING ON SITUTATION

    jev_df = clean(ragebait_candidates)
    jev_df.to_parquet("data/jev/jev_input_df.parquet")