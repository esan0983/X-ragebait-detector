from typesafe_sdk import TypeSafeClient
import pandas as pd

from utils import TYPESAFE_API_KEY

QUESTIONS = {
  "ragebait": {
    "type": "noul",
    "instructions": "Classify this post as ragebait only if its primary purpose appears to be provoking anger or outrage rather than informing, expressing a personal experience, or stating an opinion. Do not classify a post as ragebait merely because its subject matter is disturbing, controversial, political, or emotionally charged.",
    "criteria": {
      "true": "The post uses inflammatory editorializing, mocking or insulting language, slurs, deliberately exaggerated framing, unsupported sweeping claims, or other rhetorical techniques whose apparent purpose is to provoke anger or outrage. The inflammatory framing must be directed toward provoking a reaction, not merely describing an event",
      "false": "The post reports an event, summarizes a claim or news story, describes a personal experience, or expresses an opinion—even when the subject is controversial, offensive, disturbing, or emotionally charged—provided it does not add clear inflammatory editorializing, unsupported generalizations, or deliberate outrage-baiting framing."
    }
  },
  "controversial": {
    "type": "noul",
    "instructions": "Does this post address a polarizing, sensitive, or high-stakes topic in a way that naturally invites split opinions or intense debate, regardless of whether the author is intentionally provoking anger?",
    "criteria": {
      "true": "The content taps into a debate-heavy, polarizing, or sensitive topic likely to divide readers and generate high engagement through opposing viewpoints, including both earnest debate and intentional ragebait.",
      "false": "The post expresses a widely accepted view, a neutral factual observation, or a non-contentious personal statement that is unlikely to trigger significant disagreement or divided commentary."
    }
  }
}

client = TypeSafeClient(
    api_key=TYPESAFE_API_KEY,
    timeout=120.0
)

def jev(text: str) -> tuple[float, float, int, int]:
    response = client.system_one(
        state=text,
        questions=QUESTIONS,
        model="jev-latest"
    )

    answers = response.answers
    ragebait_score = answers["ragebait"].noul
    controversial_score = answers["controversial"].noul
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    return ragebait_score, controversial_score, input_tokens, output_tokens

def main(df: pd.DataFrame) -> None:
    total_input = 0
    total_output = 0
    counter = 0
    for text in df['text']:
        counter += 1
        ragebait_score, controversial_score, input_tokens, output_tokens = jev(text)
        df.loc[df['text'] == text, 'ragebait'] = ragebait_score
        df.loc[df['text'] == text, 'controversial'] = controversial_score
        total_input += input_tokens
        total_output += output_tokens
        if counter % 10 == 0:
            print(f"Number of requests: {counter}")
    
    df.to_parquet("data/jev/jev_output_df.parquet")
    print("Parquet saved!")
    print(f"Number of rows: {len(df)}")
    print(f"Input tokens spent: {total_input}")
    print(f"Output tokens spent: {total_output}")
    cost = total_input / 1e9 * 42
    print(f"Cost (note that output tokens are too cheap to measure): ${cost:.6f}")

if __name__ == "__main__":
    df = pd.read_parquet("data/jev/jev_input_df.parquet")
    main(df)