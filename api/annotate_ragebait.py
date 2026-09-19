from typesafe_sdk import TypeSafeClient
import pandas as pd
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY")

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
  },
  "time_sensitivity": {
    "type": "score",
    "instructions": "How dependent is this post's meaning or interpretability on the time in which it was posted? Judge whether a reader encountering the post without its original temporal context could still understand what it means, not whether the topic itself will remain relevant or interesting.",
    "criteria": [
      "Timeless: The post's meaning is essentially independent of when it was posted. A reader years later could understand it without knowing the original date or current events.",
      "Low: The post contains minor references to current events, trends, or circumstances, but its meaning remains mostly understandable without knowing when it was posted.",
      "Moderate: Understanding the post reasonably well requires some awareness of the period in which it was written, such as a contemporary event, trend, ongoing situation, or cultural moment.",
      "High: The post depends heavily on its original temporal context. Without knowing what was happening around the time it was posted, its meaning, implication, or reference would be difficult to understand.",
      "Ephemeral: The post is strongly tied to a moment-specific event, live situation, breaking news, reaction, trend, or rapidly changing circumstance and may become substantially unintelligible without the original context."
    ]
  },
  "nicheness": {
    "type": "score",
    "instructions": "How much specialized knowledge, cultural familiarity, community membership, or domain-specific context is required to understand the post's meaning or intended reference? Judge the accessibility of the post itself, not whether the underlying topic is popular.",
    "criteria": [
      "Broadly accessible: Most people can understand the post's meaning without specialized knowledge or familiarity with a particular community, hobby, profession, or subculture.",
      "Somewhat specialized: Most people can understand the general meaning, but familiarity with a particular interest, cultural reference, or common online context would improve comprehension.",
      "Niche: The post is primarily understandable to people familiar with a particular community, hobby, fandom, profession, game, platform culture, or specialized topic.",
      "Highly niche: Understanding the intended meaning or reference requires substantial specialized knowledge or familiarity with a specific community, subculture, technical field, or ongoing inside discussion.",
      "Extremely niche: The post is effectively directed at a very small or specialized audience and would be difficult to interpret correctly without insider knowledge or highly specific context."
    ]
  },
  "emotional_intensity": {
    "type": "score",
    "instructions": "How emotionally charged is the language of the post, regardless of whether the author is intentionally trying to provoke the reader?",
    "criteria": [
      "Neutral: Little or no emotionally charged language.",
      "Mild: Contains limited emotional language, but the overall tone remains restrained.",
      "Moderate: Noticeably emotional, with clear expressions of enthusiasm, frustration, sadness, anger, disgust, excitement, or similar emotion.",
      "High: Strong emotional language, emphatic wording, insults, profanity, or dramatic expression is prominent throughout the post.",
      "Extreme: The post is overwhelmingly emotionally charged, highly aggressive, explosive, or intensely expressive."
    ]
  },
  "content_type": {
    "type": "choice",
    "instructions": "What is the primary communicative function of the post?",
    "criteria": {
      "factual_report": "Primarily reports or describes an event, observation, or information.",
      "claim": "Primarily makes a factual or causal claim that could in principle be evaluated as true or false.",
      "opinion": "Primarily expresses a subjective judgment, preference, belief, or evaluation.",
      "personal_experience": "Primarily describes the author's own experience, feelings, or circumstances.",
      "question": "Primarily asks a question or solicits information rather than making a substantive claim.",
      "humor_or_entertainment": "Primarily attempts to entertain, joke, parody, or amuse rather than communicate a serious claim or opinion.",
      "other": "Does not fit the other categories clearly."
    }
  },
  "context_dependence": {
    "type": "score",
    "instructions": "How much external context is required to correctly understand the meaning or intended implication of the post?",
    "criteria": [
      "Self-contained: The post can be understood accurately on its own.",
      "Minor context: Some additional context would improve understanding, but the main meaning is clear.",
      "Moderate context: External information is needed to fully understand the reference, implication, or situation being discussed.",
      "High context: The post is difficult to interpret correctly without substantial knowledge of an external event, conversation, media, or preceding content.",
      "Extreme context: The post is largely unintelligible or highly ambiguous without very specific external context."
    ]
  },
  "rhetorical_target": {
    "type": "choice",
    "instructions": "Who or what is the post primarily directing its criticism, praise, mockery, or other rhetoric toward?",
    "criteria": {
      "none": "The post does not have a clear rhetorical target.",
      "individual": "A specific identifiable person is the primary target.",
      "group": "A demographic, social, political, professional, cultural, or other group is the primary target.",
      "organization": "A company, institution, government, media outlet, team, community, or other organization is the primary target.",
      "product_or_work": "A product, service, piece of media, game, artwork, or other specific work is the primary target.",
      "abstract_issue": "An idea, policy, behavior, event, phenomenon, or abstract concept is the primary target.",
      "multiple": "The post targets multiple distinct entities without one clearly being primary."
    }
  },
  "generalization": {
    "type": "score",
    "instructions": "How broadly does the post generalize beyond the specific person, event, or situation being discussed?",
    "criteria": [
      "Specific: The statement is limited to a particular person, event, situation, or clearly defined example.",
      "Limited generalization: The post makes a modest generalization but retains meaningful qualifications or limitations.",
      "Broad: The post generalizes across a substantial category of people, events, or situations.",
      "Sweeping: The post makes a broad, categorical, or absolute claim about a large group or class with little qualification.",
      "Extreme: The post makes an exceptionally broad or absolute generalization that reduces a complex subject to a simplistic categorical claim."
    ]
  },
  "sarcasm_irony": {
    "type": "score",
    "instructions": "To what extent does the post rely on sarcasm, irony, parody, or deliberately non-literal language to communicate its intended meaning?",
    "criteria": [
      "None: The post is substantially literal and does not rely on sarcasm or irony.",
      "Possible: The wording contains mild or ambiguous ironic elements, but the intended meaning remains substantially literal.",
      "Present: Sarcasm or irony is clearly used as part of the post's communication.",
      "Strong: Understanding the intended meaning requires recognizing substantial sarcasm, irony, parody, or deliberate reversal of the literal wording.",
      "Essential: The post's intended meaning is primarily dependent on interpreting its language non-literally."
    ]
  }
}

client = TypeSafeClient(
    api_key=TYPESAFE_API_KEY,
    timeout=120.0
)

def jev(text: str) -> dict:
    response = client.system_one(
        state=text,
        questions=QUESTIONS,
        model="jev-latest"
    )

    answers = response.answers
    return_dict = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    return_dict["ragebait"] = answers["ragebait"].noul
    return_dict["controversial"] = answers["controversial"].noul
    return_dict["time_sensitivity"] = answers["time_sensitivity"].score
    return_dict["nicheness"] = answers["nicheness"].score
    return_dict["emotional_intensity"] = answers["emotional_intensity"].score
    return_dict["content_type"] = answers["content_type"].choice
    return_dict["context_dependence"] = answers["context_dependence"].score
    return_dict["rhetorical_target"] = answers["rhetorical_target"].choice
    return_dict["generalization"] = answers["generalization"].score
    return_dict["sarcasm_irony"] = answers["sarcasm_irony"].score

    return return_dict

def main(df: pd.DataFrame) -> None:
    total_input = 0
    total_output = 0
    
    results_list = []
    
    for text in tqdm(df['text'], desc="Processing requests"):
        metric_dict = jev(text)
        
        total_input += metric_dict.get("input_tokens", 0)
        total_output += metric_dict.get("output_tokens", 0)

        row_answers = {
            k: v for k, v in metric_dict.items() 
            if k not in ["input_tokens", "output_tokens"]
        }
        results_list.append(row_answers)
        
    results_df = pd.DataFrame(results_list, index=df.index)
    df = pd.concat([df, results_df], axis=1)
    
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