import asyncio
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError


load_dotenv()

TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY")

CONCURRENCY = 16          # tune: start ~16, double until throughput plateaus
CHUNK_SIZE = 2000         # checkpoint granularity
CHUNK_DIR = Path("data/jev/chunks")
INPUT_PATH = "data/jev/jev_input_df.parquet"
OUTPUT_PATH = "data/jev/jev_output_df.parquet"

RETRY = RetryPolicy(
    max_retries=6,
    backoff_initial=1.0,
    backoff_max=30.0,
    timeout=None,
    # 429/5xx/timeouts/connection errors and Retry-After are handled by default
)


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


RESULT_COLS = [
    "ragebait", "controversial", "time_sensitivity", "nicheness",
    "emotional_intensity", "content_type", "context_dependence",
    "rhetorical_target", "generalization", "sarcasm_irony",
]


def parse(response) -> dict:
    a = response.answers
    return {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "ragebait": a["ragebait"].noul,
        "controversial": a["controversial"].noul,
        "time_sensitivity": a["time_sensitivity"].score,
        "nicheness": a["nicheness"].score,
        "emotional_intensity": a["emotional_intensity"].score,
        "content_type": a["content_type"].choice,
        "context_dependence": a["context_dependence"].score,
        "rhetorical_target": a["rhetorical_target"].choice,
        "generalization": a["generalization"].score,
        "sarcasm_irony": a["sarcasm_irony"].score,
    }


async def annotate_one(client, sem, text: str) -> dict | None:
    async with sem:
        try:
            resp = await client.system_one(
                state=text,
                questions=QUESTIONS,
                model="jev-latest",
            )
            return parse(resp)
        except TypeSafeError as e:
            print(f"Failed: {e!r}")
            return None


async def annotate_texts(
    client,
    sem,
    texts: list[str],
) -> list[dict | None]:
    tasks = [annotate_one(client, sem, t) for t in texts]
    return await tqdm_asyncio.gather(*tasks, leave=False)


async def process_chunk(
    client,
    sem,
    chunk: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    texts = chunk["text"].tolist()
    results = await annotate_texts(client, sem, texts)

    # One extra pass over anything that still failed.
    failed = [i for i, r in enumerate(results) if r is None]

    if failed:
        print(f"  Retrying {len(failed)} failed rows...")
        redo = await annotate_texts(
            client,
            sem,
            [texts[i] for i in failed],
        )

        for i, r in zip(failed, redo):
            results[i] = r

    n_failed = sum(r is None for r in results)

    rows = [r if r else {} for r in results]
    out = pd.concat(
        [chunk, pd.DataFrame(rows, index=chunk.index)],
        axis=1,
    )

    return out, n_failed


async def amain(df: pd.DataFrame) -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Maximum number of API requests that can be in flight simultaneously.
    sem = asyncio.Semaphore(CONCURRENCY)

    async with AsyncTypeSafeClient(
        api_key=TYPESAFE_API_KEY,
        timeout=120.0,
        retry=RETRY,
    ) as client:

        for start in range(0, len(df), CHUNK_SIZE):
            path = CHUNK_DIR / f"chunk_{start:06d}.parquet"

            # Resume: skip chunks that have already been completed.
            if path.exists():
                print(f"Skipping {path.name} (already exists)")
                continue

            chunk = df.iloc[start:start + CHUNK_SIZE]

            # ---------------------------------------------------------------
            # Benchmark this chunk.
            # ---------------------------------------------------------------
            chunk_start_time = time.perf_counter()

            out, n_failed = await process_chunk(
                client,
                sem,
                chunk,
            )

            elapsed = time.perf_counter() - chunk_start_time

            n_posts = len(chunk)
            posts_per_sec = n_posts / elapsed if elapsed > 0 else float("inf")
            failure_rate = n_failed / n_posts if n_posts > 0 else 0.0

            # Save checkpoint after the entire chunk has finished.
            out.to_parquet(path)

            print(
                f"\n{path.name} complete"
                f"\n  Posts:        {n_posts:,}"
                f"\n  Elapsed:      {elapsed:.2f} sec"
                f"\n  Throughput:   {posts_per_sec:.2f} posts/sec"
                f"\n  Failures:     {n_failed:,} ({failure_rate:.2%})"
                f"\n  Concurrency:  {CONCURRENCY}"
                f"\n  Saved:        {path}"
                f"\n"
            )

    full = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted(CHUNK_DIR.glob("chunk_*.parquet"))
        ]
    )

    full.to_parquet(OUTPUT_PATH)

    total_in = full["input_tokens"].sum()
    total_out = full["output_tokens"].sum()

    print("Parquet saved!")
    print(f"Number of rows: {len(full)}")
    print(f"Rows still missing annotations: {full['ragebait'].isna().sum()}")
    print(f"Input tokens spent: {total_in:.0f}")
    print(f"Output tokens spent: {total_out:.0f}")

    cost = total_in / 1e9 * 42
    print(f"Cost (note that output tokens are too cheap to measure): ${cost:.6f}")


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH)
    asyncio.run(amain(df))

