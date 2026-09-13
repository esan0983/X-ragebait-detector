"""
api/annotate_ragebait.py
===========================================
Annotates X (Twitter) posts as Ragebait / Controversial Non-Ragebait / None of the Above
using the OpenAI API, optimized for minimum cost via:
  1. The Batch API (50% cheaper than synchronous calls, since this is an offline job
     with no latency requirement).
  2. Chunking many tweets into a single request (amortizes the fixed cost of the
     system prompt across N tweets instead of paying for it 20,000 times).
  3. Structured Outputs (json_schema) constrained to a fixed enum, so the model
     can't ramble and can't pad the response with anything beyond the label itself
     (minimizes output tokens, which are the expensive side of the ledger).
  4. reasoning_effort="low" and temperature=0, as specified, to keep this a cheap,
     deterministic classification task rather than an expensive deliberation task.

Input:  data/gpt_input/gpt_input_1.parquet   (one column: 'text', 20,000 rows)
Output: data/processed/gpt_output_1.parquet  (columns: 'tweet', 'label')

Usage:
    export OPENAI_API_KEY=sk-...
    python annotate_ragebait.py
"""

import json
import time
import math
from pathlib import Path

import pandas as pd
from openai import OpenAI
from openai.types.batch import Batch

# --------------------------------------------------------------------------
# CONFIG — the knobs you'll actually touch
# --------------------------------------------------------------------------

MODEL = "gpt-5.6-luna"          # <-- update if the official API model string differs
REASONING_EFFORT = "low"
TEMPERATURE = 0

INPUT_PATH = Path("data/gpt_input/gpt_input_1.parquet")
OUTPUT_PATH = Path("data/processed/gpt_output_1.parquet")
BATCH_WORKDIR = Path("data/batch_jobs")

# How many tweets to pack into a single model call. Bigger = cheaper (system
# prompt cost is amortized further) but riskier (one bad/truncated response
# loses more rows, and very long inputs increase mis-count risk). 25-40 is a
# solid middle ground for short (50-280 char) posts.
CHUNK_SIZE = 30

# How often to poll the batch job for completion.
POLL_INTERVAL_SECONDS = 30

LABELS = ["Ragebait", "Controversial Non-Ragebait", "None of the Above"]

# --------------------------------------------------------------------------
# PROMPT TEMPLATE — paste your finalized annotation guidelines into the
# {criteria} slot below. Nothing else in the script needs to change when
# you do.
# --------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are an expert content moderator with 10 years of experience. \
Analyze the provided X posts and annotate each one with exactly one of the \
following labels: Ragebait, Controversial Non-Ragebait, None of the Above.

{criteria}

You will be given a numbered list of posts. Return a label for every post, \
using the same id numbers you were given. Do not skip any id. Do not invent ids."""

# <<< PASTE YOUR DETAILED CRITERIA HERE >>>
ANNOTATION_CRITERIA = """
LABEL DEFINITIONS

Evaluate each post as a standalone text string. Do not infer external context, attachments, images, links, or the author's identity/history unless that context is explicitly present in the post text itself. Base the label only on what is written.

1. Ragebait
Definition: A post whose primary discernible function is to provoke negative emotion (anger, outrage, contempt, indignation) in order to generate engagement (replies, reposts, quote-posts), regardless of whether the author sincerely holds the stated view.
Necessary condition: The text alone, without external context, must be sufficient to provoke a negative emotional reaction in a plausible reader.
Non-necessary conditions (may or may not be present; presence increases confidence but is not required): offensive language, hate speech, slurs, toxic phrasing, insults.
Distinguishing signal: Statements that are extreme, absolute, or overgeneralized to a degree that suggests performative provocation rather than a genuine, considered opinion (e.g., sweeping negative claims about a demographic group stated as flat fact, with no qualification or support).
Label as Ragebait if: the post's content and phrasing appear engineered to provoke anger/outrage as their main function, independent of topic.

2. Controversial Non-Ragebait
Definition: A post that addresses a topic prone to disagreement (e.g., politics, religion, health, social issues) but is not primarily engineered to provoke negative emotion.
Distinguishing signals:
- Tone is measured, polite, or open-minded rather than inflammatory.
- Claims are qualified, explained, or supported (e.g., reasoning, evidence, acknowledgment of nuance/counterpoints) rather than stated as absolute, unqualified fact.
- The post reads as a genuine expression of belief rather than a provocation engineered for reaction.
Label as Controversial Non-Ragebait if: the topic is contentious AND the tone/construction indicates sincere argumentation rather than provocation.

3. None of the Above
Definition: Applies when a post does not meet the criteria for Ragebait or Controversial Non-Ragebait. Includes two specific cases:
Case A — Incomplete/context-dependent post: The post's meaning or rhetorical force depends on an attached image, video, link, or other external context not present in the text itself, such that the text alone reads as incomplete or under-specified. Do not label such posts as Ragebait even if the attachment (which you cannot see) would plausibly make it provocative. Label based on the standalone text only.
Case B — Positive-emotion clickbait: The post is constructed to drive engagement but targets positive or neutral emotions (humor, excitement, curiosity, harmless jokes) rather than negative/anger-based emotions. This includes posts that could later cause outrage once external context becomes known, but do not do so based on the standalone text alone.
Label as None of the Above if: the post is either context-dependent for its provocative effect (Case A) or is engagement-oriented but not negative-emotion-oriented (Case B), or otherwise fails to meet the definitions of categories 1 and 2.

DECISION PRIORITY (apply in order)
1. If the post's standalone text is insufficient to establish meaning or provocation without external attachments/links/context → None of the Above (Case A).
2. Else if the post's standalone text is designed to provoke negative emotion as its primary function → Ragebait.
3. Else if the post engages a contentious topic with a sincere, non-inflammatory tone → Controversial Non-Ragebait.
4. Else → None of the Above (Case B or default).
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(criteria=ANNOTATION_CRITERIA.strip())

# Structured output schema: forces the model to return ONLY an array of
# {id, label} objects, with label locked to our three-way enum. This is what
# lets us safely pack many tweets into one call and parse the result
# mechanically instead of regex-scraping free text.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "ragebait_annotations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string", "enum": LABELS},
                    },
                    "required": ["id", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["annotations"],
        "additionalProperties": False,
    },
}

client = OpenAI()


# --------------------------------------------------------------------------
# Step 1: load + chunk
# --------------------------------------------------------------------------

def load_tweets(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if "text" not in df.columns:
        raise ValueError(f"Expected a 'text' column, got: {list(df.columns)}")
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # global id we can always map back to
    return df


def make_chunks(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    n_chunks = math.ceil(len(df) / chunk_size)
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def chunk_to_user_message(chunk: pd.DataFrame) -> str:
    # Local ids (0..len(chunk)-1) within the chunk are simpler for the model
    # to track than global row ids, and we map them back ourselves afterward.
    lines = [f"{local_id}. {text}" for local_id, text in enumerate(chunk["text"])]
    return "Posts:\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# Step 2: build the batch input file (JSONL)
# --------------------------------------------------------------------------

def build_batch_requests(chunks: list[pd.DataFrame]) -> list[dict]:
    requests = []
    for chunk_idx, chunk in enumerate(chunks):
        requests.append({
            "custom_id": f"chunk-{chunk_idx}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "reasoning": {"effort": REASONING_EFFORT},
                "input": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_to_user_message(chunk)},
                ],
                "text": {"format": RESPONSE_SCHEMA},
            },
        })
    return requests


def write_jsonl(requests: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")


# --------------------------------------------------------------------------
# Step 3: submit to the Batch API and wait
# --------------------------------------------------------------------------

def submit_batch(jsonl_path: Path) -> str:
    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Submitted batch {batch.id} ({len(open(jsonl_path).readlines())} requests)")
    return batch.id


def wait_for_batch(batch_id: str) -> Batch:
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  status={batch.status}  "
              f"completed={batch.request_counts.completed}/{batch.request_counts.total}  "
              f"failed={batch.request_counts.failed}")
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(POLL_INTERVAL_SECONDS)


# --------------------------------------------------------------------------
# Step 4: parse results back into (tweet, label) rows
# --------------------------------------------------------------------------

def parse_batch_output(batch, chunks: list[pd.DataFrame]) -> pd.DataFrame:
    if batch.output_file_id is None:
        raise RuntimeError(f"Batch did not complete successfully: status={batch.status}")

    output_text = client.files.content(batch.output_file_id).text
    results_by_chunk = {}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        chunk_idx = int(custom_id.split("-")[1])

        if record.get("error") is not None:
            print(f"  chunk {chunk_idx} errored: {record['error']}")
            results_by_chunk[chunk_idx] = None
            continue

        body = record["response"]["body"]
        # Responses API: the structured JSON payload lives in output_text.
        raw_json = body["output"][-1]["content"][0]["text"]
        parsed = json.loads(raw_json)
        results_by_chunk[chunk_idx] = {a["id"]: a["label"] for a in parsed["annotations"]}

    rows = []
    for chunk_idx, chunk in enumerate(chunks):
        annotations = results_by_chunk.get(chunk_idx)
        for local_id, tweet_text in enumerate(chunk["text"]):
            label = annotations.get(local_id) if annotations else None
            rows.append({"tweet": tweet_text, "label": label})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"Loading tweets from {INPUT_PATH} ...")
    df = load_tweets(INPUT_PATH)
    print(f"Loaded {len(df)} tweets.")

    chunks = make_chunks(df, CHUNK_SIZE)
    print(f"Split into {len(chunks)} chunks of up to {CHUNK_SIZE} tweets each.")

    requests = build_batch_requests(chunks)
    BATCH_WORKDIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = BATCH_WORKDIR / "gpt_input_1_batch.jsonl"
    write_jsonl(requests, jsonl_path)
    print(f"Wrote batch request file to {jsonl_path}")

    batch_id = submit_batch(jsonl_path)
    batch = wait_for_batch(batch_id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status '{batch.status}', not 'completed'.")

    result_df = parse_batch_output(batch, chunks)

    n_missing = result_df["label"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} tweets came back without a label "
              f"(failed/errored chunks). Consider re-running those chunks.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    print(f"Saved {len(result_df)} annotated rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()