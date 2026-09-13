"""
main/collection.py
========================================
Collects candidate posts for a ragebait-detection dataset using twikit.

Filters applied:
  - Posted within August 2026
  - (likes + replies + reposts) >= 500
  - 50 <= len(text) <= 280
  - Text content only -- no user/account fields are persisted

Design goals covered:
  - Random-ish sampling across many queries/topics rather than one account
  - Periodic incremental saves to data/raw/ as Parquet
  - Fully rerunnable / resumable (dedupes against what's already on disk)
  - Conservative, jittered rate-limiting + exponential backoff on errors

Usage:
    python collect_ragebait_data.py --target 5000
    python collect_ragebait_data.py --username you --email you@x.com --password ...

Auth:
    twikit needs a logged-in session. Cheapest path is to log in once and
    cache cookies to disk (cookies.json), then every future run just loads
    the cookie file -- no repeated logins, which is itself good practice
    for avoiding bans. See `ensure_login()`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import queries

try:
    from twikit import Client, TooManyRequests
except ImportError:
    print("twikit is not installed. Run: pip install twikit --break-system-packages")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATA_DIR = Path("data/raw")
PARQUET_PATH = DATA_DIR / "ragebait_candidates.parquet"
COOKIES_PATH = Path("data/raw/cookies.json")
STATE_PATH = DATA_DIR / "_collection_state.json"  # tracks query rotation progress

MONTH_START = datetime(2026, 8, 1, tzinfo=timezone.utc)
MONTH_END = datetime(2026, 9, 1, tzinfo=timezone.utc)  # exclusive

MIN_ENGAGEMENT = 100          # likes + replies + reposts
MIN_LEN, MAX_LEN = 50, 280    # character bounds on post text

MIN_SLEEP_SECS = 30.0         # baseline "be nice" delay between API calls
MAX_SLEEP_SECS = 45.0
LONG_COOLDOWN_EVERY = 15      # after N requests, take a longer breather
LONG_COOLDOWN_SECS = (180, 300)

SAVE_EVERY_N_NEW = 25         # flush to disk after this many new rows collected
RESULTS_PER_QUERY_PAGE = 20   # twikit search page size ballpark

MAX_BACKOFF_SECS = 900        # cap exponential backoff at 15 min
BACKOFF_BASE = 20.0

# A deliberately broad/diverse pool of search terms so we don't fixate on
# one account, topic, or community. Mixes politics, culture-war bait,
# sports, tech, relationships, etc. -- classic ragebait-prone domains.
# `min_faves` is applied query-side as a coarse pre-filter; we still
# re-verify the full engagement threshold ourselves after fetching.
SEARCH_QUERIES = queries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ragebait_collector")


# --------------------------------------------------------------------------- #
# Rate limiting / backoff helpers
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Enforces a jittered minimum delay between requests, plus periodic
    long cooldowns and exponential backoff on errors. This is the core
    anti-ban mechanism: real users don't fire requests at machine-gun pace
    or at perfectly regular intervals, so we randomize everything."""

    def __init__(self):
        self.request_count = 0
        self._consecutive_errors = 0

    async def wait(self):
        delay = random.uniform(MIN_SLEEP_SECS, MAX_SLEEP_SECS)
        self.request_count += 1

        if self.request_count % LONG_COOLDOWN_EVERY == 0:
            cooldown = random.uniform(*LONG_COOLDOWN_SECS)
            log.info(f"Cooldown break after {self.request_count} requests: "
                      f"sleeping {cooldown:.0f}s")
            await asyncio.sleep(cooldown)
        else:
            log.info(f"Sleeping {delay:.1f}s before next request "
                      f"(request #{self.request_count})")
            await asyncio.sleep(delay)

    async def backoff(self):
        """Call this after a rate-limit / transient error. Exponential
        backoff with jitter, capped at MAX_BACKOFF_SECS."""
        self._consecutive_errors += 1
        raw = min(MAX_BACKOFF_SECS, BACKOFF_BASE * (2 ** (self._consecutive_errors - 1)))
        jittered = raw * random.uniform(0.8, 1.2)
        log.warning(f"Backing off for {jittered:.0f}s "
                     f"(consecutive errors: {self._consecutive_errors})")
        await asyncio.sleep(jittered)

    def reset_errors(self):
        self._consecutive_errors = 0


# --------------------------------------------------------------------------- #
# Persistence / resumability
# --------------------------------------------------------------------------- #

@dataclass
class CollectionState:
    """Tracks rotation through the query pool across runs so a restart
    doesn't just hammer the same first few queries forever."""
    query_index: int = 0
    total_requests_made: int = 0
    seen_ids: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, existing_ids: set[str]) -> "CollectionState":
        if STATE_PATH.exists():
            raw = json.loads(STATE_PATH.read_text())
            state = cls(
                query_index=raw.get("query_index", 0),
                total_requests_made=raw.get("total_requests_made", 0),
                seen_ids=existing_ids,
            )
            log.info(f"Resumed state: query_index={state.query_index}, "
                      f"total_requests_made={state.total_requests_made}")
            return state
        return cls(seen_ids=existing_ids)

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps({
            "query_index": self.query_index,
            "total_requests_made": self.total_requests_made,
        }))


def load_existing_dataset() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        log.info(f"Loaded existing dataset: {len(df)} rows from {PARQUET_PATH}")
        return df
    cols = ["tweet_id", "text", "created_at", "likes", "replies",
            "reposts", "engagement_total", "query_source", "collected_at"]
    return pd.DataFrame(columns=cols)


def append_and_save(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset="tweet_id", keep="first", inplace=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PARQUET_PATH.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp_path, index=False, compression="snappy")
    tmp_path.replace(PARQUET_PATH)  # atomic-ish swap so a crash mid-write can't corrupt the file

    log.info(f"Saved dataset: +{len(new_rows)} new rows -> {len(combined)} total "
              f"({PARQUET_PATH})")
    return combined


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #

def in_target_month(created_at_str: str) -> bool:
    try:
        # twikit tweet.created_at is typically like "Wed Aug 12 14:23:01 +0000 2026"
        dt = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return False
    return MONTH_START <= dt < MONTH_END


def passes_filters(tweet) -> tuple[bool, dict]:
    text = (tweet.text or "").strip()
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return False, {}

    if not in_target_month(tweet.created_at):
        return False, {}

    likes = tweet.favorite_count or 0
    replies = tweet.reply_count or 0
    reposts = tweet.retweet_count or 0
    total = likes + replies + reposts
    if total < MIN_ENGAGEMENT:
        return False, {}

    return True, {
        "tweet_id": str(tweet.id),
        "text": text,
        "created_at": tweet.created_at,
        "likes": likes,
        "replies": replies,
        "reposts": reposts,
        "engagement_total": total,
    }

# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

async def ensure_login(client: Client, username: str | None, email: str | None,
                        password: str | None):
    """NOTE: twikit's programmatic client.login() is currently broken against
    X's live site (X now requires a JS-generated token/handshake that a plain
    HTTP client can't produce -- this surfaces as LoginRetired / error code
    366 "flow name LoginFlow is currently not accessible"). There is no
    reliable code-level fix for that from twikit's side right now, so this
    function does NOT attempt client.login() at all.

    Instead, log into x.com in a normal browser, export your session cookies
    (e.g. with a "Cookie-Editor" style extension) as JSON containing at least
    auth_token and ct0, and save that as cookies.json next to this script.
    """
    if COOKIES_PATH.exists():
        log.info(f"Loading cached session from {COOKIES_PATH}")
        client.load_cookies(str(COOKIES_PATH))
        return

    raise SystemExit(
        f"No {COOKIES_PATH} found.\n\n"
        "twikit's automated login is currently broken against X's site, so this "
        "script requires cookies exported from a real logged-in browser session:\n"
        "  1. Log into x.com normally in your browser.\n"
        "  2. Export cookies for x.com as JSON (e.g. via a 'Cookie-Editor' style "
        "extension), making sure 'auth_token' and 'ct0' are included.\n"
        f"  3. Save that JSON as {COOKIES_PATH} next to this script.\n"
        "  4. Re-run this script -- it will load the cookies directly."
    )


# --------------------------------------------------------------------------- #
# Main collection loop
# --------------------------------------------------------------------------- #

async def collect(target_total: int, max_requests: int | None):
    existing_df = load_existing_dataset()
    existing_ids = set(existing_df["tweet_id"].astype(str)) if len(existing_df) else set()
    state = CollectionState.load(existing_ids)
    limiter = RateLimiter()

    client = Client("en-US")
    args = parse_args()  # re-read for credentials (cheap, avoids threading extra params)
    await ensure_login(client, args.username, args.email, args.password)

    # Randomize query order each run so repeated restarts don't always
    # start from the same spot -- combined with state.query_index this
    # gives broad, shuffled coverage over time.
    queries = SEARCH_QUERIES.copy()
    random.shuffle(queries)

    new_rows_buffer: list[dict] = []
    current_df = existing_df
    requests_this_run = 0

    log.info(f"Starting collection. Current dataset size: {len(existing_df)}. "
              f"Target: {target_total}.")

    while len(current_df) + len(new_rows_buffer) < target_total:
        if max_requests is not None and requests_this_run >= max_requests:
            log.info("Hit --max-requests cap for this run, stopping.")
            break

        query = queries[state.query_index % len(queries)]
        state.query_index += 1

        # Query-side coarse filter narrows the firehose; we still re-check
        # everything in passes_filters() since search operators aren't exact.
        search_query = (
            f'{query} min_faves:{MIN_ENGAGEMENT // 3} lang:en '
            f'since:2026-08-01 until:2026-09-01'
        )

        try:
            log.info(f"[req #{state.total_requests_made + 1}] "
                      f"Searching: '{search_query}'")
            results = await client.search_tweet(search_query, product="Top")
            limiter.reset_errors()
        except TooManyRequests:
            log.warning("Rate limited by X (429). Entering backoff.")
            await limiter.backoff()
            continue
        except Exception as e:  # noqa: BLE001 -- want to survive any transient API hiccup
            log.warning(f"Request failed ({e!r}); backing off and retrying.")
            await limiter.backoff()
            continue

        state.total_requests_made += 1
        requests_this_run += 1

        tweets = list(results) if results else []
        random.shuffle(tweets)  # don't always process/keep in the API's default order

        found_this_page = 0
        for tweet in tweets:
            tid = str(tweet.id)
            if tid in state.seen_ids:
                continue
            ok, row = passes_filters(tweet)
            state.seen_ids.add(tid)
            if not ok:
                continue
            row["query_source"] = query
            row["collected_at"] = datetime.now(timezone.utc).isoformat()
            new_rows_buffer.append(row)
            found_this_page += 1

        log.info(f"  -> {found_this_page} qualifying posts from this query "
                  f"(buffer={len(new_rows_buffer)}, total so far="
                  f"{len(current_df) + len(new_rows_buffer)}/{target_total})")

        if len(new_rows_buffer) >= SAVE_EVERY_N_NEW:
            current_df = append_and_save(current_df, new_rows_buffer)
            new_rows_buffer = []
            state.save()

        await limiter.wait()

    # final flush
    if new_rows_buffer:
        current_df = append_and_save(current_df, new_rows_buffer)
    state.save()

    log.info(f"Done for this run. Dataset now has {len(current_df)} rows "
              f"at {PARQUET_PATH}.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect ragebait-candidate posts from X via twikit.")
    p.add_argument("--target", type=int, default=2000,
                   help="Stop once the dataset reaches this many total rows.")
    p.add_argument("--max-requests", type=int, default=None,
                   help="Optional cap on API requests made in this single run "
                        "(useful for a bounded daily session).")
    p.add_argument("--username", type=str, default=None,
                   help="Unused -- kept for CLI compatibility. Auth is via cookies.json now; "
                        "see ensure_login() docstring.")
    p.add_argument("--email", type=str, default=None, help="Unused -- see ensure_login() docstring.")
    p.add_argument("--password", type=str, default=None, help="Unused -- see ensure_login() docstring.")
    return p.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        asyncio.run(collect(cli_args.target, cli_args.max_requests))
    except KeyboardInterrupt:
        log.info("Interrupted by user -- progress already saved incrementally, safe to resume later.")