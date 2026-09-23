"""
train_ragebait_regressor.py
========================================
Fine-tunes a transformer (RoBERTa-base by default) as a regressor that maps
tweet text -> a continuous "ragebait score" in [0, 1].

Design:
  - Backbone: any HF encoder (default "roberta-base", matches the classifier
    baseline already used in this project for consistency).
  - Head: linear layer on top of the pooled [CLS]/mean embedding -> 1 scalar
    -> sigmoid, so predictions are always in [0, 1] by construction.
  - Loss: MSE (regression). We additionally report MAE, RMSE, and Spearman
    correlation, since Spearman tells you whether the *ranking* of tweets by
    "how ragebait-y" they are is accurate.
  - Hyperparameter search: Optuna, optimizing mean CV MAE (MAE is used as
    the optimization objective rather than MSE because tweet engagement/score
    distributions tend to be skewed with outliers, and MAE is less dominated
    by a few extreme points).
  - Cross-validation: K-fold (default 5) on the training split, both for the
    HPO objective (robustness against fold-specific luck) and to give a
    reliable estimate of generalization before committing to a final fit.
  - Final model: after HPO selects the best hyperparameters, we refit once
    on the FULL train+val data with those hyperparameters and evaluate on a
    held-out test split, then persist to disk for inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    import optuna
    from optuna.trial import Trial
except ImportError:  # pragma: no cover
    raise SystemExit(
        "optuna is not installed. Run: pip install optuna --break-system-packages"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ragebait_regressor")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_OUTPUT_DIR = Path("data/ml_data")
DEFAULT_MAX_LEN = 96          # generous for tweet-length text (<=280 chars)
DEFAULT_N_TRIALS = 12         # Optuna trials
DEFAULT_N_FOLDS = 3           # CV folds used both for HPO objective and final reporting
DEFAULT_TEST_SIZE = 0.05      # held-out test split, separate from CV
DEFAULT_SEED = 42
DEFAULT_HPO_SUBSAMPLE_SIZE = None  # if set, HPO trials run on a random subsample of
                                    # this many train+val rows instead of the full set;
                                    # the winning config is still refit/reported on ALL
                                    # data. Cuts HPO wall-clock roughly linearly with
                                    # subsample fraction, since hyperparameter rankings
                                    # are typically stable across dataset size while the
                                    # per-trial cost of a full K-fold pass is not.

SEARCH_SPACE = {
    "learning_rate": (1e-6, 5e-5, "log"),
    "batch_size": [16, 32, 64],
    "weight_decay": (0.0, 0.1, "linear"),
    "warmup_ratio": (0.0, 0.2, "linear"),
    "dropout": (0.0, 0.3, "linear"),
    "num_epochs": [2, 3, 4],
}

USE_AMP = torch.cuda.is_available()  # mixed precision only pays off with Tensor Cores (e.g. T4+)


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class TweetScoreDataset(Dataset):
    """Wraps tokenized tweet text + a float score target in [0, 1]."""

    def __init__(self, texts: list[str], scores: list[float], tokenizer, max_len: int):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["score"])
    if len(df) < n_before:
        log.warning(f"Dropped {n_before - len(df)} rows with non-numeric score.")

    out_of_range = ((df["score"] < 0) | (df["score"] > 1)).sum()
    if out_of_range:
        raise ValueError(
            f"{out_of_range} rows have 'score' outside [0, 1]. "
            "This function assumes scores are already normalized to that range."
        )

    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if len(df) < 50:
        log.warning(f"Only {len(df)} usable rows -- results will be noisy; "
                    "consider whether K-fold CV / HPO trial count should shrink.")
    return df


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class TweetScoreRegressor(nn.Module):
    """Transformer encoder + mean-pooling + linear head -> sigmoid([0, 1])."""

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = out.last_hidden_state  # (B, L, H)

        # Mean-pool over non-padding tokens (more stable than raw [CLS] for
        # short, noisy social-media text).
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        pooled = self.dropout(pooled)
        logit = self.head(pooled).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)  # bounded to [0, 1] by construction


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    model_name: str
    max_len: int
    learning_rate: float
    batch_size: int
    weight_decay: float
    warmup_ratio: float
    dropout: float
    num_epochs: int


def run_training(
    train_texts: list[str],
    train_scores: list[float],
    val_texts: list[str],
    val_scores: list[float],
    cfg: TrainConfig,
    tokenizer,
    device: torch.device,
    verbose: bool = False,
) -> tuple[TweetScoreRegressor, dict]:
    """Trains one model instance on (train_texts, train_scores) and evaluates
    on (val_texts, val_scores). Returns the trained model and eval metrics."""

    train_ds = TweetScoreDataset(train_texts, train_scores, tokenizer, cfg.max_len)
    val_ds = TweetScoreDataset(val_texts, val_scores, tokenizer, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = TweetScoreRegressor(cfg.model_name, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = loss_fn(preds, batch["score"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item() * len(batch["score"])
        if verbose:
            log.info(f"  epoch {epoch + 1}/{cfg.num_epochs} "
                      f"train_mse={epoch_loss / len(train_ds):.4f}")

    metrics = evaluate(model, val_loader, device)
    return model, metrics


@torch.no_grad()
def evaluate(model: TweetScoreRegressor, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
        all_preds.extend(preds.float().cpu().numpy().tolist())
        all_targets.extend(batch["score"].numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = float(np.sqrt(mse))
    spearman_corr, _ = spearmanr(all_targets, all_preds)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "spearman": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
    }


# --------------------------------------------------------------------------- #
# Optuna objective (K-fold CV inside each trial)
# --------------------------------------------------------------------------- #

def make_objective(
    texts: list[str],
    scores: list[float],
    model_name: str,
    max_len: int,
    n_folds: int,
    device: torch.device,
    seed: int,
) -> Callable[[Trial], float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    texts_arr = np.array(texts, dtype=object)
    scores_arr = np.array(scores, dtype=np.float32)

    def objective(trial: Trial) -> float:
        cfg = TrainConfig(
            model_name=model_name,
            max_len=max_len,
            learning_rate=trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"][:2], log=True),
            batch_size=trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
            weight_decay=trial.suggest_float("weight_decay", *SEARCH_SPACE["weight_decay"][:2]),
            warmup_ratio=trial.suggest_float("warmup_ratio", *SEARCH_SPACE["warmup_ratio"][:2]),
            dropout=trial.suggest_float("dropout", *SEARCH_SPACE["dropout"][:2]),
            num_epochs=trial.suggest_categorical("num_epochs", SEARCH_SPACE["num_epochs"]),
        )

        fold_maes = []
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(texts_arr)):
            set_seed(seed + fold_idx)  # vary init slightly per fold, still reproducible
            _, metrics = run_training(
                texts_arr[tr_idx].tolist(), scores_arr[tr_idx].tolist(),
                texts_arr[va_idx].tolist(), scores_arr[va_idx].tolist(),
                cfg, tokenizer, device,
            )
            fold_maes.append(metrics["mae"])

            # Let Optuna prune clearly-bad trials early instead of paying for
            # every fold.
            trial.report(float(np.mean(fold_maes)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_maes))

    return objective


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def train_ragebait_regressor(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_len: int = DEFAULT_MAX_LEN,
    n_trials: int = DEFAULT_N_TRIALS,
    n_folds: int = DEFAULT_N_FOLDS,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    hpo_subsample_size: int | None = DEFAULT_HPO_SUBSAMPLE_SIZE,
) -> dict:
    """Runs the full pipeline: validate data -> Optuna HPO with K-fold CV ->
    refit best config on full train+val -> evaluate on held-out test ->
    save final model + tokenizer + metadata to `output_dir`.

    Args:
        df: DataFrame with columns "text" (str) and "score" (float in [0, 1]).
        model_name: HF model id for the encoder backbone.
        output_dir: where to persist the final model for inference.
        max_len: max token length for tokenization.
        n_trials: number of Optuna trials.
        n_folds: number of CV folds used per trial and for final reporting.
        test_size: fraction held out (once, before HPO) as the final test set.
        seed: random seed for reproducibility.
        hpo_subsample_size: if set, Optuna trials run K-fold CV on a random
            subsample of this many rows from train+val instead of the full
            set (useful for large datasets, where full-data HPO is too slow
            to be worth repeating n_trials times). The winning config is
            still refit and reported on the FULL train+val/test data
            regardless of this setting -- only the search itself is
            subsampled. Leave as None to run HPO on the full train+val set.

    Returns:
        dict with keys: "model_dir", "best_params", "cv_metrics" (mean/std
        per metric across final CV), and "test_metrics" (on the held-out
        test set after refitting on all train+val data).
    """
    set_seed(seed)
    device = get_device()
    log.info(f"Using device: {device}")

    df = validate_dataframe(df)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )
    log.info(f"Split: {len(train_val_df)} train+val / {len(test_df)} held-out test "
              f"(test set is NEVER touched during HPO/CV).")

    # ---- Hyperparameter search (Optuna, objective = mean CV MAE) ----
    # Optionally run the search itself on a subsample of train+val: hyperparameter
    # rankings are typically stable across dataset size, but a full K-fold pass
    # per trial is not -- on large datasets this is where the time actually goes.
    if hpo_subsample_size is not None and hpo_subsample_size < len(train_val_df):
        hpo_df = train_val_df.sample(n=hpo_subsample_size, random_state=seed)
        log.info(f"HPO will search on a subsample of {len(hpo_df)} rows "
                  f"(out of {len(train_val_df)} train+val rows); "
                  "the winning config is refit on the full set afterward.")
    else:
        hpo_df = train_val_df
        if hpo_subsample_size is not None:
            log.info(f"hpo_subsample_size ({hpo_subsample_size}) >= train+val size "
                      f"({len(train_val_df)}); running HPO on the full set.")

    objective = make_objective(
        texts=hpo_df["text"].tolist(),
        scores=hpo_df["score"].tolist(),
        model_name=model_name,
        max_len=max_len,
        n_folds=n_folds,
        device=device,
        seed=seed,
    )
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    log.info(f"Starting Optuna search: {n_trials} trials, {n_folds}-fold CV per trial, "
              f"on {len(hpo_df)} rows.")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    log.info(f"Best params: {best_params} (mean CV MAE={study.best_value:.4f})")

    best_cfg = TrainConfig(model_name=model_name, max_len=max_len, **best_params)

    # ---- Report CV metrics for the winning config (all metrics, not just MAE) ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    texts_arr = np.array(train_val_df["text"].tolist(), dtype=object)
    scores_arr = np.array(train_val_df["score"].tolist(), dtype=np.float32)

    cv_fold_metrics = []
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(texts_arr)):
        set_seed(seed + fold_idx)
        _, metrics = run_training(
            texts_arr[tr_idx].tolist(), scores_arr[tr_idx].tolist(),
            texts_arr[va_idx].tolist(), scores_arr[va_idx].tolist(),
            best_cfg, tokenizer, device, verbose=True,
        )
        log.info(f"  fold {fold_idx + 1}/{n_folds}: {metrics}")
        cv_fold_metrics.append(metrics)

    cv_summary = {
        f"{k}_mean": float(np.mean([m[k] for m in cv_fold_metrics]))
        for k in cv_fold_metrics[0]
    }
    cv_summary.update({
        f"{k}_std": float(np.std([m[k] for m in cv_fold_metrics]))
        for k in cv_fold_metrics[0]
    })
    log.info(f"CV summary (best config): {cv_summary}")

    # ---- Refit on ALL train+val data with the winning config ----
    log.info("Refitting best config on full train+val split before final test evaluation.")
    set_seed(seed)
    final_model, test_metrics = run_training(
        train_val_df["text"].tolist(), train_val_df["score"].tolist(),
        test_df["text"].tolist(), test_df["score"].tolist(),
        best_cfg, tokenizer, device, verbose=True,
    )
    log.info(f"Held-out test metrics: {test_metrics}")

    # ---- Persist model, tokenizer, and metadata for inference ----
    torch.save(final_model.state_dict(), output_dir / "model_state.pt")
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "model_name": model_name,
        "max_len": max_len,
        "dropout": best_cfg.dropout,
        "best_params": best_params,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
        "n_train_val": len(train_val_df),
        "n_test": len(test_df),
        "hpo_subsample_size": hpo_subsample_size,
        "n_hpo_rows": len(hpo_df),
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved final model + tokenizer + metadata to {output_dir}")

    return {
        "model_dir": str(output_dir),
        "best_params": best_params,
        "cv_metrics": cv_summary,
        "test_metrics": test_metrics,
    }


# --------------------------------------------------------------------------- #
# Inference helper (lightweight, no Optuna/training deps needed at call time)
# --------------------------------------------------------------------------- #

def load_regressor_for_inference(model_dir: str | Path) -> Callable[[list[str]], list[float]]:
    """Loads a saved model from `model_dir` and returns a function that maps
    a list of tweet strings -> a list of scores in [0, 1]."""
    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TweetScoreRegressor(metadata["model_name"], dropout=metadata["dropout"])
    model.load_state_dict(torch.load(model_dir / "model_state.pt", map_location=device))
    model.to(device)
    model.eval()
    max_len = metadata["max_len"]

    @torch.no_grad()
    def predict(texts: list[str]) -> list[float]:
        enc = tokenizer(
            texts, truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        )
        preds = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        return preds.cpu().numpy().tolist()

    return predict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the ragebait score regressor.")
    p.add_argument("--data", type=str, required=True,
                   help="Path to a Parquet/CSV file with 'text' and 'score' columns.")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--hpo-subsample-size", type=int, default=DEFAULT_HPO_SUBSAMPLE_SIZE,
                   help="Run Optuna trials on a random subsample of this many "
                        "train+val rows instead of the full set (final refit still "
                        "uses all data). Recommended for large datasets.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data)
    if data_path.suffix == ".csv":
        input_df = pd.read_csv(data_path)
    else:
        input_df = pd.read_parquet(data_path)

    result = train_ragebait_regressor(
        input_df,
        model_name=args.model_name,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        test_size=args.test_size,
        seed=args.seed,
        hpo_subsample_size=args.hpo_subsample_size,
    )
    print(json.dumps(result, indent=2))