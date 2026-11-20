"""Paths and helpers shared by the training / evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from veriscript.config import PROJECT_ROOT

PROCESSED = PROJECT_ROOT / "data" / "processed"
CORPUS = PROCESSED / "corpus.parquet"
FEATURES = PROCESSED / "features.parquet"
PREDICTIONS = PROCESSED / "predictions.parquet"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
MAIN_SPLITS = ("train", "cal", "val", "test")


def load_corpus() -> pd.DataFrame:
    if not CORPUS.exists():
        raise SystemExit(f"{CORPUS} not found. Run `python -m data.build_corpus` first.")
    return pd.read_parquet(CORPUS)


def load_features() -> pd.DataFrame:
    if not FEATURES.exists():
        raise SystemExit(f"{FEATURES} not found. Run `python -m train.extract_features` first.")
    return pd.read_parquet(FEATURES)


def corpus_with_features() -> pd.DataFrame:
    c = load_corpus()
    f = load_features()
    return c.merge(f, on="id", how="inner")


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
