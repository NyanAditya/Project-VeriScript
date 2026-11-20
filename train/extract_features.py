"""Step 3: compute raw heuristic features for every corpus document (one GPT-2 pass each).

    python -m train.extract_features [--device cuda] [--limit N]

Resumable: results are checkpointed to data/processed/features.parquet every --chunk documents.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from veriscript.config import load_config
from veriscript.features import FeatureExtractor
from veriscript.preprocess import Preprocessor
from veriscript.utils import get_device

from .common import FEATURES, load_corpus


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    dev = get_device(args.device)
    pre = Preprocessor(min_words=cfg["preprocess"]["min_words"])
    rl = cfg["reference_lm"]
    ex = FeatureExtractor(rl["name"], dev, rl["max_tokens"], rl["min_sentence_tokens"], pre)

    corpus = load_corpus()
    done = pd.read_parquet(FEATURES) if FEATURES.exists() else pd.DataFrame(columns=["id"])
    todo = corpus[~corpus.id.isin(set(done.id))]
    if args.limit:
        todo = todo.iloc[: args.limit]
    print(f"device={dev}; {len(done)} cached, {len(todo)} to extract")

    frames = [done] if len(done) else []
    t0 = time.time()
    for start in range(0, len(todo), args.chunk):
        part = todo.iloc[start : start + args.chunk]
        rows = []
        for doc_id, analysis in zip(part.id, pre.analyze_many(part.text.tolist())):
            f = ex.extract_analysis(analysis)
            f["id"] = doc_id
            rows.append(f)
        frames.append(pd.DataFrame(rows))
        merged = pd.concat(frames, ignore_index=True)
        pos_cols = [c for c in merged.columns if c.startswith("pos_bigram:")]
        merged[pos_cols] = merged[pos_cols].fillna(0.0)
        merged.to_parquet(FEATURES, index=False)
        frames = [merged]
        n = start + len(part)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)} docs  ({rate:.1f} docs/s, ~{(len(todo) - n) / rate / 60:.1f} min left)", flush=True)
    print(f"wrote {FEATURES}")


if __name__ == "__main__":
    main()
