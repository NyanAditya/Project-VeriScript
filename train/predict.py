"""Cache the predictions of both stages on every evaluation split.

    python -m train.predict [--heuristic models/heuristic/heuristic.joblib]
                            [--transformer models/transformer/roberta-base] [--tag main]

Threshold tuning (Algorithm 4.3) and the evaluation reuse these cached predictions, so the
band search needs no further model inference. The same detector classes used at run time
produce them, so what is evaluated is what is deployed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from veriscript.config import load_config, resolve_path
from veriscript.heuristic import HeuristicModel
from veriscript.transformer import TransformerDetector
from veriscript.utils import get_device

from .common import PROCESSED, corpus_with_features

EVAL_SPLITS = ("cal", "val", "test", "heldout_generator", "raid_attack", "mage_ood", "mage_ood_para", "fair_val", "fair_test")


def heuristic_predictions(model: HeuristicModel, df: pd.DataFrame) -> pd.DataFrame:
    p, s = model.predict_proba_matrix(model.vectorize(df))
    return pd.DataFrame({"id": df.id.values, "p_h": p, "s_h": s, "p_h_uncal": 1 / (1 + np.exp(-s))})


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic", default=cfg["heuristic"]["model_path"])
    ap.add_argument("--transformer", default=cfg["transformer"]["model_dir"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--splits", nargs="*", default=list(EVAL_SPLITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-transformer", action="store_true")
    args = ap.parse_args(argv)

    df = corpus_with_features()
    df = df[df.split.isin(args.splits)].reset_index(drop=True)
    model = HeuristicModel.load(resolve_path(args.heuristic))
    out = heuristic_predictions(model, df)
    print(f"heuristic predictions: {len(out)}")

    if not args.skip_transformer:
        tc = cfg["transformer"]
        trf = TransformerDetector(resolve_path(args.transformer), tc["max_len"], tc["stride"], get_device(args.device), 32)
        t0 = time.time()
        p_t, n_w = [], []
        step = 512
        for i in range(0, len(df), step):
            p, n = trf.predict_proba_many(df.text.iloc[i : i + step].tolist())
            p_t.append(p)
            n_w.append(n)
            print(f"  transformer {min(i + step, len(df))}/{len(df)}  ({time.time() - t0:.0f}s)", flush=True)
        out["p_t"] = np.concatenate(p_t)
        out["n_windows"] = np.concatenate(n_w)
    else:
        out["p_t"] = np.nan
        out["n_windows"] = 0

    meta_cols = ["id", "split", "label", "source_dataset", "domain", "generator", "generator_family", "attack", "native", "n_words"]
    res = df[meta_cols].merge(out, on="id")
    path = PROCESSED / ("predictions.parquet" if args.tag == "main" else f"predictions_{args.tag}.parquet")
    res.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
