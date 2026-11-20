"""Step 4: train the heuristic head (LR and XGBoost), select by validation AUROC, calibrate.

    python -m train.train_heuristic [--head auto|lr|xgb] [--calibrator auto|platt|isotonic]
                                    [--exclude-family llama] [--out models/heuristic/heuristic.joblib]

Uses the cached features of the train / val / cal splits (Algorithm 4.1, lines 5-8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veriscript.config import load_config, resolve_path
from veriscript.heuristic import fit_heuristic

from .common import REPORTS, corpus_with_features, write_json


def train(head="auto", calibrator="auto", exclude_family=None, seed=42, df=None):
    df = corpus_with_features() if df is None else df
    if exclude_family:
        df = df[~((df.label == 1) & (df.generator_family == exclude_family))]
    parts = {s: df[df.split == s] for s in ("train", "val", "cal")}
    model, report = fit_heuristic(
        parts["train"], parts["train"].label.to_numpy(),
        parts["val"], parts["val"].label.to_numpy(),
        parts["cal"], parts["cal"].label.to_numpy(),
        head=head, calibrator=calibrator, seed=seed,
    )
    model.metadata["exclude_family"] = exclude_family
    return model, report


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="auto", choices=["auto", "lr", "xgb"])
    ap.add_argument("--calibrator", default="auto", choices=["auto", "platt", "isotonic", "none"])
    ap.add_argument("--exclude-family", default=None, help="drop this generator family (leave-one-generator-out)")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    ap.add_argument("--out", default=cfg["heuristic"]["model_path"])
    args = ap.parse_args(argv)

    model, report = train(args.head, args.calibrator, args.exclude_family, args.seed)
    out = resolve_path(args.out)
    model.save(out)
    write_json(report, REPORTS / f"heuristic_training{'_no_' + args.exclude_family if args.exclude_family else ''}.json")
    print(f"selected head: {report['selected_head']}  calibrator: {report['selected_calibrator']}")
    for h, info in report["heads"].items():
        print(f"  {h}: {info}")
    print(f"  val ECE: {report['val_ece']}")
    print(f"  POS bigrams: {report['pos_bigrams']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
