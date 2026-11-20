"""Step 6: tune the ambiguity band on the validation split (Algorithm 4.3) and define the
fast / balanced / thorough operating points (Sec. 4.3.5).

    python -m train.tune_thresholds [--eps-balanced 0.02] [--step 0.01]

Accuracy constraint: F1 on `val`. Fairness constraint: Delta-FPR on `fair_val`
(native vs non-native human essays), default delta = transformer-only disparity.
Writes models/operating_points.json and reports/threshold_tuning.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from veriscript.gate import tune_band

from .common import MODELS, PREDICTIONS, REPORTS, write_json

OPERATING_EPS = {"thorough": 0.005, "balanced": 0.02, "fast": 0.05}


def tune_all(preds: pd.DataFrame, step: float = 0.01, eps_map=None, delta=None, prob_col="p_h", fairness="worst_group") -> dict:
    eps_map = eps_map or OPERATING_EPS
    val = preds[preds.split == "val"]
    fair = preds[preds.split == "fair_val"]
    kw = {}
    if len(fair) and fair.native.notna().any():
        kw = dict(fair_p_h=fair[prob_col].values, fair_p_t=fair.p_t.values, fair_native=fair.native.astype(bool).values)
    out = {}
    for name, eps in eps_map.items():
        out[name] = tune_band(val[prob_col].values, val.p_t.values, val.label.values, eps=eps, delta=delta, step=step, fairness=fairness, **kw)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--eps-balanced", type=float, default=0.02)
    ap.add_argument("--delta", type=float, default=None, help="max FPR disparity (default: transformer-only disparity)")
    ap.add_argument("--fairness", default="worst_group", choices=["worst_group", "disparity", "none"])
    args = ap.parse_args(argv)
    if not PREDICTIONS.exists():
        raise SystemExit("Run `python -m train.predict` first.")
    preds = pd.read_parquet(PREDICTIONS)
    eps_map = dict(OPERATING_EPS, balanced=args.eps_balanced)
    res = tune_all(preds, args.step, eps_map, args.delta, fairness=args.fairness)
    # Record the alternatives for the report (not deployed).
    alternatives = {f: tune_all(preds, args.step, eps_map, args.delta, fairness=f) for f in ("worst_group", "disparity", "none") if f != args.fairness}
    for name, r in res.items():
        print(f"{name:9s} band=[{r['t_low']:.2f}, {r['t_high']:.2f}]  escalation={r['alpha']:.3f}  "
              f"F1 cascade={r['f1_cascade']:.4f} vs transformer={r['f1_transformer']:.4f}  feasible={r['feasible']}")
    points = {k: {"t_low": v["t_low"], "t_high": v["t_high"]} for k, v in res.items()}
    (MODELS).mkdir(exist_ok=True)
    (MODELS / "operating_points.json").write_text(json.dumps({"operating_points": points, "tuning": res}, indent=2))
    write_json({"deployed": {"fairness": args.fairness, **res}, "alternatives": alternatives}, REPORTS / "threshold_tuning.json")
    for f, alt in alternatives.items():
        print(f"  [{f}] " + "  ".join(f"{k}=[{v['t_low']:.2f},{v['t_high']:.2f}] a={v['alpha']:.3f}" for k, v in alt.items()))
    print(f"wrote {MODELS / 'operating_points.json'}")


if __name__ == "__main__":
    main()
