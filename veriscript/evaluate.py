"""Evaluation toolkit (Sec. 4.3.7, 6.1.1.7, 6.1.5): accuracy, bootstrap CIs, latency, fairness."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from .heuristic import expected_calibration_error


def classification_metrics(y, p, tau: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= tau).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ece": expected_calibration_error(y, p),
    }
    neg = y == 0
    out["fpr"] = float(pred[neg].mean()) if neg.any() else float("nan")
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


def _f1_vec(y, pred):
    tp = (pred & (y == 1)).sum(-1)
    fp = (pred & (y == 0)).sum(-1)
    fn = (~pred & (y == 1)).sum(-1)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def _auroc_vec(y, p):
    """Rank-based AUROC for each row (Mann-Whitney U)."""
    from scipy.stats import rankdata

    ranks = rankdata(p, axis=-1)
    pos = (y == 1)
    n1 = pos.sum(-1)
    n0 = (~pos).sum(-1)
    s = (ranks * pos).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(y, p, tau: float = 0.5, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05, idx=None) -> dict:
    """Percentile bootstrap CIs for F1 and AUROC."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y))) if idx is None else idx
    yb, pb = y[idx], p[idx]
    f1 = _f1_vec(yb, pb >= tau)
    auc = _auroc_vec(yb, pb)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"f1_ci": [float(v) for v in np.nanpercentile(f1, q)], "auroc_ci": [float(v) for v in np.nanpercentile(auc, q)]}


def paired_bootstrap(y, p_a, p_b, tau: float = 0.5, n_boot: int = 1000, seed: int = 42) -> dict:
    """Paired bootstrap of F1(a) - F1(b) on identical resamples; returns mean diff, CI and p-value."""
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb = y[idx]
    da = _f1_vec(yb, np.asarray(p_a)[idx] >= tau)
    db = _f1_vec(yb, np.asarray(p_b)[idx] >= tau)
    diff = da - db
    p_two = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {
        "f1_diff": float(diff.mean()),
        "f1_diff_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "p_value": float(min(p_two, 1.0)),
    }


def fpr_disparity(p, native, tau: float = 0.5) -> dict:
    """All texts are human-written, so every positive is a false positive."""
    pred = np.asarray(p) >= tau
    native = np.asarray(native, bool)
    fpr_n = float(pred[native].mean()) if native.any() else float("nan")
    fpr_nn = float(pred[~native].mean()) if (~native).any() else float("nan")
    return {"fpr_native": fpr_n, "fpr_non_native": fpr_nn, "delta_fpr": fpr_nn - fpr_n, "n_native": int(native.sum()), "n_non_native": int((~native).sum())}


def recovery_speedup(f1_c, f1_h, f1_t, t_c, t_t) -> dict:
    """Eq. 4.8."""
    denom = f1_t - f1_h
    return {
        "recovery": float((f1_c - f1_h) / denom) if abs(denom) > 1e-9 else float("nan"),
        "speedup": float(t_t / t_c) if t_c > 0 else float("nan"),
    }


def latency_stats(times_ms: Sequence[float]) -> dict:
    t = np.asarray(times_ms, float)
    if len(t) == 0:
        return {"mean_ms": float("nan"), "p95_ms": float("nan"), "throughput_docs_per_s": float("nan"), "n": 0}
    return {
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "throughput_docs_per_s": float(len(t) / (t.sum() / 1e3)) if t.sum() > 0 else float("nan"),
        "n": int(len(t)),
    }


def measure_latency(fn: Callable[[str], object], texts: Sequence[str], warmup: int = 20) -> list[float]:
    """Latency protocol (Sec. 6.1.5): warm-up documents are not recorded."""
    for t in texts[:warmup]:
        fn(t)
    out = []
    for t in texts[warmup:]:
        t0 = time.perf_counter()
        fn(t)
        out.append(1e3 * (time.perf_counter() - t0))
    return out


def model_footprint(path) -> dict:
    """Size on disk (MB) of a model file or directory."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "size_mb": float("nan")}
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return {"path": str(p), "size_mb": round(sum(f.stat().st_size for f in files) / 2**20, 2)}


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def plot_roc(curves: dict[str, tuple], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y, probs: dict[str, np.ndarray], path, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=n_bins)
        ax.plot(mean, frac, marker="o", label=f"{name} (ECE={expected_calibration_error(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction AI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(points: list[dict], baselines: dict[str, tuple[float, float]], path, chosen: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [p["relative_cost"] for p in points]
    ys = [p["f1"] for p in points]
    ax.plot(xs, ys, "-o", ms=3, label="cascade operating points")
    for name, (c, f) in baselines.items():
        ax.scatter([c], [f], s=60, zorder=3, label=name)
    if chosen:
        for name, (c, f) in chosen.items():
            ax.annotate(name, (c, f), textcoords="offset points", xytext=(5, -10), fontsize=8)
            ax.scatter([c], [f], marker="x", color="k")
    ax.set_xlabel("Relative expected cost  E[C] / (C_h + C_t)")
    ax.set_ylabel("F1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
