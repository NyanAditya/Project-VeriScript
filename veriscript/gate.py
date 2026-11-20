"""Confidence gate and validation-based band tuning (Eq. 3.2, Algorithm 4.3, Eq. 4.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAST_PATH = "fast_path"
ESCALATED = "escalated"


@dataclass
class ConfidenceGate:
    t_low: float = 0.1
    t_high: float = 0.9
    tau: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.t_low <= self.t_high <= 1.0:
            raise ValueError(f"need 0 <= t_low <= t_high <= 1, got {self.t_low}, {self.t_high}")

    def route(self, p_h: float) -> str:
        """Escalate iff T_low <= p_h <= T_high (band edges are inclusive)."""
        return ESCALATED if self.t_low <= p_h <= self.t_high else FAST_PATH

    def escalate_mask(self, p_h: np.ndarray) -> np.ndarray:
        p_h = np.asarray(p_h)
        return (p_h >= self.t_low) & (p_h <= self.t_high)

    def combine(self, p_h: np.ndarray, p_t: np.ndarray) -> np.ndarray:
        return np.where(self.escalate_mask(p_h), p_t, p_h)


# ---------------------------------------------------------------------------------------------
# Vectorised evaluation of every band on cached predictions
# ---------------------------------------------------------------------------------------------
def _f1(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """F1 along the last axis for boolean predictions of shape (..., N)."""
    yb = y.astype(bool)
    tp = (pred & yb).sum(-1)
    fp = (pred & ~yb).sum(-1)
    fn = (~pred & yb).sum(-1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def band_grid(step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(0.5 / step)) + 1
    return np.linspace(0.0, 0.5, n), np.linspace(0.5, 1.0, n)


def evaluate_bands(
    p_h: np.ndarray,
    p_t: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    step: float = 0.01,
    fair_p_h: np.ndarray | None = None,
    fair_p_t: np.ndarray | None = None,
    fair_native: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Escalation rate, F1 and (optionally) FPR disparity for every (T_low, T_high) on the grid."""
    p_h, p_t, y = np.asarray(p_h, float), np.asarray(p_t, float), np.asarray(y, int)
    lows, highs = band_grid(step)
    TL, TH = np.meshgrid(lows, highs, indexing="ij")  # (nl, nh)
    esc = (p_h[None, None, :] >= TL[..., None]) & (p_h[None, None, :] <= TH[..., None])
    p = np.where(esc, p_t[None, None, :], p_h[None, None, :])
    out = {
        "t_low": TL,
        "t_high": TH,
        "alpha": esc.mean(-1),
        "f1": _f1(p >= tau, y),
    }
    if fair_p_h is not None:
        fh, ft = np.asarray(fair_p_h, float), np.asarray(fair_p_t, float)
        native = np.asarray(fair_native, bool)
        fesc = (fh[None, None, :] >= TL[..., None]) & (fh[None, None, :] <= TH[..., None])
        fpos = np.where(fesc, ft[None, None, :], fh[None, None, :]) >= tau  # all texts are human
        fpr_nn = fpos[..., ~native].mean(-1) if (~native).any() else np.zeros(TL.shape)
        fpr_n = fpos[..., native].mean(-1) if native.any() else np.zeros(TL.shape)
        out["fpr_non_native"] = fpr_nn
        out["fpr_native"] = fpr_n
        out["delta_fpr"] = fpr_nn - fpr_n
    return out


def tune_band(
    p_h,
    p_t,
    y,
    eps: float = 0.02,
    delta: float | None = None,
    tau: float = 0.5,
    step: float = 0.01,
    fair_p_h=None,
    fair_p_t=None,
    fair_native=None,
    fairness: str = "worst_group",
) -> dict:
    """Algorithm 4.3: the band with the lowest escalation rate subject to
    F1_cascade >= F1_transformer - eps and a fairness constraint on the native / non-native set.

    fairness:
      ``worst_group`` (default) - FPR on non-native writing no higher than the transformer-only
                                  detector's (hypothesis H4). Cannot be met by raising native FPR.
      ``disparity``             - Delta-FPR <= delta (success criterion P5 as stated in the report);
                                  ``delta`` defaults to the transformer-only disparity + 1 SE.
      ``none``                  - accuracy constraint only.
    Ties in escalation rate are broken by higher F1.
    """
    grid = evaluate_bands(p_h, p_t, y, tau, step, fair_p_h, fair_p_t, fair_native)
    y = np.asarray(y, int)
    f1_t = float(_f1(np.asarray(p_t) >= tau, y))
    ok = grid["f1"] >= f1_t - eps - 1e-12
    fair_info = {}
    if "delta_fpr" in grid:
        fp = np.asarray(fair_p_t) >= tau
        native = np.asarray(fair_native, bool)
        fpr_nn_t = float(fp[~native].mean()) if (~native).any() else 0.0
        fpr_n_t = float(fp[native].mean()) if native.any() else 0.0
        d_t = fpr_nn_t - fpr_n_t
        # Fairness sets are small, so the transformer's disparity is itself uncertain. By default the
        # cascade may exceed it by one standard error of that estimate (difference of two proportions).
        se = float(np.sqrt(fpr_nn_t * (1 - fpr_nn_t) / max((~native).sum(), 1) + fpr_n_t * (1 - fpr_n_t) / max(native.sum(), 1)))
        if delta is None:
            delta = max(d_t, 0.0) + se
        if fairness == "disparity":
            ok &= grid["delta_fpr"] <= delta + 1e-12
        if fairness in ("disparity", "worst_group"):
            # H4: never flag non-native writers more often than the transformer-only detector.
            ok &= grid["fpr_non_native"] <= fpr_nn_t + 1e-12
        fair_info = {"fairness": fairness, "delta_fpr_transformer": d_t, "delta_se": se, "delta": delta,
                     "fpr_non_native_transformer": fpr_nn_t, "fpr_native_transformer": fpr_n_t}
    if not ok.any():
        # Fall back to the transformer-only band (always feasible for accuracy).
        i, j = 0, grid["t_high"].shape[1] - 1
        feasible = False
    else:
        alpha = np.where(ok, grid["alpha"], np.inf)
        cand = np.argwhere(alpha == alpha.min())
        best = max(cand, key=lambda ij: grid["f1"][tuple(ij)])
        i, j = int(best[0]), int(best[1])
        feasible = True
    res = {
        "t_low": float(grid["t_low"][i, j]),
        "t_high": float(grid["t_high"][i, j]),
        "alpha": float(grid["alpha"][i, j]),
        "f1_cascade": float(grid["f1"][i, j]),
        "f1_transformer": f1_t,
        "eps": eps,
        "feasible": feasible,
        **fair_info,
    }
    if "delta_fpr" in grid:
        res["delta_fpr_cascade"] = float(grid["delta_fpr"][i, j])
        res["fpr_non_native_cascade"] = float(grid["fpr_non_native"][i, j])
        res["fpr_native_cascade"] = float(grid["fpr_native"][i, j])
    return res


def tradeoff_curve(p_h, p_t, y, cost_h: float, cost_t: float, lambdas=None, tau: float = 0.5, step: float = 0.01):
    """Sweep lambda in J = (1 - F1) + lambda * E[C] / (C_h + C_t) (Eq. 4.7); returns operating points."""
    grid = evaluate_bands(p_h, p_t, y, tau, step)
    rel_cost = (cost_h + grid["alpha"] * cost_t) / (cost_h + cost_t)
    lambdas = np.logspace(-3, 1, 40) if lambdas is None else lambdas
    pts = []
    for lam in lambdas:
        J = (1 - grid["f1"]) + lam * rel_cost
        i, j = np.unravel_index(np.argmin(J), J.shape)
        pts.append(
            {
                "lambda": float(lam),
                "t_low": float(grid["t_low"][i, j]),
                "t_high": float(grid["t_high"][i, j]),
                "alpha": float(grid["alpha"][i, j]),
                "f1": float(grid["f1"][i, j]),
                "relative_cost": float(rel_cost[i, j]),
            }
        )
    return pts
