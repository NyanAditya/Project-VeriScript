"""Heuristic stage: standardisation, LR / XGBoost head, calibration and explanation (Sec. 4.3.3).

The serialisable :class:`HeuristicModel` holds everything learned from data. The run-time
:class:`HeuristicDetector` pairs it with a :class:`~veriscript.features.FeatureExtractor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .explain import Explanation, top_features
from .features import POS_PREFIX, SCALAR_FEATURES, FeatureExtractor

N_POS_BIGRAMS = 20


# ---------------------------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------------------------
def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-width bins (Eq. 4.5), using the positive-class probability as confidence."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


class PlattCalibrator:
    """p = sigmoid(a * s + b) fitted on held-out scores (Eq. 1.9)."""

    def fit(self, s, y):
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(s).reshape(-1, 1), y)
        return self

    def transform(self, s):
        return self.lr.predict_proba(np.asarray(s, dtype=float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, s, y):
        from sklearn.isotonic import IsotonicRegression

        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(s), y)
        return self

    def transform(self, s):
        return self.iso.predict(np.asarray(s, dtype=float))


class IdentityCalibrator:
    """Uncalibrated sigmoid of the raw score (used for the RQ2 ablation)."""

    def fit(self, s, y):
        return self

    def transform(self, s):
        return 1.0 / (1.0 + np.exp(-np.asarray(s, dtype=float)))


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
@dataclass
class HeuristicModel:
    feature_names: list[str]
    mean_: np.ndarray
    scale_: np.ndarray
    head_type: str
    head: Any
    calibrator_type: str
    calibrator: Any
    human_median: dict[str, float] = field(default_factory=dict)
    ai_median: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- vectorisation ----------------------------------------------------------------------
    def vectorize(self, raws: list[dict[str, float]] | pd.DataFrame) -> np.ndarray:
        if isinstance(raws, pd.DataFrame):
            df = raws.reindex(columns=self.feature_names)
        else:
            df = pd.DataFrame([{k: r.get(k, 0.0) for k in self.feature_names} for r in raws])
        return df.fillna(0.0).to_numpy(dtype=np.float64)

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    # -- scoring ----------------------------------------------------------------------------
    def raw_score(self, Xs: np.ndarray) -> np.ndarray:
        """Uncalibrated log-odds score s_h(d) of the head."""
        if self.head_type == "lr":
            return self.head.decision_function(Xs)
        import xgboost as xgb

        return self.head.get_booster().predict(xgb.DMatrix(Xs, feature_names=_xgb_names(self.feature_names)), output_margin=True)

    def predict_proba_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (calibrated p_h, raw score) for a raw (unstandardised) feature matrix."""
        s = self.raw_score(self.standardize(X))
        return np.clip(self.calibrator.transform(s), 0.0, 1.0), s

    def contributions(self, x_std: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the log-odds for one standardised vector."""
        if self.head_type == "lr":
            return self.head.coef_[0] * x_std
        import xgboost as xgb

        dm = xgb.DMatrix(x_std.reshape(1, -1), feature_names=_xgb_names(self.feature_names))
        return self.head.get_booster().predict(dm, pred_contribs=True)[0, :-1]

    # -- persistence -------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HeuristicModel":
        return joblib.load(path)


def _xgb_names(names: list[str]) -> list[str]:
    # XGBoost forbids [, ] and < in feature names.
    return [n.replace("[", "(").replace("]", ")").replace("<", "lt") for n in names]


# ---------------------------------------------------------------------------------------------
# Training (Algorithm 4.1, lines 5-8)
# ---------------------------------------------------------------------------------------------
def select_pos_bigrams(df: pd.DataFrame, y: np.ndarray, k: int = N_POS_BIGRAMS, seed: int = 42) -> list[str]:
    from sklearn.feature_selection import mutual_info_classif

    cols = [c for c in df.columns if c.startswith(POS_PREFIX)]
    if not cols:
        return []
    X = df[cols].fillna(0.0).to_numpy()
    # Only consider bigrams that occur in at least 1% of training documents.
    keep = (X > 0).mean(axis=0) >= 0.01
    cols = [c for c, kp in zip(cols, keep) if kp]
    X = X[:, keep]
    mi = mutual_info_classif(X, y, random_state=seed)
    order = np.argsort(-mi)[:k]
    return [cols[i] for i in order]


def _fit_lr(Xs, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    grid = GridSearchCV(
        LogisticRegression(solver="lbfgs", max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    grid.fit(Xs, y)
    return grid.best_estimator_, {"C": grid.best_params_["C"], "cv_auroc": float(grid.best_score_)}


def _fit_xgb(Xs, y, Xv, yv, names, seed, n_iter: int = 12):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    best, best_auc, best_params = None, -1.0, None
    fnames = _xgb_names(names)
    Xs_df = pd.DataFrame(Xs, columns=fnames)
    Xv_df = pd.DataFrame(Xv, columns=fnames)
    for _ in range(n_iter):
        params = {
            "max_depth": int(rng.integers(3, 7)),
            "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
            "min_child_weight": float(rng.choice([1, 3, 5])),
            "reg_lambda": float(rng.choice([1.0, 3.0, 10.0])),
        }
        clf = xgb.XGBClassifier(
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        clf.fit(Xs_df, y, eval_set=[(Xv_df, yv)], verbose=False)
        auc = roc_auc_score(yv, clf.predict_proba(Xv_df)[:, 1])
        if auc > best_auc:
            best, best_auc, best_params = clf, auc, {**params, "n_trees": int(clf.best_iteration + 1)}
    return best, {**best_params, "val_auroc": float(best_auc)}


def fit_heuristic(
    train: pd.DataFrame,
    y_train: np.ndarray,
    val: pd.DataFrame,
    y_val: np.ndarray,
    cal: pd.DataFrame,
    y_cal: np.ndarray,
    head: str = "auto",
    calibrator: str = "auto",
    seed: int = 42,
) -> tuple[HeuristicModel, dict]:
    """Train LR and/or XGBoost, select by validation AUROC, calibrate on the calibration split."""
    from sklearn.metrics import roc_auc_score

    pos = select_pos_bigrams(train, y_train, seed=seed)
    names = list(SCALAR_FEATURES) + pos
    proto = HeuristicModel(names, np.zeros(len(names)), np.ones(len(names)), "lr", None, "none", None)
    Xtr, Xv, Xc = proto.vectorize(train), proto.vectorize(val), proto.vectorize(cal)
    mean_ = Xtr.mean(axis=0)
    scale_ = Xtr.std(axis=0)
    scale_[scale_ == 0] = 1.0
    Xtr_s, Xv_s, Xc_s = (Xtr - mean_) / scale_, (Xv - mean_) / scale_, (Xc - mean_) / scale_

    report: dict[str, Any] = {"n_features": len(names), "pos_bigrams": pos, "heads": {}}
    candidates = {}
    if head in ("auto", "lr"):
        lr, info = _fit_lr(Xtr_s, y_train, seed)
        info["val_auroc"] = float(roc_auc_score(y_val, lr.decision_function(Xv_s)))
        candidates["lr"] = lr
        report["heads"]["lr"] = info
    if head in ("auto", "xgb"):
        xg, info = _fit_xgb(Xtr_s, y_train, Xv_s, y_val, names, seed)
        candidates["xgb"] = xg
        report["heads"]["xgb"] = info
    chosen = max(candidates, key=lambda h: report["heads"][h]["val_auroc"])
    report["selected_head"] = chosen

    model = HeuristicModel(names, mean_, scale_, chosen, candidates[chosen], "none", IdentityCalibrator())
    s_cal = model.raw_score(Xc_s)
    s_val = model.raw_score(Xv_s)
    cals = {"none": IdentityCalibrator()}
    if calibrator in ("auto", "platt"):
        cals["platt"] = PlattCalibrator().fit(s_cal, y_cal)
    if calibrator in ("auto", "isotonic"):
        cals["isotonic"] = IsotonicCalibrator().fit(s_cal, y_cal)
    eces = {k: expected_calibration_error(y_val, c.transform(s_val)) for k, c in cals.items()}
    report["val_ece"] = eces
    if calibrator == "none":
        chosen_cal = "none"
    else:
        chosen_cal = min((k for k in cals if k != "none"), key=lambda k: eces[k])
    model.calibrator_type, model.calibrator = chosen_cal, cals[chosen_cal]
    report["selected_calibrator"] = chosen_cal

    ytr = np.asarray(y_train)
    model.human_median = {n: float(np.median(Xtr[ytr == 0, i])) for i, n in enumerate(names)}
    model.ai_median = {n: float(np.median(Xtr[ytr == 1, i])) for i, n in enumerate(names)}
    model.metadata = {"seed": seed, "n_train": int(len(ytr))}
    return model, report


# ---------------------------------------------------------------------------------------------
# Run-time detector
# ---------------------------------------------------------------------------------------------
class HeuristicDetector:
    """Detector interface (Fig. 5.4): features -> calibrated probability -> explanation."""

    name = "heuristic"

    def __init__(self, model: HeuristicModel, extractor: FeatureExtractor):
        self.model = model
        self.extractor = extractor

    @classmethod
    def from_path(cls, path: str | Path, extractor: FeatureExtractor) -> "HeuristicDetector":
        return cls(HeuristicModel.load(path), extractor)

    def features(self, text: str) -> dict[str, float]:
        return self.extractor.extract(text)

    def predict_proba(self, text_or_features: str | dict) -> float:
        raw = self.features(text_or_features) if isinstance(text_or_features, str) else text_or_features
        p, _ = self.model.predict_proba_matrix(self.model.vectorize([raw]))
        return float(p[0])

    def predict_proba_many(self, raws: list[dict]) -> np.ndarray:
        if not raws:
            return np.zeros(0)
        return self.model.predict_proba_matrix(self.model.vectorize(raws))[0]

    def explain(self, raw: dict[str, float], k: int = 5, note: str | None = None) -> Explanation:
        x = self.model.vectorize([raw])[0]
        contrib = self.model.contributions(self.model.standardize(x))
        return top_features(self.model, x, contrib, k=k, note=note)
