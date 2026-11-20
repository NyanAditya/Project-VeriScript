"""Shared fixtures: a cascade built from stub components so tests need no downloaded models."""

import numpy as np
import pandas as pd
import pytest

from veriscript.cascade import CascadeDetector
from veriscript.features import SCALAR_FEATURES, FeatureExtractor, pos_bigram_profile, stylometric_features
from veriscript.gate import ConfidenceGate
from veriscript.heuristic import HeuristicDetector, fit_heuristic
from veriscript.preprocess import Preprocessor


class StubExtractor(FeatureExtractor):
    """Replaces the GPT-2 pass with a deterministic function of the text."""

    def lm_features(self, analysis):
        n = len(analysis.words)
        uniq = len({w.lower() for w in analysis.words})
        ratio = uniq / max(n, 1)
        return {
            "mean_log_prob": -1.0 - 4 * ratio,
            "perplexity": float(np.exp(1.0 + 4 * ratio)),
            "std_log_prob": 1.0 + ratio,
            "min_sentence_perplexity": 5.0,
            "max_sentence_perplexity": 50.0 * ratio,
            "top10_rank_fraction": 1.0 - ratio,
            "top100_rank_fraction": 0.2,
            "tail_rank_fraction": ratio / 5,
            "mean_entropy": 2.0 + ratio,
            "perplexity_burstiness": ratio,
        }


class StubTransformer:
    """'AI' iff the text contains the word 'delve' (deterministic)."""

    available = True

    def predict_with_windows(self, text):
        return (0.97 if "delve" in text.lower() else 0.03), 1

    def predict_proba_many(self, texts):
        p = np.array([self.predict_with_windows(t)[0] for t in texts])
        return p, np.ones(len(texts), dtype=int)


def synthetic_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    rows = []
    for label in y:
        r = {k: rng.normal() for k in SCALAR_FEATURES}
        r["perplexity"] = rng.normal(40 if label == 0 else 15, 6)
        r["top10_rank_fraction"] = rng.normal(0.5 if label == 0 else 0.7, 0.05)
        r["pos_bigram:DET_NOUN"] = rng.uniform(0, 0.2)
        rows.append(r)
    return pd.DataFrame(rows), y


@pytest.fixture(scope="session")
def stub_cascade():
    pre = Preprocessor(min_words=20, use_spacy=False)
    ex = StubExtractor(preprocessor=pre)
    texts_h = [" ".join(f"word{j}" for j in range(i, i + 60)) for i in range(40)]
    texts_a = [" ".join(["the cat sat on the mat"] * 10 + [f"x{i}"]) for i in range(40)]
    raws = [ex.extract(t) for t in texts_h + texts_a]
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    df = pd.DataFrame(raws)
    model, _ = fit_heuristic(df, y, df, y, df, y, head="lr", calibrator="platt")
    heu = HeuristicDetector(model, ex)
    det = CascadeDetector(pre, heu, StubTransformer(), ConfidenceGate(0.2, 0.8),
                          operating_points={"balanced": {"t_low": 0.2, "t_high": 0.8}, "thorough": {"t_low": 0.0, "t_high": 1.0}})
    return det
