"""Unit tests against hand-computed values (Sec. 6.3.5)."""

import numpy as np
import pytest

from veriscript.evaluate import bootstrap_ci, classification_metrics, fpr_disparity, recovery_speedup
from veriscript.features import burstiness, coefficient_of_variation, stylometric_features, type_token_ratio
from veriscript.gate import ESCALATED, FAST_PATH, ConfidenceGate, evaluate_bands, tune_band
from veriscript.heuristic import expected_calibration_error
from veriscript.preprocess import Preprocessor, normalise, remove_boilerplate, text_hash, word_count


# -- features -----------------------------------------------------------------------------
def test_burstiness_equal_lengths_is_minus_one():
    assert burstiness([12, 12, 12, 12]) == -1.0


def test_burstiness_known_value():
    # mu = 10, sigma = 5 -> (5 - 10) / (5 + 10) = -1/3
    assert burstiness([5, 15]) == pytest.approx(-1 / 3)


def test_burstiness_empty():
    assert burstiness([]) == 0.0


def test_ttr_uses_prefix_only():
    words = [f"w{i}" for i in range(200)] + ["w0"] * 1000
    assert type_token_ratio(words, prefix=200) == 1.0
    assert type_token_ratio(["a", "A", "b", "b"]) == 0.5


def test_coefficient_of_variation():
    assert coefficient_of_variation([10.0]) == 0.0
    assert coefficient_of_variation([5.0, 15.0]) == pytest.approx(0.5)


def test_stylometric_punctuation_rates():
    pre = Preprocessor(use_spacy=False)
    a = pre.analyze("Hello, world; this is one. Second sentence: here, it is.")
    f = stylometric_features(a)
    assert f["comma_rate"] == pytest.approx(1.0)
    assert f["semicolon_rate"] == pytest.approx(0.5)
    assert f["colon_rate"] == pytest.approx(0.5)


# -- pre-processing -----------------------------------------------------------------------
def test_normalise_strips_markup_and_whitespace():
    t = normalise("## Title\n\n<b>Bold</b>   **text** with [a link](http://x.y)\r\n\n\n\nEnd！")
    assert "<b>" not in t and "**" not in t and "http" not in t
    assert "a link" in t
    assert "\n\n\n" not in t
    assert t.endswith("End!")  # NFKC folds full-width punctuation


def test_boilerplate_removed():
    t = remove_boilerplate("As an AI language model, I cannot do that. The sky is blue. I hope this helps!")
    assert t == "The sky is blue."


def test_length_check_and_hash():
    pre = Preprocessor(min_words=5, use_spacy=False)
    assert not pre.is_sufficient("one two three four")
    assert pre.is_sufficient("one two three four five")
    assert word_count("It's a dog's life.") == 4
    assert text_hash("Hello,  World!") == text_hash("hello world")


def test_regex_sentence_split():
    pre = Preprocessor(use_spacy=False)
    assert pre.sentences("First one. Second one! Third? yes") == ["First one.", "Second one!", "Third? yes"]


# -- gate ---------------------------------------------------------------------------------
def test_gate_band_edges_inclusive():
    g = ConfidenceGate(0.2, 0.8)
    assert g.route(0.2) == ESCALATED
    assert g.route(0.8) == ESCALATED
    assert g.route(0.5) == ESCALATED
    assert g.route(0.1999) == FAST_PATH
    assert g.route(0.8001) == FAST_PATH


def test_gate_rejects_bad_band():
    with pytest.raises(ValueError):
        ConfidenceGate(0.7, 0.3)


def test_gate_limiting_cases():
    p_h = np.array([0.0, 0.3, 0.7, 1.0])
    assert ConfidenceGate(0.0, 1.0).escalate_mask(p_h).all()  # transformer-only
    assert ConfidenceGate(0.5, 0.5).escalate_mask(p_h).sum() == 0  # heuristic-only


def test_tune_band_picks_cheapest_feasible():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p_t = np.where(y == 1, 0.95, 0.05)  # perfect transformer
    # heuristic is confident & right at the extremes, but random in the middle region
    p_h = np.where(y == 1, rng.uniform(0.3, 1.0, 1000), rng.uniform(0.0, 0.7, 1000))
    res = tune_band(p_h, p_t, y, eps=0.0)
    assert res["feasible"]
    assert res["f1_cascade"] == pytest.approx(1.0)
    # every doc in [0.3, 0.7] must be escalated, nothing more needed
    assert res["t_low"] <= 0.3 and res["t_high"] >= 0.7
    assert res["alpha"] < 1.0


def test_evaluate_bands_shapes_and_fairness():
    y = np.array([0, 1, 0, 1])
    grid = evaluate_bands([0.1, 0.9, 0.4, 0.6], [0.0, 1.0, 0.0, 1.0], y, step=0.25,
                          fair_p_h=[0.6, 0.2], fair_p_t=[0.0, 0.0], fair_native=[False, True])
    assert grid["f1"].shape == (3, 3)
    # band (0.5, 0.75) escalates the non-native essay (p_h = 0.6) -> no false positive
    i, j = 2, 1
    assert grid["t_low"][i, j] == 0.5 and grid["t_high"][i, j] == 0.75
    assert grid["delta_fpr"][i, j] == 0.0


# -- metrics ------------------------------------------------------------------------------
def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.5
    assert m["auroc"] == 1.0


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 400), 0, 1)
    ci = bootstrap_ci(y, p, n_boot=200)
    f1 = classification_metrics(y, p)["f1"]
    assert ci["f1_ci"][0] <= f1 <= ci["f1_ci"][1]


def test_fpr_disparity_and_recovery():
    d = fpr_disparity([0.9, 0.1, 0.8, 0.7], [True, True, False, False])
    assert d["fpr_native"] == 0.5 and d["fpr_non_native"] == 1.0 and d["delta_fpr"] == 0.5
    r = recovery_speedup(f1_c=0.95, f1_h=0.85, f1_t=0.97, t_c=50, t_t=200)
    assert r["recovery"] == pytest.approx(10 / 12)
    assert r["speedup"] == 4.0
