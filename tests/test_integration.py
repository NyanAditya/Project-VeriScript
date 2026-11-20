"""Integration tests: full cascade on fixed documents with stub models (deterministic)."""

import numpy as np
import pytest

from tests.conftest import synthetic_frame
from veriscript.cascade import summarize
from veriscript.gate import ESCALATED, FAST_PATH
from veriscript.heuristic import HeuristicModel, fit_heuristic

HUMAN_TEXT = " ".join(f"token{j}" for j in range(80))
AI_TEXT = " ".join(["the cat sat on the mat"] * 12)


def test_fit_heuristic_selects_head_and_calibrator():
    df, y = synthetic_frame()
    model, rep = fit_heuristic(df, y, df, y, df, y, head="auto")
    assert rep["selected_head"] in ("lr", "xgb")
    assert rep["selected_calibrator"] in ("platt", "isotonic")
    assert len(model.feature_names) == 18 + len(rep["pos_bigrams"])
    p, _ = model.predict_proba_matrix(model.vectorize(df))
    assert ((p >= 0.5) == y).mean() > 0.9


def test_model_roundtrip(tmp_path):
    df, y = synthetic_frame()
    model, _ = fit_heuristic(df, y, df, y, df, y, head="xgb")
    model.save(tmp_path / "h.joblib")
    m2 = HeuristicModel.load(tmp_path / "h.joblib")
    X = model.vectorize(df)
    np.testing.assert_allclose(model.predict_proba_matrix(X)[0], m2.predict_proba_matrix(X)[0])
    assert len(m2.contributions(m2.standardize(X[0]))) == len(m2.feature_names)


def test_insufficient_text(stub_cascade):
    r = stub_cascade.detect("too short")
    assert r.label == "undetermined" and r.probability is None
    assert "insufficient" in r.note


def test_fast_path_has_feature_explanation(stub_cascade):
    stub_cascade.set_operating_point("balanced")
    r = stub_cascade.detect(AI_TEXT)
    assert r.route == FAST_PATH
    assert r.label == "ai_generated"
    assert len(r.explanation.top_features) == 5
    js = r.to_json()
    assert js["latency_ms"]["transformer"] == 0.0
    assert "review" in js["explanation"]["note"]


def test_thorough_escalates_everything(stub_cascade):
    stub_cascade.set_operating_point("thorough")
    r = stub_cascade.detect(HUMAN_TEXT + " delve")
    assert r.route == ESCALATED
    assert r.explanation.decided_by == "transformer"
    assert r.probability == pytest.approx(0.97)
    stub_cascade.set_operating_point("balanced")


def test_deterministic(stub_cascade):
    a = stub_cascade.detect(HUMAN_TEXT).probability
    b = stub_cascade.detect(HUMAN_TEXT).probability
    assert a == b


def test_batch_matches_single(stub_cascade):
    stub_cascade.set_operating_point("thorough")
    docs = [("h", HUMAN_TEXT), ("a", AI_TEXT + " delve"), ("s", "short")]
    res = stub_cascade.detect_batch(docs)
    assert [r.id for r in res] == ["h", "a", "s"]
    assert res[1].label == "ai_generated" and res[0].label == "human_written"
    assert res[2].label == "undetermined"
    s = summarize(res)
    assert s["n_decided"] == 2 and s["escalation_rate"] == 1.0
    stub_cascade.set_operating_point("balanced")
