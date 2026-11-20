"""Slow test: real GPT-2 + spaCy feature extraction."""

import pytest

from veriscript.features import SCALAR_FEATURES, FeatureExtractor

TEXT = (
    "The committee met on Tuesday to discuss the budget. Nobody expected the argument that followed, "
    "least of all the chair, who had spent the weekend at her sister's farm and arrived with mud on her boots. "
    "By noon they had agreed on nothing. Coffee ran out. Someone suggested pizza; someone else said that "
    "pizza was how the trouble started last year. In the end the vote was postponed until March."
)


@pytest.mark.slow
def test_real_feature_vector():
    ex = FeatureExtractor("gpt2", device="cpu")
    f = ex.extract(TEXT)
    for k in SCALAR_FEATURES:
        assert k in f, k
    assert 1.0 < f["perplexity"] < 1e4
    assert f["min_sentence_perplexity"] <= f["max_sentence_perplexity"]
    assert 0 <= f["top10_rank_fraction"] <= 1
    assert abs(f["top10_rank_fraction"] + f["top100_rank_fraction"] + f["tail_rank_fraction"]) <= 1.0001
    assert any(k.startswith("pos_bigram:") for k in f)
    assert -1 <= f["sentence_length_burstiness"] <= 1
