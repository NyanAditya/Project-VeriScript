"""Heuristic feature extraction (Table 4.1, Fig. 5.3).

One GPT-2 small forward pass gives per-token log-probability, rank and entropy (features 1-9,
plus sentence perplexities for feature 11). The remaining features need only the text.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter

import numpy as np

from .lexicon import FUNCTION_WORDS
from .preprocess import Analysis, Preprocessor

PPL_CLIP = 1e4

# Scalar features in a fixed order. POS-bigram features are appended by the heuristic model,
# which selects the 20 most informative bigrams on the training split.
SCALAR_FEATURES: list[str] = [
    "mean_log_prob",               # 1
    "perplexity",                  # 2
    "std_log_prob",                # 3
    "min_sentence_perplexity",     # 4
    "max_sentence_perplexity",     # 5
    "top10_rank_fraction",         # 6
    "top100_rank_fraction",        # 7
    "tail_rank_fraction",          # 8
    "mean_entropy",                # 9
    "sentence_length_burstiness",  # 10
    "perplexity_burstiness",       # 11
    "type_token_ratio",            # 12
    "mean_word_length",            # 13
    "function_word_rate",          # 14
    "comma_rate",                  # 15 (punctuation profile, 4 values)
    "semicolon_rate",
    "colon_rate",
    "dash_rate",
]
POS_PREFIX = "pos_bigram:"

FEATURE_GROUPS = {
    "mean_log_prob": "perplexity",
    "perplexity": "perplexity",
    "std_log_prob": "perplexity",
    "min_sentence_perplexity": "rank (GLTR)",
    "max_sentence_perplexity": "rank (GLTR)",
    "top10_rank_fraction": "rank (GLTR)",
    "top100_rank_fraction": "rank (GLTR)",
    "tail_rank_fraction": "rank (GLTR)",
    "mean_entropy": "entropy",
    "sentence_length_burstiness": "burstiness",
    "perplexity_burstiness": "burstiness",
}


def feature_group(name: str) -> str:
    return FEATURE_GROUPS.get(name, "stylometric")


# ---------------------------------------------------------------------------------------------
# Text-only features (Listing 6.3)
# ---------------------------------------------------------------------------------------------
def burstiness(sentence_lengths) -> float:
    """B = (sigma - mu) / (sigma + mu), in [-1, 1]; -1 means perfectly regular lengths (Eq. 1.5)."""
    if len(sentence_lengths) == 0:
        return 0.0
    mu, sigma = float(np.mean(sentence_lengths)), float(np.std(sentence_lengths))
    return (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0


def type_token_ratio(word_list, prefix: int = 200) -> float:
    """TTR on a fixed-length prefix so that it is not a proxy for document length (Eq. 1.6)."""
    w = [x.lower() for x in word_list[:prefix]]
    return len(set(w)) / max(len(w), 1)


def coefficient_of_variation(values) -> float:
    if len(values) < 2:
        return 0.0
    mu = float(np.mean(values))
    return float(np.std(values)) / mu if mu > 0 else 0.0


def stylometric_features(analysis: Analysis) -> dict[str, float]:
    ws = analysis.words
    n_words = max(len(ws), 1)
    n_sents = max(len(analysis.sentences), 1)
    text = analysis.text
    dashes = text.count("—") + text.count("–") + text.count(" - ") + text.count("--")
    feats = {
        "sentence_length_burstiness": burstiness([len(s.words) for s in analysis.sentences if s.words]),
        "type_token_ratio": type_token_ratio(ws),
        "mean_word_length": float(np.mean([len(w) for w in ws])) if ws else 0.0,
        "function_word_rate": sum(1 for w in ws if w.lower() in FUNCTION_WORDS) / n_words,
        "comma_rate": text.count(",") / n_sents,
        "semicolon_rate": text.count(";") / n_sents,
        "colon_rate": text.count(":") / n_sents,
        "dash_rate": dashes / n_sents,
    }
    return feats


def pos_bigram_profile(analysis: Analysis) -> dict[str, float]:
    """Normalised counts of universal POS bigrams within sentences (all bigrams; selection is learned)."""
    counts: Counter[str] = Counter()
    for s in analysis.sentences:
        tags = [t for t in s.pos if t not in ("SPACE",)]
        counts.update(f"{a}_{b}" for a, b in zip(tags, tags[1:]))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {POS_PREFIX + k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------------------------------
# Reference-model features (Listing 6.2)
# ---------------------------------------------------------------------------------------------
class FeatureExtractor:
    """Computes the raw heuristic feature dictionary for a document."""

    def __init__(
        self,
        ref_lm: str = "gpt2",
        device: str = "cpu",
        max_tokens: int = 2048,
        min_sentence_tokens: int = 5,
        preprocessor: Preprocessor | None = None,
    ):
        self.ref_lm_name = ref_lm
        self.device = device
        self.max_tokens = max_tokens
        self.min_sentence_tokens = min_sentence_tokens
        self.pre = preprocessor or Preprocessor()
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.ref_lm_name)
            model = AutoModelForCausalLM.from_pretrained(self.ref_lm_name)
            model.eval().to(self.device)
            if self.device == "cuda":
                model = model.half()
            self._model = model
            self._ctx = int(getattr(model.config, "n_positions", 1024))
            torch.set_grad_enabled(False)
        return self._model, self._tok

    def token_stats(self, text: str):
        """Return (log_prob, rank, entropy, char_end) arrays for every scored token."""
        import torch
        import torch.nn.functional as F

        model, tok = self._load()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"][: self.max_tokens]
        offs = enc["offset_mapping"][: self.max_tokens]
        bos = tok.bos_token_id
        chunk = self._ctx - 1
        lps, ranks, ents = [], [], []
        with torch.no_grad():
            for i in range(0, len(ids), chunk):
                piece = [bos] + ids[i : i + chunk]
                x = torch.tensor([piece], device=self.device)
                logits = model(x).logits[0, :-1, :].float()  # position j predicts token j+1
                target = x[0, 1:]
                logp = F.log_softmax(logits, dim=-1)
                tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                rank = (logp > tok_logp.unsqueeze(-1)).sum(-1) + 1
                entropy = -(logp.exp() * logp).sum(-1)
                lps.append(tok_logp.cpu().numpy())
                ranks.append(rank.cpu().numpy())
                ents.append(entropy.cpu().numpy())
        if not lps:
            empty = np.zeros(0)
            return empty, empty, empty, empty
        ends = np.array([e for _, e in offs], dtype=np.int64)
        return np.concatenate(lps), np.concatenate(ranks), np.concatenate(ents), ends

    def lm_features(self, analysis: Analysis) -> dict[str, float]:
        logp, rank, ent, ends = self.token_stats(analysis.text)
        if len(logp) == 0:
            return {k: 0.0 for k in SCALAR_FEATURES[:9] + ["perplexity_burstiness"]}
        mean_lp = float(np.mean(logp))
        doc_ppl = float(min(math.exp(-mean_lp), PPL_CLIP))
        # Map tokens to sentences by character offset (token end - 1 lies inside its sentence).
        starts = [s.start for s in analysis.sentences]
        sent_lp: dict[int, list[float]] = {}
        for lp, e in zip(logp, ends):
            j = bisect.bisect_right(starts, int(e) - 1) - 1
            if j >= 0:
                sent_lp.setdefault(j, []).append(float(lp))
        sent_ppl = [
            min(math.exp(-float(np.mean(v))), PPL_CLIP)
            for v in sent_lp.values()
            if len(v) >= self.min_sentence_tokens
        ]
        if not sent_ppl:
            sent_ppl = [doc_ppl]
        return {
            "mean_log_prob": mean_lp,
            "perplexity": doc_ppl,
            "std_log_prob": float(np.std(logp)),
            "min_sentence_perplexity": float(min(sent_ppl)),
            "max_sentence_perplexity": float(max(sent_ppl)),
            "top10_rank_fraction": float(np.mean(rank <= 10)),
            "top100_rank_fraction": float(np.mean((rank > 10) & (rank <= 100))),
            "tail_rank_fraction": float(np.mean(rank > 1000)),
            "mean_entropy": float(np.mean(ent)),
            "perplexity_burstiness": coefficient_of_variation(sent_ppl),
        }

    def extract_analysis(self, analysis: Analysis) -> dict[str, float]:
        feats: dict[str, float] = {}
        feats.update(self.lm_features(analysis))
        feats.update(stylometric_features(analysis))
        feats.update(pos_bigram_profile(analysis))
        return feats

    def extract(self, text: str) -> dict[str, float]:
        """Raw feature dictionary for clean text (scalar features + full POS-bigram profile)."""
        return self.extract_analysis(self.pre.analyze(text))

    @staticmethod
    def names() -> list[str]:
        return list(SCALAR_FEATURES)
