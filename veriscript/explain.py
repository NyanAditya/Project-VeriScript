"""Explanation objects (Sec. 4.3.6): top contributing features with human / AI reference values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .features import POS_PREFIX, feature_group

DEFAULT_NOTE = "Probabilistic estimate; review before acting on it."


@dataclass
class FeatureContribution:
    name: str
    value: float
    contribution: float          # change in log-odds of "AI-generated"
    direction: str               # "towards_ai" | "towards_human"
    human_median: float
    ai_median: float
    group: str


@dataclass
class Explanation:
    top_features: list[FeatureContribution] = field(default_factory=list)
    note: str = DEFAULT_NOTE
    decided_by: str = "heuristic"            # "heuristic" | "transformer"
    heuristic_probability: float | None = None
    transformer_probability: float | None = None
    n_windows: int | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for f in d["top_features"]:
            for key in ("value", "contribution", "human_median", "ai_median"):
                f[key] = round(float(f[key]), 4)
        for key in ("heuristic_probability", "transformer_probability"):
            if d[key] is not None:
                d[key] = round(float(d[key]), 4)
        return d

    def render(self) -> str:
        lines = []
        if self.decided_by == "transformer":
            lines.append(
                f"Decided by the transformer model (p={self.transformer_probability:.3f}); "
                f"heuristic stage was uncertain (p={self.heuristic_probability:.3f})."
            )
        else:
            lines.append("Decided by the heuristic stage. Most influential features:")
        for f in self.top_features:
            arrow = "-> AI" if f.direction == "towards_ai" else "-> human"
            lines.append(
                f"  {pretty_name(f.name)} = {f.value:.3g}  ({arrow}; human median {f.human_median:.3g}, "
                f"AI median {f.ai_median:.3g})"
            )
        lines.append(self.note)
        return "\n".join(lines)


def pretty_name(name: str) -> str:
    if name.startswith(POS_PREFIX):
        return "POS bigram " + name[len(POS_PREFIX):].replace("_", " ")
    return name.replace("_", " ")


def top_features(model, x_raw: np.ndarray, contrib: np.ndarray, k: int = 5, note: str | None = None) -> Explanation:
    order = np.argsort(-np.abs(contrib))[:k]
    items = []
    for i in order:
        name = model.feature_names[i]
        c = float(contrib[i])
        items.append(
            FeatureContribution(
                name=name,
                value=float(x_raw[i]),
                contribution=c,
                direction="towards_ai" if c > 0 else "towards_human",
                human_median=float(model.human_median.get(name, float("nan"))),
                ai_median=float(model.ai_median.get(name, float("nan"))),
                group=feature_group(name),
            )
        )
    return Explanation(top_features=items, note=note or DEFAULT_NOTE)
