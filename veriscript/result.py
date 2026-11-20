"""DetectionResult returned by every detection call (Table 3.2, Listing 5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .explain import Explanation

AI = "ai_generated"
HUMAN = "human_written"
UNDETERMINED = "undetermined"
REJECTED = "rejected"


@dataclass
class DetectionResult:
    label: str
    probability: float | None
    route: str | None
    latency_ms: dict[str, float] = field(default_factory=dict)
    explanation: Explanation | None = None
    n_words: int = 0
    id: str | None = None
    note: str | None = None

    @property
    def label_int(self) -> int | None:
        return {AI: 1, HUMAN: 0}.get(self.label)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.id is not None:
            out["id"] = self.id
        out.update(
            {
                "label": self.label,
                "probability": None if self.probability is None else round(float(self.probability), 4),
                "route": self.route,
                "latency_ms": {k: round(float(v), 2) for k, v in self.latency_ms.items()},
                "n_words": self.n_words,
                "explanation": self.explanation.to_json() if self.explanation else None,
            }
        )
        if self.note:
            out["note"] = self.note
        return out
