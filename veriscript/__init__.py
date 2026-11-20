"""VeriScript: a hybrid heuristic-transformer cascade for AI-generated text detection."""

__version__ = "0.1.0"

from .gate import ConfidenceGate  # noqa: E402
from .result import DetectionResult  # noqa: E402


def load_detector(config=None, system: str = "cascade", operating_point: str | None = None, device: str | None = None):
    """Convenience constructor for :class:`veriscript.cascade.CascadeDetector`."""
    from .cascade import CascadeDetector

    return CascadeDetector.from_config(config, system=system, operating_point=operating_point, device=device)


__all__ = ["ConfidenceGate", "DetectionResult", "load_detector", "__version__"]
