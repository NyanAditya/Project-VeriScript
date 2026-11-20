"""Small shared helpers: device selection, seeding, timing."""

from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np


def get_device(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if pref == "cuda" or pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if pref == "cuda":
            raise RuntimeError("CUDA requested but not available")
    return "cpu"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


class Timer:
    """Accumulating wall-clock timer in milliseconds."""

    def __init__(self) -> None:
        self.ms = 0.0

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.ms += 1e3 * (time.perf_counter() - t0)
