"""Configuration loading (YAML) with project-root-relative path resolution."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("VERISCRIPT_HOME", PACKAGE_DIR.parent)).resolve()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def resolve_path(p: str | os.PathLike) -> Path:
    """Resolve a config path against the project root unless it is absolute."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config, merge tuned operating points and optional overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    tuned = resolve_path(cfg.get("operating_points_path", "models/operating_points.json"))
    if tuned.exists():
        with open(tuned, encoding="utf-8") as fh:
            data = json.load(fh)
        points = data.get("operating_points", data)
        cfg.setdefault("operating_points", {})
        for name, pt in points.items():
            cfg["operating_points"][name] = {"t_low": float(pt["t_low"]), "t_high": float(pt["t_high"])}
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    return cfg
