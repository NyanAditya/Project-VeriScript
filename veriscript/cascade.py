"""CascadeDetector facade (Algorithm 4.2, Listing 6.4, Fig. 5.4)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import load_config, resolve_path
from .explain import DEFAULT_NOTE, Explanation
from .features import FeatureExtractor
from .gate import ESCALATED, FAST_PATH, ConfidenceGate
from .heuristic import HeuristicDetector, HeuristicModel
from .preprocess import Preprocessor, word_count
from .result import AI, HUMAN, UNDETERMINED, DetectionResult
from .transformer import TransformerDetector
from .utils import get_device

SYSTEMS = ("cascade", "heuristic", "transformer")


class CascadeDetector:
    """Runs the heuristic stage on every document and escalates only those inside the band.

    ``system`` selects one of the three evaluated systems:
      * ``cascade``     - confidence-gated routing (default)
      * ``heuristic``   - heuristic-only baseline (never escalates)
      * ``transformer`` - transformer-only baseline (always escalates, heuristic stage skipped)
    """

    def __init__(
        self,
        pre: Preprocessor,
        heu: HeuristicDetector | None,
        trf: TransformerDetector | None,
        gate: ConfidenceGate,
        system: str = "cascade",
        top_k: int = 5,
        note: str = DEFAULT_NOTE,
        operating_points: dict | None = None,
        log_path: str | Path | None = None,
    ):
        if system not in SYSTEMS:
            raise ValueError(f"system must be one of {SYSTEMS}")
        self.pre, self.heu, self.trf, self.gate = pre, heu, trf, gate
        self.system = system
        self.top_k = top_k
        self.note = note
        self.operating_points = operating_points or {}
        self.operating_point = "custom"
        self.log_path = Path(log_path) if log_path else None

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict | str | None = None, system: str = "cascade", operating_point: str | None = None, device: str | None = None):
        cfg = cfg if isinstance(cfg, dict) else load_config(cfg)
        dev = get_device(device or cfg.get("device", "auto"))
        pre = Preprocessor(min_words=cfg["preprocess"]["min_words"])
        heu = None
        if system != "transformer":
            rl = cfg["reference_lm"]
            extractor = FeatureExtractor(rl["name"], dev, rl["max_tokens"], rl["min_sentence_tokens"], pre)
            path = resolve_path(cfg["heuristic"]["model_path"])
            if not path.exists():
                raise FileNotFoundError(f"No heuristic model at {path}. Run `python -m train.train_heuristic` first.")
            heu = HeuristicDetector(HeuristicModel.load(path), extractor)
        trf = None
        tc = cfg["transformer"]
        if system != "heuristic":
            trf = TransformerDetector(resolve_path(tc["model_dir"]), tc["max_len"], tc["stride"], dev, tc["batch_size"])
        g = cfg["gate"]
        det = cls(
            pre,
            heu,
            trf,
            ConfidenceGate(g.get("t_low", 0.1), g.get("t_high", 0.9), g.get("tau", 0.5)),
            system=system,
            top_k=cfg.get("explain", {}).get("top_k", 5),
            note=cfg.get("explain", {}).get("note", DEFAULT_NOTE),
            operating_points=cfg.get("operating_points", {}),
            log_path=resolve_path(cfg["logging"]["path"]) if cfg.get("logging", {}).get("enabled") else None,
        )
        op = operating_point or g.get("operating_point", "balanced")
        if op != "custom":
            det.set_operating_point(op)
        return det

    def set_operating_point(self, name: str) -> None:
        if name not in self.operating_points:
            raise KeyError(f"unknown operating point '{name}'; available: {sorted(self.operating_points)}")
        pt = self.operating_points[name]
        self.gate = ConfidenceGate(pt["t_low"], pt["t_high"], self.gate.tau)
        self.operating_point = name

    def set_thresholds(self, t_low: float, t_high: float, tau: float | None = None) -> None:
        self.gate = ConfidenceGate(t_low, t_high, self.gate.tau if tau is None else tau)
        self.operating_point = "custom"

    def config_json(self) -> dict:
        return {
            "system": self.system,
            "operating_point": self.operating_point,
            "t_low": self.gate.t_low,
            "t_high": self.gate.t_high,
            "tau": self.gate.tau,
            "operating_points": self.operating_points,
            "min_words": self.pre.min_words,
        }

    # -- helpers ---------------------------------------------------------------------------
    def _label(self, p: float) -> str:
        return AI if p >= self.gate.tau else HUMAN

    def _escalate(self, p_h: float | None) -> bool:
        if self.system == "transformer":
            return True
        if self.system == "heuristic":
            return False
        return self.gate.route(p_h) == ESCALATED

    def _insufficient(self, n_words: int, doc_id=None, ms=0.0) -> DetectionResult:
        return DetectionResult(
            UNDETERMINED, None, None, {"heuristic": 0.0, "transformer": 0.0, "total": ms}, None, n_words, doc_id,
            note=f"insufficient text (fewer than {self.pre.min_words} words)",
        )

    def _log(self, res: DetectionResult) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res.to_json()) + "\n")

    # -- single document (Algorithm 4.2) ------------------------------------------------------
    def detect(self, text: str, doc_id: str | None = None) -> DetectionResult:
        t_start = time.perf_counter()
        text = self.pre.clean(text)
        n_words = word_count(text)
        if n_words < self.pre.min_words:
            return self._insufficient(n_words, doc_id, 1e3 * (time.perf_counter() - t_start))

        t_h = t_t = 0.0
        p_h, raw = None, None
        if self.system != "transformer":
            t0 = time.perf_counter()
            raw = self.heu.features(text)
            p_h = self.heu.predict_proba(raw)
            t_h = time.perf_counter() - t0

        if self._escalate(p_h):
            t1 = time.perf_counter()
            p, n_win = self.trf.predict_with_windows(text)
            t_t = time.perf_counter() - t1
            route = ESCALATED
            expl = self.heu.explain(raw, self.top_k, self.note) if raw is not None else Explanation(note=self.note)
            expl.decided_by = "transformer"
            expl.transformer_probability = p
            expl.heuristic_probability = p_h
            expl.n_windows = n_win
        else:
            p, route = p_h, FAST_PATH
            expl = self.heu.explain(raw, self.top_k, self.note)
            expl.heuristic_probability = p_h

        res = DetectionResult(
            label=self._label(p),
            probability=float(p),
            route=route,
            latency_ms={"heuristic": 1e3 * t_h, "transformer": 1e3 * t_t, "total": 1e3 * (time.perf_counter() - t_start)},
            explanation=expl,
            n_words=n_words,
            id=doc_id,
        )
        self._log(res)
        return res

    # -- batch (Fig. 5.7): heuristic for all, then mini-batched transformer on the queue ------------
    def detect_batch(self, docs: Iterable[str | tuple[str, str]], progress=None) -> list[DetectionResult]:
        items = [(d if isinstance(d, tuple) else (str(i), d)) for i, d in enumerate(docs)]
        results: list[DetectionResult | None] = [None] * len(items)
        queue: list[int] = []
        cleaned: dict[int, str] = {}
        raws: dict[int, dict] = {}
        p_hs: dict[int, float] = {}
        t_h: dict[int, float] = {}
        for k, (doc_id, text) in enumerate(items):
            t0 = time.perf_counter()
            if text is None:
                results[k] = DetectionResult("rejected", None, None, {}, None, 0, doc_id, note="text could not be extracted")
                continue
            text = self.pre.clean(text)
            n = word_count(text)
            if n < self.pre.min_words:
                results[k] = self._insufficient(n, doc_id)
                continue
            cleaned[k] = text
            if self.system != "transformer":
                raws[k] = self.heu.features(text)
                p_hs[k] = self.heu.predict_proba(raws[k])
            t_h[k] = 1e3 * (time.perf_counter() - t0)
            if self._escalate(p_hs.get(k)):
                queue.append(k)
            else:
                expl = self.heu.explain(raws[k], self.top_k, self.note)
                expl.heuristic_probability = p_hs[k]
                results[k] = DetectionResult(
                    self._label(p_hs[k]), p_hs[k], FAST_PATH, {"heuristic": t_h[k], "transformer": 0.0, "total": t_h[k]},
                    expl, n, doc_id,
                )
            if progress:
                progress(k + 1, len(items))
        if queue:
            t1 = time.perf_counter()
            p_t, n_win = self.trf.predict_proba_many([cleaned[k] for k in queue])
            per_doc = 1e3 * (time.perf_counter() - t1) / len(queue)
            for idx, k in enumerate(queue):
                raw = raws.get(k)
                expl = self.heu.explain(raw, self.top_k, self.note) if raw is not None else Explanation(note=self.note)
                expl.decided_by = "transformer"
                expl.transformer_probability = float(p_t[idx])
                expl.heuristic_probability = p_hs.get(k)
                expl.n_windows = int(n_win[idx])
                results[k] = DetectionResult(
                    self._label(p_t[idx]), float(p_t[idx]), ESCALATED,
                    {"heuristic": t_h[k], "transformer": per_doc, "total": t_h[k] + per_doc},
                    expl, word_count(cleaned[k]), items[k][0],
                )
        for r in results:
            self._log(r)
        return results  # type: ignore[return-value]


def summarize(results: list[DetectionResult]) -> dict:
    """Batch summary statistics (FR10)."""
    decided = [r for r in results if r.probability is not None]
    esc = [r for r in decided if r.route == ESCALATED]
    totals = [r.latency_ms.get("total", 0.0) for r in decided]
    return {
        "n_documents": len(results),
        "n_decided": len(decided),
        "n_ai_generated": sum(r.label == AI for r in decided),
        "n_human_written": sum(r.label == HUMAN for r in decided),
        "n_undetermined": sum(r.label == UNDETERMINED for r in results),
        "n_rejected": sum(r.label == "rejected" for r in results),
        "escalation_rate": len(esc) / len(decided) if decided else 0.0,
        "mean_latency_ms": float(np.mean(totals)) if totals else 0.0,
        "p95_latency_ms": float(np.percentile(totals, 95)) if totals else 0.0,
        "throughput_docs_per_s": (len(decided) / (sum(totals) / 1e3)) if totals and sum(totals) > 0 else 0.0,
    }
