"""`veriscript` command line tool (Sec. 6.3.4, Listing 6.5)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="VeriScript: hybrid heuristic-transformer AI-text detector.")

SystemOpt = typer.Option("cascade", "--system", help="cascade | heuristic | transformer")
ModeOpt = typer.Option(None, "--mode", "-m", help="operating point: fast | balanced | thorough")
DeviceOpt = typer.Option(None, "--device", help="auto | cpu | cuda (default from config)")
ConfigOpt = typer.Option(None, "--config", "-c", help="path to a YAML config")


def _detector(config, system, mode, device, t_low=None, t_high=None):
    from .cascade import CascadeDetector
    from .config import load_config

    cfg = load_config(config)
    det = CascadeDetector.from_config(cfg, system=system, operating_point=mode, device=device)
    if t_low is not None and t_high is not None:
        det.set_thresholds(t_low, t_high)
    return det


def _load_docs(path: Path, text_col: str = "text", id_col: str = "id"):
    from .io import iter_folder

    if path.is_dir():
        return list(iter_folder(path))
    if path.suffix.lower() in (".csv", ".tsv"):
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with open(path, encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
        return [(r.get(id_col) or str(i), r[text_col], None) for i, r in enumerate(rows)]
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        ids = df[id_col] if id_col in df else df.index.astype(str)
        return [(str(i), t, None) for i, t in zip(ids, df[text_col])]
    raise typer.BadParameter("expected a folder, .csv, .tsv or .parquet file")


def write_report(results, out: Path, summary: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        e = r.explanation
        rows.append({
            "id": r.id, "label": r.label, "probability": "" if r.probability is None else round(r.probability, 4),
            "route": r.route or "", "n_words": r.n_words,
            "heuristic_probability": "" if not e or e.heuristic_probability is None else round(e.heuristic_probability, 4),
            "transformer_probability": "" if not e or e.transformer_probability is None else round(e.transformer_probability, 4),
            "latency_ms": round(r.latency_ms.get("total", 0.0), 2),
            "top_features": "; ".join(f"{f.name}={f.value:.3g}" for f in (e.top_features if e else [])),
            "note": r.note or "",
        })
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps({"summary": summary, "results": [r.to_json() for r in results]}, indent=2), encoding="utf-8")
    else:
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["id"])
            w.writeheader()
            w.writerows(rows)
        out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


@app.command()
def detect(
    path: Optional[Path] = typer.Argument(None, help=".txt, .pdf or .docx file (or use --text)"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="text to check"),
    as_json: bool = typer.Option(False, "--json", help="print the JSON result"),
    system: str = SystemOpt, mode: Optional[str] = ModeOpt, device: Optional[str] = DeviceOpt, config: Optional[Path] = ConfigOpt,
    t_low: Optional[float] = typer.Option(None, help="custom band lower limit"),
    t_high: Optional[float] = typer.Option(None, help="custom band upper limit"),
):
    """Check a single document."""
    from .io import ExtractionError, extract_text

    if text is None:
        if path is None:
            raise typer.BadParameter("give a file or --text")
        try:
            text = extract_text(path)
        except ExtractionError as exc:
            typer.secho(f"error: {exc}", fg="red", err=True)
            raise typer.Exit(2)
    det = _detector(config, system, mode, device, t_low, t_high)
    r = det.detect(text, doc_id=path.name if path else None)
    if as_json:
        typer.echo(json.dumps(r.to_json(), indent=2))
        return
    if r.probability is None:
        typer.secho(f"{r.label}: {r.note}", fg="yellow")
        return
    color = "red" if r.label == "ai_generated" else "green"
    typer.secho(f"{r.label}  p(AI) = {r.probability:.3f}", fg=color, bold=True)
    typer.echo(f"route: {r.route}   latency: {r.latency_ms['total']:.0f} ms "
               f"(heuristic {r.latency_ms['heuristic']:.0f}, transformer {r.latency_ms['transformer']:.0f})")
    typer.echo(r.explanation.render())


@app.command()
def batch(
    path: Path = typer.Argument(..., help="folder of files, or CSV/TSV/parquet with id,text columns"),
    out: Path = typer.Option(Path("report.csv"), "--out", "-o", help="report path (.csv or .json)"),
    text_col: str = typer.Option("text"), id_col: str = typer.Option("id"),
    system: str = SystemOpt, mode: Optional[str] = ModeOpt, device: Optional[str] = DeviceOpt, config: Optional[Path] = ConfigOpt,
):
    """Process a batch and write a CSV/JSON report with summary statistics."""
    from .cascade import summarize
    from .result import DetectionResult

    docs = _load_docs(path, text_col, id_col)
    det = _detector(config, system, mode, device)
    typer.echo(f"{len(docs)} documents; system={system} operating point={det.operating_point} "
               f"band=[{det.gate.t_low:.2f}, {det.gate.t_high:.2f}]")
    ok = [(i, t) for i, t, e in docs if e is None]
    with typer.progressbar(length=len(ok), label="heuristic stage") as bar:
        last = [0]

        def prog(k, n):
            bar.update(k - last[0])
            last[0] = k

        res = det.detect_batch(ok, progress=prog)
    res += [DetectionResult("rejected", None, None, {}, None, 0, i, note=e) for i, t, e in docs if e is not None]
    summary = summarize(res)
    summary.update({"system": system, "operating_point": det.operating_point, "t_low": det.gate.t_low, "t_high": det.gate.t_high})
    write_report(res, out, summary)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"report written to {out}")


@app.command()
def evaluate(
    data: Path = typer.Option(..., "--data", help="labelled CSV/parquet with text,label (1 = AI)"),
    systems: str = typer.Option("heuristic,transformer,cascade", "--systems"),
    mode: Optional[str] = ModeOpt, device: Optional[str] = DeviceOpt, config: Optional[Path] = ConfigOpt,
    out: Optional[Path] = typer.Option(None, "--out", help="write metrics JSON here"),
    limit: Optional[int] = typer.Option(None, help="evaluate on the first N documents"),
):
    """Accuracy and efficiency of each system on a labelled set (FR11)."""
    import numpy as np
    import pandas as pd

    from .evaluate import classification_metrics, latency_stats

    df = pd.read_parquet(data) if data.suffix == ".parquet" else pd.read_csv(data)
    if limit:
        df = df.iloc[:limit]
    report = {}
    for system in [s.strip() for s in systems.split(",") if s.strip()]:
        det = _detector(config, system, mode, device)
        res = [det.detect(t) for t in df.text.astype(str)]
        keep = np.array([r.probability is not None for r in res])
        p = np.array([r.probability if r.probability is not None else np.nan for r in res])
        m = classification_metrics(df.label.values[keep], p[keep], det.gate.tau)
        m.update({f"latency_{k}": v for k, v in latency_stats([r.latency_ms["total"] for r in res if r.probability is not None]).items()})
        m["escalation_rate"] = float(np.mean([r.route == "escalated" for r, k in zip(res, keep) if k]))
        m["n_undetermined"] = int((~keep).sum())
        report[system] = m
        typer.echo(f"{system:12s} F1={m['f1']:.4f} AUROC={m['auroc']:.4f} FPR={m['fpr']:.4f} "
                   f"mean={m['latency_mean_ms']:.1f}ms p95={m['latency_p95_ms']:.1f}ms escalation={m['escalation_rate']:.3f}")
    if out:
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


@app.command("retrain-heuristic")
def retrain_heuristic(
    data: Path = typer.Option(..., "--data", help="labelled CSV/parquet with text,label (1 = AI) [,split]"),
    out: Path = typer.Option(Path("models/heuristic/heuristic_retrained.joblib"), "--out"),
    head: str = typer.Option("auto", help="auto | lr | xgb"),
    device: Optional[str] = DeviceOpt, config: Optional[Path] = ConfigOpt,
):
    """Retrain and recalibrate the heuristic head on new labelled data (FR12).

    Without a `split` column the data are split 70/15/15 into train / cal / val at random.
    """
    import numpy as np
    import pandas as pd

    from .config import load_config
    from .features import FeatureExtractor
    from .heuristic import fit_heuristic
    from .preprocess import Preprocessor
    from .utils import get_device

    cfg = load_config(config)
    df = pd.read_parquet(data) if data.suffix == ".parquet" else pd.read_csv(data)
    if "split" not in df:
        r = np.random.default_rng(cfg.get("seed", 42)).random(len(df))
        df["split"] = np.where(r < 0.7, "train", np.where(r < 0.85, "cal", "val"))
    pre = Preprocessor(cfg["preprocess"]["min_words"])
    rl = cfg["reference_lm"]
    ex = FeatureExtractor(rl["name"], get_device(device or cfg["device"]), rl["max_tokens"], rl["min_sentence_tokens"], pre)
    texts = [pre.clean(t) for t in df.text.astype(str)]
    feats = []
    with typer.progressbar(texts, label="features") as it:
        for t in it:
            feats.append(ex.extract(t))
    F = pd.DataFrame(feats)
    F["split"], F["label"] = df.split.values, df.label.values
    parts = {s: F[F.split == s] for s in ("train", "val", "cal")}
    model, rep = fit_heuristic(parts["train"], parts["train"].label.values, parts["val"], parts["val"].label.values,
                               parts["cal"], parts["cal"].label.values, head=head)
    model.save(out)
    typer.echo(json.dumps({k: rep[k] for k in ("selected_head", "selected_calibrator", "val_ece", "heads")}, indent=2, default=str))
    typer.echo(f"saved {out}. Point heuristic.model_path in the config at it, then re-run `veriscript tune`.")


@app.command()
def tune(
    data: Path = typer.Option(..., "--data", help="labelled validation CSV/parquet with text,label [,native]"),
    eps: float = typer.Option(0.02, help="max F1 loss vs transformer-only for the 'balanced' point"),
    device: Optional[str] = DeviceOpt, config: Optional[Path] = ConfigOpt,
    write: bool = typer.Option(True, help="write the tuned operating points"),
):
    """Re-tune the ambiguity band and operating points on labelled validation data (Algorithm 4.3)."""
    import numpy as np
    import pandas as pd

    from .config import load_config, resolve_path
    from .gate import tune_band

    cfg = load_config(config)
    df = pd.read_parquet(data) if data.suffix == ".parquet" else pd.read_csv(data)
    det = _detector(config, "cascade", None, device)
    texts = [det.pre.clean(t) for t in df.text.astype(str)]
    p_h = np.array([det.heu.predict_proba(t) for t in texts])
    p_t, _ = det.trf.predict_proba_many(texts)
    res = {}
    for name, e in {"thorough": eps / 4, "balanced": eps, "fast": eps * 2.5}.items():
        res[name] = tune_band(p_h, p_t, df.label.values, eps=e)
        typer.echo(f"{name:9s} [{res[name]['t_low']:.2f}, {res[name]['t_high']:.2f}] escalation={res[name]['alpha']:.3f} "
                   f"F1={res[name]['f1_cascade']:.4f} (transformer {res[name]['f1_transformer']:.4f})")
    if write:
        path = resolve_path(cfg["operating_points_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"operating_points": {k: {"t_low": v["t_low"], "t_high": v["t_high"]} for k, v in res.items()},
                                    "tuning": res}, indent=2))
        typer.echo(f"wrote {path}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the REST service (FastAPI)."""
    import uvicorn

    uvicorn.run("app.api:app", host=host, port=port)


@app.command()
def demo(port: int = 8501):
    """Start the Streamlit web interface."""
    import subprocess

    from .config import PROJECT_ROOT

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py"),
                    "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
