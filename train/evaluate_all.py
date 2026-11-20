"""Step 8 / Objective O4: unified evaluation of heuristic-only, transformer-only and cascade.

    python -m train.evaluate_all [--latency-docs 200] [--devices cpu cuda] [--no-latency]

Accuracy, calibration, bootstrap CIs and paired tests, per-generator and per-domain breakdowns,
cross-generator generalisation (held-out family + heuristic LOGO), robustness (RAID attacks,
MAGE OOD), fairness (Delta-FPR), calibration ablation (RQ2), latency / throughput / footprint
(Sec. 6.1.5) and the Recovery / Speed-up figures of merit (Eq. 4.8).
Writes reports/results.json, reports/results.md and plots in reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from veriscript.config import load_config, resolve_path
from veriscript.evaluate import (bootstrap_ci, classification_metrics, fpr_disparity, latency_stats, model_footprint,
                                 paired_bootstrap, plot_calibration, plot_roc, plot_tradeoff, recovery_speedup)
from veriscript.gate import ConfidenceGate, evaluate_bands, tradeoff_curve

from .common import PREDICTIONS, PROCESSED, REPORTS, load_corpus, write_json

FIG = REPORTS / "figures"


def system_probs(df: pd.DataFrame, points: dict) -> dict[str, np.ndarray]:
    out = {"heuristic": df.p_h.values, "transformer": df.p_t.values}
    for name, pt in points.items():
        out[f"cascade_{name}"] = ConfidenceGate(pt["t_low"], pt["t_high"]).combine(df.p_h.values, df.p_t.values)
    return out


def escalation(df, points):
    return {name: float(ConfidenceGate(pt["t_low"], pt["t_high"]).escalate_mask(df.p_h.values).mean()) for name, pt in points.items()}


def split_metrics(df, points, tau=0.5):
    probs = system_probs(df, points)
    res = {k: classification_metrics(df.label.values, p, tau) for k, p in probs.items()}
    return res, probs


def per_group(df, points, col, tau=0.5):
    """F1 on (AI texts of the group + all human texts); recall on the group's AI texts."""
    humans = df[df.label == 0]
    out = {}
    for g, part in df[df.label == 1].groupby(col):
        sub = pd.concat([part, humans]) if col == "generator_family" else df[df[col] == g]
        probs = system_probs(sub, points)
        out[str(g)] = {k: {"f1": classification_metrics(sub.label.values, p, tau)["f1"],
                           "recall": float((p[sub.label.values == 1] >= tau).mean()),
                           "n_ai": int((sub.label == 1).sum())} for k, p in probs.items()}
    return out


# ---------------------------------------------------------------------------------------------
# Latency (Sec. 6.1.5)
# ---------------------------------------------------------------------------------------------
def measure_latency(texts: list[str], device: str, warmup: int = 20) -> dict:
    from veriscript.cascade import CascadeDetector

    cfg = load_config()
    out = {}
    for system, op in [("heuristic", None), ("transformer", None), ("cascade", "fast"), ("cascade", "balanced"), ("cascade", "thorough")]:
        det = CascadeDetector.from_config(cfg, system=system, operating_point=op or "balanced", device=device)
        for t in texts[:warmup]:
            det.detect(t)
        times, routes = [], []
        for t in texts[warmup:]:
            t0 = time.perf_counter()
            r = det.detect(t)
            times.append(1e3 * (time.perf_counter() - t0))
            routes.append(r.route)
        key = system if op is None else f"cascade_{op}"
        st = latency_stats(times)
        st["escalation_rate"] = float(np.mean([r == "escalated" for r in routes]))
        # batch throughput (batch sizes 1, 8, 32) via detect_batch
        bt = {}
        sample = texts[warmup:]
        for bs in (1, 8, 32):
            t0 = time.perf_counter()
            for i in range(0, len(sample), bs):
                det.detect_batch(sample[i : i + bs])
            bt[str(bs)] = len(sample) / (time.perf_counter() - t0)
        st["batch_throughput_docs_per_s"] = bt
        out[key] = st
        print(f"  [{device}] {key:18s} mean={st['mean_ms']:.1f}ms p95={st['p95_ms']:.1f}ms esc={st['escalation_rate']:.2f}", flush=True)
        del det
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------
def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def markdown_report(R: dict) -> str:
    L = ["# VeriScript evaluation results", ""]
    L.append(f"Operating points (tuned on `val`): " + ", ".join(
        f"**{k}** [{v['t_low']:.2f}, {v['t_high']:.2f}]" for k, v in R["operating_points"].items()))
    L.append("")
    for split, res in R["splits"].items():
        L.append(f"## Split `{split}` (n={res['metrics']['heuristic']['n']})")
        L.append("")
        L.append("| System | Accuracy | Precision | Recall | F1 | AUROC | FPR | ECE | Escalation |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for sys_, m in res["metrics"].items():
            esc = res["escalation"].get(sys_.replace("cascade_", ""), None) if sys_.startswith("cascade") else (1.0 if sys_ == "transformer" else 0.0)
            L.append(f"| {sys_} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                     f"{fmt(m['auroc'])} | {fmt(m['fpr'])} | {fmt(m['ece'])} | {fmt(esc, 3)} |")
        if "bootstrap" in res:
            L.append("")
            L.append("95% bootstrap CIs (1000 resamples): " + "; ".join(
                f"{k}: F1 [{v['f1_ci'][0]:.4f}, {v['f1_ci'][1]:.4f}], AUROC [{v['auroc_ci'][0]:.4f}, {v['auroc_ci'][1]:.4f}]"
                for k, v in res["bootstrap"].items()))
        if "paired" in res:
            L.append("")
            for k, v in res["paired"].items():
                L.append(f"- Paired bootstrap {k}: dF1 = {v['f1_diff']:+.4f} (95% CI [{v['f1_diff_ci'][0]:+.4f}, {v['f1_diff_ci'][1]:+.4f}], p = {v['p_value']:.3f})")
        L.append("")
    if R.get("fairness"):
        L.append("## Fairness (`fair_test`: human essays only)")
        L.append("")
        L.append("| System | FPR native | FPR non-native | Delta-FPR |")
        L.append("|---|---|---|---|")
        for k, v in R["fairness"].items():
            L.append(f"| {k} | {fmt(v['fpr_native'], 3)} | {fmt(v['fpr_non_native'], 3)} | {fmt(v['delta_fpr'], 3)} |")
        L.append("")
    if R.get("per_generator"):
        L.append("## Per generator family (test split; F1 on family AI + all humans)")
        L.append("")
        systems = list(next(iter(R["per_generator"].values())).keys())
        L.append("| Family | n AI | " + " | ".join(systems) + " |")
        L.append("|---|---|" + "---|" * len(systems))
        for fam, v in R["per_generator"].items():
            n = next(iter(v.values()))["n_ai"]
            L.append(f"| {fam} | {n} | " + " | ".join(fmt(v[s]['f1']) for s in systems) + " |")
        L.append("")
    if R.get("logo_heuristic"):
        L.append("## Leave-one-generator-out (heuristic stage retrained without each family)")
        L.append("")
        L.append("| Held-out family | F1 (LOGO) | F1 (trained with family) | Recall (LOGO) |")
        L.append("|---|---|---|---|")
        for fam, v in R["logo_heuristic"].items():
            L.append(f"| {fam} | {fmt(v['f1_logo'])} | {fmt(v['f1_in'])} | {fmt(v['recall_logo'])} |")
        L.append("")
    if R.get("calibration_ablation"):
        L.append("## Calibration ablation (RQ2, val split): cascade F1 at matched escalation rate")
        L.append("")
        L.append("| Escalation rate | Calibrated | Raw score |")
        L.append("|---|---|---|")
        for row in R["calibration_ablation"]:
            L.append(f"| {row['alpha']:.2f} | {fmt(row['f1_calibrated'])} | {fmt(row['f1_raw'])} |")
        L.append("")
    if R.get("latency"):
        L.append("## Latency (end to end from clean text, after 20 warm-up docs)")
        L.append("")
        L.append("| Device | System | Mean ms | p95 ms | Docs/s | Batch 8 docs/s | Batch 32 docs/s | Escalation |")
        L.append("|---|---|---|---|---|---|---|---|")
        for dev, d in R["latency"].items():
            for k, v in d.items():
                bt = v["batch_throughput_docs_per_s"]
                L.append(f"| {dev} | {k} | {v['mean_ms']:.1f} | {v['p95_ms']:.1f} | {v['throughput_docs_per_s']:.2f} | "
                         f"{bt['8']:.2f} | {bt['32']:.2f} | {v['escalation_rate']:.2f} |")
        L.append("")
    if R.get("figures_of_merit"):
        L.append("## Figures of merit (Eq. 4.8)")
        L.append("")
        for k, v in R["figures_of_merit"].items():
            L.append(f"- {k}: Recovery = {fmt(v['recovery'], 3)}, Speed-up = {fmt(v['speedup'], 2)}x")
        L.append("")
    if R.get("footprint"):
        L.append("## Model footprint")
        L.append("")
        for k, v in R["footprint"].items():
            L.append(f"- {k}: {v.get('size_mb')} MB on disk" + (f", {v['n_parameters'] / 1e6:.1f}M parameters" if v.get("n_parameters") else ""))
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-docs", type=int, default=200)
    ap.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    tau = cfg["gate"]["tau"]
    points = cfg["operating_points"]
    preds = pd.read_parquet(PREDICTIONS)
    R: dict = {"operating_points": points, "splits": {}}

    # -- accuracy on each split ------------------------------------------------------------------
    for split in ["test", "heldout_generator", "mage_ood", "mage_ood_para", "raid_attack", "val"]:
        df = preds[preds.split == split]
        if df.empty:
            continue
        metrics, probs = split_metrics(df, points, tau)
        entry = {"metrics": metrics, "escalation": escalation(df, points)}
        if split in ("test", "heldout_generator") and df.label.nunique() == 2:
            entry["bootstrap"] = {k: bootstrap_ci(df.label.values, p, tau, args.n_boot) for k, p in probs.items()}
            entry["paired"] = {
                "cascade_balanced - transformer": paired_bootstrap(df.label.values, probs["cascade_balanced"], probs["transformer"], tau, args.n_boot),
                "cascade_balanced - heuristic": paired_bootstrap(df.label.values, probs["cascade_balanced"], probs["heuristic"], tau, args.n_boot),
            }
        if split == "raid_attack":
            entry["per_attack_recall"] = {
                a: {k: float((p[(df.attack == a).values] >= tau).mean()) for k, p in probs.items()} for a in sorted(df.attack.unique())
            }
        R["splits"][split] = entry
        print(f"{split}: " + ", ".join(f"{k} F1={m['f1']:.4f}" for k, m in metrics.items()))

    test = preds[preds.split == "test"]
    R["per_generator"] = per_group(test, points, "generator_family", tau)
    R["per_domain"] = per_group(test, points, "domain", tau)
    probs = system_probs(test, points)
    plot_roc({k: (test.label.values, p) for k, p in probs.items() if k in ("heuristic", "transformer", "cascade_balanced")}, FIG / "roc_test.png")
    plot_calibration(test.label.values, {"heuristic (calibrated)": test.p_h.values, "heuristic (raw)": test.p_h_uncal.values,
                                         "transformer": test.p_t.values}, FIG / "calibration_test.png")

    # -- fairness ------------------------------------------------------------------------------
    fair = preds[preds.split == "fair_test"]
    if len(fair):
        R["fairness"] = {k: fpr_disparity(p, fair.native.astype(bool).values, tau) for k, p in system_probs(fair, points).items()}

    # -- calibration ablation (RQ2) on val -------------------------------------------------------
    val = preds[preds.split == "val"]
    g_cal = evaluate_bands(val.p_h.values, val.p_t.values, val.label.values, tau)
    g_raw = evaluate_bands(val.p_h_uncal.values, val.p_t.values, val.label.values, tau)
    abl = []
    for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        def best(g):
            m = g["alpha"] <= a + 1e-9
            return float(g["f1"][m].max()) if m.any() else float("nan")
        abl.append({"alpha": a, "f1_calibrated": best(g_cal), "f1_raw": best(g_raw)})
    R["calibration_ablation"] = abl

    # -- held-out generator LOGO for the heuristic stage -------------------------------------------
    logo = {}
    for path in sorted(PROCESSED.glob("predictions_no_*.parquet")):
        fam = path.stem.replace("predictions_no_", "")
        lp = pd.read_parquet(path)
        lt = lp[lp.split == "test"]
        sub = lt[(lt.label == 0) | (lt.generator_family == fam)]
        if (sub.label == 1).sum() == 0:
            continue
        ins = test[(test.label == 0) | (test.generator_family == fam)]
        logo[fam] = {
            "f1_logo": classification_metrics(sub.label.values, sub.p_h.values, tau)["f1"],
            "recall_logo": float((sub.p_h.values[sub.label.values == 1] >= tau).mean()),
            "f1_in": classification_metrics(ins.label.values, ins.p_h.values, tau)["f1"],
            "n_ai": int((sub.label == 1).sum()),
        }
        if lp.p_t.notna().any():
            for name, pt in points.items():
                pc = ConfidenceGate(pt["t_low"], pt["t_high"]).combine(sub.p_h.values, sub.p_t.values)
                logo[fam][f"f1_cascade_{name}"] = classification_metrics(sub.label.values, pc, tau)["f1"]
            logo[fam]["f1_transformer"] = classification_metrics(sub.label.values, sub.p_t.values, tau)["f1"]
    R["logo_heuristic"] = logo

    # -- footprint -----------------------------------------------------------------------------
    trf_dir = resolve_path(cfg["transformer"]["model_dir"])
    R["footprint"] = {
        "heuristic head": model_footprint(resolve_path(cfg["heuristic"]["model_path"])),
        "reference LM (gpt2)": {"size_mb": 548.0, "n_parameters": 124_439_808},
        "transformer": model_footprint(trf_dir),
    }
    info = trf_dir / "veriscript_training.json"
    if info.exists():
        R["footprint"]["transformer"]["n_parameters"] = json.loads(info.read_text())["n_parameters"]

    # -- latency ---------------------------------------------------------------------------------
    if not args.no_latency:
        corpus = load_corpus()
        texts = corpus[corpus.split == "test"].sample(args.latency_docs + 20, random_state=0).text.tolist()
        R["latency"] = {}
        for dev in args.devices:
            try:
                R["latency"][dev] = measure_latency(texts, dev)
            except RuntimeError as exc:
                print(f"  skipping {dev}: {exc}")
        fom = {}
        for dev, d in R["latency"].items():
            for op in points:
                k = f"cascade_{op}"
                fom[f"{dev} {op}"] = recovery_speedup(
                    R["splits"]["test"]["metrics"][k]["f1"], R["splits"]["test"]["metrics"]["heuristic"]["f1"],
                    R["splits"]["test"]["metrics"]["transformer"]["f1"], d[k]["mean_ms"], d["transformer"]["mean_ms"])
        R["figures_of_merit"] = fom
        # trade-off curve with measured stage costs (CPU if available)
        d0 = R["latency"].get("cpu") or next(iter(R["latency"].values()))
        c_h = d0["heuristic"]["mean_ms"]
        c_t = d0["transformer"]["mean_ms"]
        curve = tradeoff_curve(val.p_h.values, val.p_t.values, val.label.values, c_h, c_t, tau=tau)
        R["tradeoff_curve"] = curve
        tf1 = R["splits"]["val"]["metrics"]
        chosen = {}
        for op, pt in points.items():
            a = float(ConfidenceGate(pt["t_low"], pt["t_high"]).escalate_mask(val.p_h.values).mean())
            chosen[op] = ((c_h + a * c_t) / (c_h + c_t), tf1[f"cascade_{op}"]["f1"])
        plot_tradeoff(curve, {"heuristic only": (c_h / (c_h + c_t), tf1["heuristic"]["f1"]), "transformer only": (1.0, tf1["transformer"]["f1"])},
                      FIG / "tradeoff_val.png", chosen)

    write_json(R, REPORTS / "results.json")
    (REPORTS / "results.md").write_text(markdown_report(R), encoding="utf-8")
    print(f"wrote {REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
