"""Black-and-white figures and extra statistics for the project report (Chapter 7).

    python -m train.report_figures [--out Project-Report/images/results]

Reads data/processed/predictions*.parquet, reports/results.json and the trained heuristic
model; writes PNG figures and reports/report_extras.json. No model inference is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from veriscript.config import PROJECT_ROOT, load_config, resolve_path
from veriscript.evaluate import classification_metrics
from veriscript.gate import ConfidenceGate
from veriscript.heuristic import HeuristicModel

from .common import PREDICTIONS, PROCESSED, REPORTS, write_json

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6, "savefig.dpi": 220, "savefig.bbox": "tight",
})
SYS = {"heuristic": "Heuristic only", "transformer": "Transformer only", "cascade_fast": "Cascade (fast)",
       "cascade_balanced": "Cascade (balanced)", "cascade_thorough": "Cascade (thorough)"}
HATCH = {"heuristic": "", "transformer": "////", "cascade_fast": "....", "cascade_balanced": "xxx", "cascade_thorough": "\\\\\\\\"}
GREY = {"heuristic": "0.85", "transformer": "0.35", "cascade_fast": "0.95", "cascade_balanced": "0.7", "cascade_thorough": "0.55"}


def probs(df, points):
    out = {"heuristic": df.p_h.values, "transformer": df.p_t.values}
    for k, pt in points.items():
        out[f"cascade_{k}"] = ConfidenceGate(pt["t_low"], pt["t_high"]).combine(df.p_h.values, df.p_t.values)
    return out


def bars(ax, groups, systems, values, ylabel, ylim=None, fmt="{:.2f}"):
    x = np.arange(len(groups))
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        v = [values[g][s] for g in groups]
        b = ax.bar(x + (i - (len(systems) - 1) / 2) * w, v, w, label=SYS.get(s, s), color=GREY.get(s, "0.6"),
                   edgecolor="black", hatch=HATCH.get(s, ""), linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "Project-Report" / "images" / "results"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    points = cfg["operating_points"]
    R = json.loads((REPORTS / "results.json").read_text())
    P = pd.read_parquet(PREDICTIONS)
    test, val = P[P.split == "test"], P[P.split == "val"]
    extras: dict = {}

    # 1. ROC curves (test) -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pr = probs(test, points)
    for s, ls in [("heuristic", "--"), ("transformer", "-"), ("cascade_thorough", ":")]:
        fpr, tpr, _ = roc_curve(test.label, pr[s])
        ax.plot(fpr, tpr, ls, color="black", lw=1.4, label=f"{SYS[s]} (AUROC {R['splits']['test']['metrics'][s]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8.5)
    inset = ax.inset_axes([0.42, 0.32, 0.45, 0.4])
    for s, ls in [("heuristic", "--"), ("transformer", "-"), ("cascade_thorough", ":")]:
        fpr, tpr, _ = roc_curve(test.label, pr[s])
        inset.plot(fpr, tpr, ls, color="black", lw=1.2)
    inset.set_xlim(0, 0.15)
    inset.set_ylim(0.8, 1.0)
    inset.tick_params(labelsize=7)
    fig.savefig(out / "roc_test.png")
    plt.close(fig)

    # 2. Trade-off curve (val) ----------------------------------------------------------------
    c = R["stage_costs_ms"]
    tot = c["heuristic"] + c["transformer"]
    curve = R["tradeoff_curve"]
    vm = R["splits"]["val"]["metrics"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    xs = [p["relative_cost"] for p in curve]
    ax.plot(xs, [p["f1"] for p in curve], "-o", color="black", ms=3, lw=1.2, label="Cascade operating points")
    ax.scatter([c["heuristic"] / tot], [vm["heuristic"]["f1"]], marker="s", s=60, facecolor="white", edgecolor="black", zorder=3, label="Heuristic only")
    ax.scatter([c["transformer"] / tot], [vm["transformer"]["f1"]], marker="^", s=70, color="black", zorder=3, label="Transformer only")
    for op, (x, y) in R["tradeoff_chosen"].items():
        ax.scatter([x], [y], marker="x", color="black", s=50, zorder=4)
        ax.annotate(op, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8.5)
    ax.set_xlabel(r"Relative expected cost  $\mathbb{E}[C]/(C_h+C_t)$ (CPU)")
    ax.set_ylabel("F1 (validation)")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.savefig(out / "tradeoff_val.png")
    plt.close(fig)

    # 3. Reliability diagram (test) ------------------------------------------------------------
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for name, p, ls, mk in [("Heuristic, isotonic-calibrated", test.p_h.values, "-", "o"),
                            ("Heuristic, raw score", test.p_h_uncal.values, "--", "s"),
                            ("Transformer", test.p_t.values, ":", "^")]:
        frac, mean = calibration_curve(test.label, p, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, ls, marker=mk, color="black", ms=4, lw=1.2, label=name, markerfacecolor="white")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraction AI-generated")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(out / "calibration_test.png")
    plt.close(fig)

    # 4. Distribution of p_h with the thorough band ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    b = np.linspace(0, 1, 41)
    ax.hist(test.p_h[test.label == 0], bins=b, color="0.85", edgecolor="black", lw=0.6, label="Human-written")
    ax.hist(test.p_h[test.label == 1], bins=b, histtype="step", color="black", lw=1.4, hatch="////", label="AI-generated")
    th = points["thorough"]
    ax.axvline(th["t_low"], color="black", ls="--", lw=1)
    ax.axvline(th["t_high"], color="black", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"Calibrated heuristic probability $p_h(d)$")
    ax.set_ylabel("Documents (log scale)")
    ax.text((th["t_low"] + th["t_high"]) / 2, ax.get_ylim()[1] * 0.4, "escalated\n(thorough band)", ha="center", fontsize=8.5)
    ax.legend(fontsize=8.5, loc="upper center")
    fig.savefig(out / "ph_distribution.png")
    plt.close(fig)

    # 5. F1 across evaluation splits ----------------------------------------------------------
    splits = [("test", "Test (in-dist.)"), ("heldout_generator", "Held-out\nMistral"), ("mage_ood", "MAGE OOD"),
              ("mage_ood_para", "MAGE OOD\nparaphrased")]
    systems = ["heuristic", "transformer", "cascade_balanced", "cascade_thorough"]
    vals = {lab: {s: R["splits"][k]["metrics"][s]["f1"] for s in systems} for k, lab in splits}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars(ax, [l for _, l in splits], systems, vals, "F1", (0.5, 1.0))
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.savefig(out / "f1_by_split.png")
    plt.close(fig)

    # 6. Robustness: detection rate per RAID attack ---------------------------------------------
    pa = R["splits"]["raid_attack"]["per_attack_recall"]
    order = sorted(pa, key=lambda a: pa[a]["heuristic"])
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bars(ax, [a.replace("_", "\n") for a in order], ["heuristic", "transformer", "cascade_thorough"],
         {a.replace("_", "\n"): pa[a] for a in order}, "Detection rate (recall)", (0, 1.05))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out / "raid_attacks.png")
    plt.close(fig)

    # 7. Fairness: FPR by group ---------------------------------------------------------------
    fr = R["fairness"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    systems_f = ["heuristic", "transformer", "cascade_thorough"]
    bars(ax, ["Native writers", "Non-native writers"], systems_f,
         {"Native writers": {s: fr[s]["fpr_native"] for s in systems_f},
          "Non-native writers": {s: fr[s]["fpr_non_native"] for s in systems_f}}, "False positive rate", (0, 0.75))
    ax.legend(fontsize=8.5, loc="upper left")
    fig.savefig(out / "fairness_fpr.png")
    plt.close(fig)

    # 8. Latency ------------------------------------------------------------------------------
    lat = R["latency"]
    systems_l = ["heuristic", "transformer", "cascade_fast", "cascade_balanced", "cascade_thorough"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, dev, title in [(axes[0], "cpu", "CPU (8 threads)"), (axes[1], "cuda", "GPU (RTX 3060 Ti)")]:
        v = [lat[dev][s]["mean_ms"] for s in systems_l]
        ax.bar(range(len(systems_l)), v, color=[GREY[s] for s in systems_l], edgecolor="black",
               hatch=None, linewidth=0.7)
        for i, s in enumerate(systems_l):
            ax.patches[i].set_hatch(HATCH[s])
            ax.text(i, v[i], f"{v[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(systems_l)))
        ax.set_xticklabels(["Heur.", "Trans.", "Casc.\nfast", "Casc.\nbal.", "Casc.\nthor."], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mean latency per document (ms)")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "latency.png")
    plt.close(fig)

    # 9. LOGO recall per family ---------------------------------------------------------------
    lg = R["logo_heuristic"]
    pg = R["per_generator"]
    fams = [f for f in ["gpt", "cohere", "dolly", "bloomz", "llama", "mpt", "flant5"] if f in lg]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(fams))
    ax.bar(x - 0.2, [pg[f]["heuristic"]["recall"] for f in fams], 0.4, color="0.85", edgecolor="black", label="Family seen in training")
    ax.bar(x + 0.2, [lg[f]["recall_logo"] for f in fams], 0.4, color="0.4", edgecolor="black", hatch="////", label="Family held out (LOGO)")
    ax.set_xticks(x)
    ax.set_xticklabels([{"gpt": "GPT", "flant5": "Flan-T5", "mpt": "MPT", "bloomz": "BLOOMZ"}.get(f, f.capitalize()) for f in fams])
    ax.set_ylabel("Heuristic detection rate (recall)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "logo_recall.png")
    plt.close(fig)

    # 10. Feature importance (XGBoost total gain) ---------------------------------------------
    model = HeuristicModel.load(resolve_path(cfg["heuristic"]["model_path"]))
    from veriscript.heuristic import _xgb_names

    gain = model.head.get_booster().get_score(importance_type="total_gain")
    names = dict(zip(_xgb_names(model.feature_names), model.feature_names))
    imp = sorted(((names.get(k, k), v) for k, v in gain.items()), key=lambda t: -t[1])[:15]
    total = sum(gain.values())
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels = [n.replace("pos_bigram:", "POS ").replace("_", " ") for n, _ in imp][::-1]
    ax.barh(labels, [100 * v / total for _, v in imp][::-1], color="0.5", edgecolor="black")
    ax.set_xlabel("Share of total split gain (%)")
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "feature_importance.png")
    plt.close(fig)
    extras["feature_importance_pct"] = {n: round(100 * v / total, 2) for n, v in imp}

    # Extra statistics -------------------------------------------------------------------------
    extras["confusion_test"] = {}
    for s, p in pr.items():
        tn, fp, fn, tp = confusion_matrix(test.label, p >= 0.5).ravel()
        extras["confusion_test"][s] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    buckets = pd.cut(test.n_words, [99, 250, 500, 10**6], labels=["100-250", "250-500", ">500"])
    extras["f1_by_length"] = {}
    for bk in ["100-250", "250-500", ">500"]:
        m = (buckets == bk).values
        sub = test[m]
        ps = probs(sub, points)
        extras["f1_by_length"][bk] = {"n": int(m.sum()), **{s: round(classification_metrics(sub.label, ps[s])["f1"], 4)
                                                            for s in ("heuristic", "transformer", "cascade_thorough")},
                                      "escalation_thorough": round(float(ConfidenceGate(**points["thorough"]).escalate_mask(sub.p_h.values).mean()), 3)}
    extras["by_source"] = {}
    for src, sub in test.groupby("source_dataset"):
        ps = probs(sub, points)
        extras["by_source"][src] = {"n": len(sub), **{s: round(classification_metrics(sub.label, ps[s])["f1"], 4)
                                                      for s in ("heuristic", "transformer", "cascade_thorough")}}
    extras["windows_test"] = {"mean": float(test.n_windows.mean()), "max": int(test.n_windows.max()),
                              "share_multi": float((test.n_windows > 1).mean())}
    esc_docs = test[ConfidenceGate(**points["thorough"]).escalate_mask(test.p_h.values)]
    extras["escalated_thorough"] = {
        "n": len(esc_docs), "heuristic_acc_on_escalated": float(((esc_docs.p_h >= 0.5) == esc_docs.label).mean()),
        "transformer_acc_on_escalated": float(((esc_docs.p_t >= 0.5) == esc_docs.label).mean()),
        "heuristic_acc_on_fast": float(((test.p_h >= 0.5) == test.label)[~test.id.isin(esc_docs.id)].mean()),
    }
    write_json(extras, REPORTS / "report_extras.json")
    print(json.dumps(extras, indent=1))
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
