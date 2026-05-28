"""
plot_results.py
---------------
Generates all plots from the ACS-FL experiments, mirroring
Figures 1-5 in He et al. (IEEE IoT J. 2024).

Usage
-----
    # After running experiments:
    python results/plot_results.py

    # Or call directly from Python:
    from results.plot_results import plot_all
    plot_all(history_dict, outdir="results/figures")
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

COLORS   = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
MARKERS  = ["o", "s", "^", "D", "v", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]

def _style_ax(ax, title="", xlabel="Communication Round", ylabel="Test Accuracy"):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1 replica: Effect of cluster size  (Experiment A)
# ---------------------------------------------------------------------------

def plot_cluster_size(
    results_by_k: Dict[int, Dict],
    noise_free_results: Optional[Dict[int, Dict]] = None,
    outdir: str = "results/figures",
):
    """
    results_by_k : {n_clusters: history_dict}
    noise_free_results : same structure but with perturbation_mode='none'
    """
    n = len(results_by_k)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (k, hist) in zip(axes, sorted(results_by_k.items())):
        rounds = hist["round"]
        accs   = hist["mean_acc"]

        ax.plot(rounds, accs, color=COLORS[0], lw=1.8, label=f"ACS-FL (k={k})")

        if noise_free_results and k in noise_free_results:
            nf = noise_free_results[k]
            ax.plot(nf["round"], nf["mean_acc"],
                    color=COLORS[1], lw=1.4, linestyle="--",
                    label="Noise-free")
            # Shade gap
            rounds_nf = np.array(nf["round"])
            accs_nf   = np.array(nf["mean_acc"])
            rounds_a  = np.array(rounds)
            accs_a    = np.array(accs)
            ax.fill_between(rounds_a, accs_a, accs_nf,
                            alpha=0.15, color="grey", label="Gap")

        _style_ax(ax, title=f"Clusters = {k}")

    fig.suptitle("Fig 1 – Effect of Number of Clusters on Accuracy",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "fig1_cluster_size.png"))


# ---------------------------------------------------------------------------
# Figure 2 replica: Adaptive vs Fixed LDP  (Experiment B)
# ---------------------------------------------------------------------------

def plot_adaptive_vs_fixed(
    results: Dict[str, Dict],
    outdir: str = "results/figures",
):
    """
    results keys expected: 'eps=6.0_mode=adaptive', 'eps=6.0_mode=fixed', …
    """
    epsilons = sorted(set(
        float(k.split("_")[0].split("=")[1]) for k in results
    ))
    n = len(epsilons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ci, mode in enumerate(["adaptive", "fixed"]):
            key  = f"eps={eps}_mode={mode}"
            hist = results.get(key)
            if hist is None:
                continue
            label = f"{mode.capitalize()} range"
            ax.plot(hist["round"], hist["mean_acc"],
                    color=COLORS[ci], lw=1.8,
                    linestyle=LINESTYLES[ci], label=label)
        _style_ax(ax, title=f"ε = {eps}")

    fig.suptitle("Fig 2 – Adaptive LDP vs Fixed LDP",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "fig2_adaptive_vs_fixed.png"))


# ---------------------------------------------------------------------------
# Figure 3 replica: ACS-FL vs LDP-FL  (uses cluster-size results as proxy)
# ---------------------------------------------------------------------------

def plot_acsfl_vs_baseline(
    acsfl_results: Dict[str, Dict],
    baseline_results: Dict[str, Dict],
    outdir: str = "results/figures",
):
    """
    acsfl_results    : {label: history_dict}  for ACS-FL runs
    baseline_results : {label: history_dict}  for LDP-FL / Duchi runs
    """
    all_keys = list(acsfl_results.keys())
    n = len(all_keys)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, all_keys):
        acsfl = acsfl_results[key]
        ax.plot(acsfl["round"], acsfl["mean_acc"],
                color=COLORS[0], lw=1.8, label="ACS-FL")
        if key in baseline_results:
            base = baseline_results[key]
            ax.plot(base["round"], base["mean_acc"],
                    color=COLORS[1], lw=1.4, linestyle="--", label="LDP-FL (Duchi)")
        _style_ax(ax, title=key)

    fig.suptitle("Fig 3 – ACS-FL vs LDP-FL Baseline",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "fig3_acsfl_vs_baseline.png"))


# ---------------------------------------------------------------------------
# Figure 4 replica: Generalised PM vs Duchi  (Experiment C)
# ---------------------------------------------------------------------------

def plot_gpm_vs_duchi(
    results: Dict[str, Dict],
    outdir: str = "results/figures",
):
    epsilons = sorted(set(
        float(k.split("_")[0].split("=")[1]) for k in results
    ))
    n = len(epsilons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ci, mode in enumerate(["adaptive", "duchi"]):
            key  = f"eps={eps}_mode={mode}"
            hist = results.get(key)
            if hist is None:
                continue
            label = "Generalised PM" if mode == "adaptive" else "Duchi's Solution"
            ax.plot(hist["round"], hist["mean_acc"],
                    color=COLORS[ci], lw=1.8,
                    linestyle=LINESTYLES[ci], label=label)
        _style_ax(ax, title=f"ε = {eps}")

    fig.suptitle("Fig 4 – Generalised PM vs Duchi's Mechanism",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "fig4_gpm_vs_duchi.png"))


# ---------------------------------------------------------------------------
# Figure 5 replica: DCT compression ratios  (Experiment D)
# ---------------------------------------------------------------------------

def plot_dct_compression(
    results: Dict[str, Dict],
    outdir: str = "results/figures",
):
    epsilons = sorted(set(
        float(k.split("_")[0].split("=")[1]) for k in results
    ))
    etas = sorted(set(
        float(k.split("_")[1].split("=")[1]) for k in results
    ))
    n = len(epsilons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        for ci, eta in enumerate(etas):
            key  = f"eps={eps}_eta={eta}"
            hist = results.get(key)
            if hist is None:
                continue
            label = f"η={eta:.0%}" if eta < 1.0 else "No compression"
            ax.plot(hist["round"], hist["mean_acc"],
                    color=COLORS[ci % len(COLORS)], lw=1.5,
                    linestyle=LINESTYLES[ci % len(LINESTYLES)],
                    marker=MARKERS[ci % len(MARKERS)],
                    markevery=max(1, len(hist["round"]) // 8),
                    markersize=4, label=label)
        _style_ax(ax, title=f"ε = {eps}")

    fig.suptitle("Fig 5 – Effect of DCT Compression Ratio",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "fig5_dct_compression.png"))


# ---------------------------------------------------------------------------
# Privacy budget curve
# ---------------------------------------------------------------------------

def plot_privacy_budget(
    history: Dict,
    label: str = "ACS-FL",
    outdir: str = "results/figures",
):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history["round"], history["privacy_budget"],
            color=COLORS[0], lw=1.8, label=label)
    _style_ax(ax, title="Cumulative Privacy Budget",
              ylabel="ε (cumulative)")
    fig.tight_layout()
    _save(fig, os.path.join(outdir, "privacy_budget.png"))


# ---------------------------------------------------------------------------
# Convenience: plot everything from saved JSON files
# ---------------------------------------------------------------------------

def plot_all_from_json(results_dir: str = "results", outdir: str = "results/figures"):
    """Load all experiment JSON files and generate every figure."""
    def _load(fname):
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    expA = _load("exp_A_cluster_size.json")
    expB = _load("exp_B_adaptive_vs_fixed.json")
    expC = _load("exp_C_gpm_vs_duchi.json")
    expD = _load("exp_D_dct_compression.json")

    # Convert JSON dicts back to history-like dicts for plotting
    def _to_hist(acc_list):
        return {"round": list(range(1, len(acc_list) + 1)), "mean_acc": acc_list}

    if expA:
        plot_cluster_size({int(k): _to_hist(v) for k, v in expA.items()},
                          outdir=outdir)
    if expB:
        plot_adaptive_vs_fixed({k: _to_hist(v) for k, v in expB.items()},
                               outdir=outdir)
    if expC:
        plot_gpm_vs_duchi({k: _to_hist(v) for k, v in expC.items()},
                          outdir=outdir)
    if expD:
        plot_dct_compression({k: _to_hist(v) for k, v in expD.items()},
                             outdir=outdir)

    print("\nAll figures saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()
    plot_all_from_json(args.results_dir, args.outdir)
