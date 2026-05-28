"""
perturbation.py
---------------
Implements adaptive layer-wise weight perturbation – Algorithm 2 of

    He et al., "Clustered Federated Learning With Adaptive Local
    Differential Privacy on Heterogeneous IoT Data", IEEE IoT J. 2024.

Key idea
--------
Instead of using one global clipping range for the entire model, we compute
a centre c_l and radius r_l *per layer* (from the server's current model
statistics) and perturb each weight at the granularity of individual values
rather than coarse ±1 buckets.  This reduces excess noise and improves
model utility (see Lemma 1 / Theorem 3 in the paper).

Fixed-range baseline
--------------------
Also exposes `fixed_range_perturbation` for the ablation study in Sec. IV-B.
"""

import numpy as np
from typing import Dict, Tuple

from ldp.generalized_pm import apply_gpm_to_layer, apply_duchi_to_layer


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights     = Dict[str, np.ndarray]
LayerStats  = Dict[str, Tuple[float, float]]   # {name: (c_l, r_l)}
ClipRanges  = Dict[int, LayerStats]            # {cluster_id: LayerStats}


# ---------------------------------------------------------------------------
# Algorithm 2 – LocalPerturbation (adaptive range, per-layer)
# ---------------------------------------------------------------------------

def compute_layer_stats(weights: Weights) -> LayerStats:
    """
    Compute (centre, radius) for every parameter tensor in *weights*.

    c_l  = mean of all values in layer l
    r_l  = max|w - c_l|  for w in layer l
    """
    stats: LayerStats = {}
    for name, W in weights.items():
        c = float(W.mean())
        r = float(np.abs(W - c).max())
        stats[name] = (c, max(r, 1e-8))
    return stats


def clip_weights(
    weights: Weights, layer_stats: LayerStats
) -> Weights:
    """
    Layer-wise clipping (equation 16 in the paper):
        w ← clip(w, c_l - r_l, c_l + r_l)
    """
    clipped: Weights = {}
    for name, W in weights.items():
        c, r = layer_stats[name]
        clipped[name] = np.clip(W, c - r, c + r)
    return clipped


def adaptive_perturbation(
    weights: Weights,
    layer_stats: LayerStats,
    epsilon: float,
    seed: int = None,
) -> Weights:
    """
    Algorithm 2 – LocalPerturbation.

    Steps
    -----
    1. For each layer l:
       a. Clip   w ← clip(w, c_l - r_l, c_l + r_l)
       b. Perturb w* ← GeneralisedPM(w, c_l, r_l, ε)
    2. Return perturbed weights dict.

    Parameters
    ----------
    weights     : local model weights after training
    layer_stats : {layer_name: (c_l, r_l)} from the server
    epsilon     : per-weight privacy budget ε
    seed        : random seed (None = non-deterministic)

    Returns
    -------
    perturbed weights dict (same structure as *weights*)
    """
    rng_base = np.random.default_rng(seed)
    perturbed: Weights = {}

    for i, (name, W) in enumerate(weights.items()):
        c, r = layer_stats[name]
        # --- Step 1a: clip to [c-r, c+r]
        W_clipped = np.clip(W, c - r, c + r)
        # --- Step 1b: apply GPM element-wise
        layer_seed = int(rng_base.integers(0, 2**31)) + i
        perturbed[name] = apply_gpm_to_layer(W_clipped, c, r, epsilon,
                                             seed=layer_seed)
    return perturbed


# ---------------------------------------------------------------------------
# Fixed-range baseline (ablation / comparison in Sec. IV-B)
# ---------------------------------------------------------------------------

def fixed_range_perturbation(
    weights: Weights,
    epsilon: float,
    c_fixed: float = 0.0,
    r_fixed: float = 0.015,
    seed: int = None,
) -> Weights:
    """
    All layers share the same fixed clipping range [c_fixed - r_fixed, c_fixed + r_fixed].
    Used as a baseline in the paper (Fig. 2).
    """
    rng_base = np.random.default_rng(seed)
    perturbed: Weights = {}

    for i, (name, W) in enumerate(weights.items()):
        W_clipped = np.clip(W, c_fixed - r_fixed, c_fixed + r_fixed)
        layer_seed = int(rng_base.integers(0, 2**31)) + i
        perturbed[name] = apply_gpm_to_layer(
            W_clipped, c_fixed, r_fixed, epsilon, seed=layer_seed
        )
    return perturbed


# ---------------------------------------------------------------------------
# Duchi baseline (ablation / comparison in Sec. IV-C)
# ---------------------------------------------------------------------------

def duchi_perturbation(
    weights: Weights,
    epsilon: float,
    seed: int = None,
) -> Weights:
    """
    Applies Duchi's mechanism to every weight.
    Weights are normalised to [-1, 1] before perturbation (Duchi requirement).
    """
    rng_base = np.random.default_rng(seed)
    perturbed: Weights = {}

    for i, (name, W) in enumerate(weights.items()):
        # Normalise to [-1, 1]
        W_min, W_max = W.min(), W.max()
        span = max(W_max - W_min, 1e-8)
        W_norm = 2.0 * (W - W_min) / span - 1.0
        layer_seed = int(rng_base.integers(0, 2**31)) + i
        W_pert = apply_duchi_to_layer(W_norm, epsilon, seed=layer_seed)
        # Rescale back (approximation – Duchi outputs are in [-C, C])
        perturbed[name] = W_pert
    return perturbed


# ---------------------------------------------------------------------------
# Server-side helper: broadcast clipping ranges for all cluster models
# ---------------------------------------------------------------------------

def compute_all_cluster_stats(
    cluster_models: Dict[int, Weights]
) -> ClipRanges:
    """
    For each cluster model, compute layer stats.
    Returns {cluster_id: LayerStats}.
    """
    return {cid: compute_layer_stats(w) for cid, w in cluster_models.items()}


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from models.mlp import init_weights, flatten_weights

    w = init_weights(n_features=7, hidden=128, n_classes=2)
    stats = compute_layer_stats(w)
    print("Layer stats:")
    for k, (c, r) in stats.items():
        print(f"  {k:4s}: centre={c:.5f}  radius={r:.5f}")

    w_pert = adaptive_perturbation(w, stats, epsilon=3.0, seed=42)
    print("\nAdaptive perturbation done.")

    w_fixed = fixed_range_perturbation(w, epsilon=3.0, seed=42)
    print("Fixed-range perturbation done.")

    # Check mean preservation (should be approximate)
    orig_mean = np.mean([v.mean() for v in w.values()])
    pert_mean = np.mean([v.mean() for v in w_pert.values()])
    print(f"\nMean weight (original): {orig_mean:.5f}")
    print(f"Mean weight (perturbed): {pert_mean:.5f}   (should be close)")
