"""
generalized_pm.py
-----------------
Implements the Generalised Piecewise Mechanism (GPM) – Algorithm 1 of

    "Clustered Federated Learning With Adaptive Local Differential Privacy
     on Heterogeneous IoT Data"  (He et al., IEEE IoT J. 2024)

The GPM is an ε-LDP randomiser for a single scalar weight w ∈ [c-r, c+r].
Key properties proven in the paper:
  • Satisfies ε-LDP   (Theorem 1)
  • Unbiased estimator of w   (Theorem 2)
  • Variance ≤ 4r²·e^(ε/2) / [3·(e^(ε/2)-1)²]   (Lemma 1)

Usage
-----
    from ldp.generalized_pm import generalized_pm, apply_gpm_to_layer

    w_star = generalized_pm(w=0.3, c=0.0, r=0.5, epsilon=3.0)
    W_star = apply_gpm_to_layer(W, c=c_l, r=r_l, epsilon=3.0)
"""

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Core scalar mechanism (Algorithm 1)
# ---------------------------------------------------------------------------

def _gpm_params(r: float, epsilon: float):
    """Pre-compute the constants b and p from the paper."""
    half_eps = epsilon / 2.0
    e_half   = np.exp(half_eps)
    b = (e_half + 1.0) / (e_half - 1.0)          # eq. after (4)
    p = (np.exp(epsilon) - e_half) / (2.0 * r * (e_half + 1.0))
    return b, p, e_half


def _tau(w: float, c: float, r: float, b: float) -> float:
    """Lower boundary of the high-probability interval τ(w)."""
    return (r / 2.0) * ((b + 1.0) * ((w - c) / r) - b + 1.0) + c


def generalized_pm(
    w: float,
    c: float,
    r: float,
    epsilon: float,
    rng: np.random.Generator = None,
) -> float:
    """
    Perturb a single weight w with ε-LDP using the Generalised PM.

    Parameters
    ----------
    w       : original weight value (must be in [c-r, c+r])
    c       : centre of the layer's weight range
    r       : radius of the layer's weight range  (r > 0)
    epsilon : privacy budget  ε > 0
    rng     : numpy Generator (created internally if None)

    Returns
    -------
    w_star  : perturbed weight  ∈ [c - r·b, c + r·b]
    """
    if rng is None:
        rng = np.random.default_rng()

    r    = max(r, 1e-8)
    b, _, e_half = _gpm_params(r, epsilon)

    rb   = r * b
    tau_ = _tau(w, c, r, b)
    sig_ = tau_ + r * (b - 1.0)

    # Ensure tau ≤ sigma (numerical safety)
    if tau_ > sig_:
        tau_, sig_ = sig_, tau_

    # Clamp outputs to valid range
    lo, hi = c - rb, c + rb

    # Sampling probability for high-probability interval  (Algorithm 1, line 2)
    prob_high = e_half / (e_half + 1.0)

    x = rng.uniform(0.0, 1.0)
    if x < prob_high:
        # Sample uniformly from [τ(w), σ(w)]
        t1 = max(lo, tau_)
        t2 = min(hi, sig_)
        if t1 >= t2:
            return float(np.clip(w, lo, hi))   # degenerate fallback
        return float(rng.uniform(t1, t2))
    else:
        # Sample uniformly from [c-rb, τ(w)) ∪ (σ(w), c+rb]
        left_len  = max(0.0, tau_ - lo)
        right_len = max(0.0, hi  - sig_)
        total_len = left_len + right_len
        if total_len < 1e-12:
            return float(np.clip(w, lo, hi))   # degenerate fallback
        u = rng.uniform(0.0, total_len)
        if u < left_len:
            return float(lo + u)
        else:
            return float(sig_ + (u - left_len))


# ---------------------------------------------------------------------------
# Vectorised version – apply GPM to a whole weight array
# ---------------------------------------------------------------------------

def apply_gpm_to_layer(
    W: np.ndarray,
    c: float,
    r: float,
    epsilon: float,
    seed: int = None,
) -> np.ndarray:
    """
    Apply the Generalised PM independently to every element of W.

    Parameters
    ----------
    W       : weight array (any shape)
    c       : layer centre
    r       : layer radius
    epsilon : privacy budget
    seed    : optional random seed for reproducibility

    Returns
    -------
    W_star  : perturbed array (same shape as W)
    """
    rng    = np.random.default_rng(seed)
    flat   = W.ravel()
    result = np.empty_like(flat)
    for i, wi in enumerate(flat):
        result[i] = generalized_pm(float(wi), c, r, epsilon, rng)
    return result.reshape(W.shape)


# ---------------------------------------------------------------------------
# Theoretical variance bound (for logging / analysis)
# ---------------------------------------------------------------------------

def variance_bound(r: float, epsilon: float) -> float:
    """
    Upper bound on Var[M(w)] from Lemma 1:
        4 r² e^(ε/2) / [3 (e^(ε/2) - 1)²]
    """
    e_half = np.exp(epsilon / 2.0)
    return (4.0 * r**2 * e_half) / (3.0 * (e_half - 1.0)**2)


# ---------------------------------------------------------------------------
# Baseline: Duchi's mechanism  (for comparison in experiments)
# ---------------------------------------------------------------------------

def duchi_mechanism(
    w: float,
    epsilon: float,
    rng: np.random.Generator = None,
) -> float:
    """
    Duchi et al. (2013) univariate LDP mechanism for w ∈ [-1, 1].
    Used as a baseline in Section IV-C of the paper.
    """
    if rng is None:
        rng = np.random.default_rng()
    e_eps = np.exp(epsilon)
    C     = (e_eps + 1.0) / (e_eps - 1.0)
    p     = 0.5 + 0.5 * w * (e_eps - 1.0) / (e_eps + 1.0)
    sign  = 1.0 if rng.uniform() < p else -1.0
    return float(sign * C)


def apply_duchi_to_layer(
    W: np.ndarray, epsilon: float, seed: int = None
) -> np.ndarray:
    """Apply Duchi's mechanism to each element of W (normalised to [-1,1])."""
    rng    = np.random.default_rng(seed)
    flat   = np.clip(W.ravel(), -1.0, 1.0)
    result = np.array([duchi_mechanism(float(w), epsilon, rng) for w in flat])
    return result.reshape(W.shape)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    w_true = 0.3
    c, r, eps = 0.0, 1.0, 3.0
    n_trials   = 100_000

    samples = [generalized_pm(w_true, c, r, eps, rng) for _ in range(n_trials)]
    mean_est  = np.mean(samples)
    var_est   = np.var(samples)
    var_bound = variance_bound(r, eps)

    print(f"True w         : {w_true:.4f}")
    print(f"E[M(w)]        : {mean_est:.4f}   (should ≈ {w_true})")
    print(f"Var[M(w)]      : {var_est:.4f}   (bound = {var_bound:.4f})")
    print(f"Unbiased check : {'PASS' if abs(mean_est - w_true) < 0.01 else 'FAIL'}")
    print(f"Variance check : {'PASS' if var_est <= var_bound * 1.05 else 'FAIL'}")
