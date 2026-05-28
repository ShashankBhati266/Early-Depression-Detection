"""
dct_compression.py
------------------
Implements DCT-based model compression – Algorithm 4 of

    He et al., "Clustered Federated Learning With Adaptive Local
    Differential Privacy on Heterogeneous IoT Data", IEEE IoT J. 2024.

Two-fold benefit
----------------
1. Reduces the *number* of weights that need LDP perturbation in each round
   → less total noise injected.
2. Reduces communication overhead (clients transmit fewer scalars).

Sliding-window extraction ensures every weight eventually participates
in training across multiple communication rounds, compensating for
information loss.

Server reconstruction uses Inverse DCT + round-averaging smoothing.
"""

import numpy as np
from scipy.fft import dct, idct
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Layer-level DCT helpers
# ---------------------------------------------------------------------------

def _dct_layer(W: np.ndarray) -> np.ndarray:
    """Apply 1-D DCT (type-II, orthonormal) to a flattened weight array."""
    flat = W.ravel().astype(np.float64)
    return dct(flat, type=2, norm="ortho")


def _idct_layer(coeffs: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Inverse DCT: reconstruct weights from coefficients."""
    flat = idct(coeffs, type=2, norm="ortho")
    return flat.reshape(original_shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Algorithm 4 – DCT-Compression (client side)
# ---------------------------------------------------------------------------

def dct_compress(
    weights: Weights,
    compression_ratio: float,
    round_number: int,
) -> Tuple[Weights, Dict[str, Tuple[int, int]]]:
    """
    Algorithm 4 – DCTCompression (client side).

    For each layer l:
      1. Compute DCT of the flattened weight vector.
      2. Extract a sliding window of size η·len(W_l) starting at position
         (round_number * window_size) % len(W_l).
      3. Return only the extracted coefficients.

    Parameters
    ----------
    weights           : perturbed local model weights
    compression_ratio : η ∈ (0, 1].  1.0 = no compression.
    round_number      : current communication round t (0-indexed)

    Returns
    -------
    compressed_weights : {layer_name: compressed_coefficient_array}
    window_info        : {layer_name: (start_index, full_length)}
                         needed by the server for reconstruction.
    """
    compressed: Weights      = {}
    window_info: Dict[str, Tuple[int, int]] = {}

    for name, W in weights.items():
        coeffs      = _dct_layer(W)
        full_len    = len(coeffs)
        window_size = max(1, int(np.round(compression_ratio * full_len)))

        # Sliding-window start index (cyclic)
        start = (round_number * window_size) % full_len

        # Extract the window (may wrap around)
        indices = np.arange(start, start + window_size) % full_len
        extracted = coeffs[indices]

        compressed[name]  = extracted.astype(np.float32)
        window_info[name] = (start, full_len)

    return compressed, window_info


# ---------------------------------------------------------------------------
# Server-side reconstruction
# ---------------------------------------------------------------------------

def server_reconstruct(
    received_list: List[Tuple[Weights, Dict[str, Tuple[int, int]]]],
    template_weights: Weights,
    previous_weights: Optional[Weights] = None,
) -> Weights:
    """
    Server-side reconstruction of a cluster model from compressed updates.

    Steps (per layer):
      1. Build a dummy coefficient vector of full length (zeroes).
      2. For each received client update, accumulate values at their window
         positions.
      3. Average non-zero positions; pad empty positions with 0.
      4. Apply Inverse DCT to reconstruct weights.
      5. Average with previous round's model to smooth information loss.

    Parameters
    ----------
    received_list    : [(compressed_weights, window_info), …] from one cluster
    template_weights : weight dict (used only for shape information)
    previous_weights : cluster model from the previous round (for smoothing)

    Returns
    -------
    Reconstructed weight dict.
    """
    if not received_list:
        if previous_weights is not None:
            return {k: v.copy() for k, v in previous_weights.items()}
        return {k: np.zeros_like(v) for k, v in template_weights.items()}

    reconstructed: Weights = {}

    for name, W_ref in template_weights.items():
        orig_shape = W_ref.shape
        full_len   = W_ref.size

        # Accumulate coefficient sums and counts per position
        coeff_sum   = np.zeros(full_len, dtype=np.float64)
        coeff_count = np.zeros(full_len, dtype=np.int32)

        for comp_w, win_info in received_list:
            if name not in comp_w:
                continue
            start, _ = win_info[name]
            chunk     = comp_w[name].ravel()
            win_size  = len(chunk)
            indices   = np.arange(start, start + win_size) % full_len
            np.add.at(coeff_sum,   indices, chunk)
            np.add.at(coeff_count, indices, 1)

        # Average where we have data; leave 0 elsewhere (padding)
        avg_coeffs = np.where(
            coeff_count > 0,
            coeff_sum / np.maximum(coeff_count, 1),
            0.0,
        )

        # Inverse DCT → reconstructed weights
        W_rec = _idct_layer(avg_coeffs, orig_shape)

        # Smooth with previous round to counteract information loss
        if previous_weights is not None and name in previous_weights:
            W_rec = 0.5 * (W_rec + previous_weights[name])

        reconstructed[name] = W_rec

    return reconstructed


# ---------------------------------------------------------------------------
# Convenience: compress → collect → reconstruct for one cluster in one round
# ---------------------------------------------------------------------------

def compress_and_reconstruct(
    client_weights_list: List[Weights],
    compression_ratio: float,
    round_number: int,
    template_weights: Weights,
    previous_weights: Optional[Weights] = None,
) -> Weights:
    """
    Full DCT-compression pipeline for a single cluster in one FL round.

    1. Each client compresses its perturbed weights.
    2. Server reconstructs the cluster model.

    Parameters
    ----------
    client_weights_list : list of perturbed weight dicts from one cluster
    compression_ratio   : η
    round_number        : current round (0-indexed)
    template_weights    : weight dict for shape reference
    previous_weights    : last round's cluster model

    Returns
    -------
    Reconstructed cluster model weights.
    """
    compressed_list = []
    for w in client_weights_list:
        comp, winfo = dct_compress(w, compression_ratio, round_number)
        compressed_list.append((comp, winfo))

    return server_reconstruct(compressed_list, template_weights, previous_weights)


# ---------------------------------------------------------------------------
# Utility: effective noise reduction factor
# ---------------------------------------------------------------------------

def noise_reduction_factor(compression_ratio: float) -> float:
    """
    Approximate factor by which DCT compression reduces the total LDP noise
    injected per round.  (Ratio of parameters perturbed to total parameters.)
    """
    return compression_ratio


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.mlp import init_weights

    template = init_weights(n_features=7, hidden=128, n_classes=2, seed=0)
    # Simulate 5 clients in one cluster
    clients_w = [init_weights(n_features=7, hidden=128, n_classes=2, seed=i+1)
                 for i in range(5)]

    for eta in [0.25, 0.5, 0.70, 0.90, 1.0]:
        rec = compress_and_reconstruct(
            clients_w, compression_ratio=eta,
            round_number=0, template_weights=template
        )
        # Compare to plain average (no compression)
        from models.mlp import average_weights
        plain = average_weights(clients_w)

        diff = np.mean([
            np.abs(rec[k] - plain[k]).mean()
            for k in template
        ])
        total_params = sum(v.size for v in template.values())
        params_sent  = sum(
            max(1, int(round(eta * v.size)))
            for v in template.values()
        )
        print(f"η={eta:.2f} | params sent: {params_sent}/{total_params} "
              f"({100*eta:.0f}%) | mean abs diff vs plain avg: {diff:.6f}")
