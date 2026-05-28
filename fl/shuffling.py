"""
shuffling.py
------------
Implements parameter shuffling – Algorithm 3 of

    He et al., "Clustered Federated Learning With Adaptive Local
    Differential Privacy on Heterogeneous IoT Data", IEEE IoT J. 2024.

Purpose
-------
Mitigates privacy-budget explosion by breaking the link between a client's
identity and its model updates:

  1. Each client's cluster identity is mapped to a randomised fake id via
     the ID distributor.
  2. Every weight is labelled with (fake_id, location_in_model).
  3. A uniformly random time-delay is sampled per weight so the server
     cannot correlate arrivals to a single client.

The server still aggregates correctly because all clients in the same
cluster share the same fake id (see Section III-C of the paper).

Privacy note
------------
Because the shuffling mechanism randomises transmission order and hides
cluster identities, it prevents the server from accumulating a per-client
privacy budget across rounds – effectively amplifying privacy without
spending extra budget (Rényi amplification via shuffling).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights = Dict[str, np.ndarray]
Triple  = Tuple[int, int, float, float]   # (fake_id, location, value, delay)


# ---------------------------------------------------------------------------
# ID Distributor
# ---------------------------------------------------------------------------

class IDDistributor:
    """
    Maintains a deterministic but secret mapping  cluster_id → fake_id.
    The server never has access to this mapping.

    In a real deployment this would live inside a trusted hardware enclave
    or be implemented via a cryptographic protocol; here we simulate it
    with a seeded permutation held only by the clients.
    """

    def __init__(self, n_clusters: int, seed: int = None):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_clusters)
        self._map: Dict[int, int] = {i: int(perm[i]) for i in range(n_clusters)}
        self._inv: Dict[int, int] = {v: k for k, v in self._map.items()}

    def encode(self, cluster_id: int) -> int:
        """Client-side: map real cluster id → fake id."""
        return self._map[cluster_id]

    def decode(self, fake_id: int) -> int:
        """
        Inverse mapping (used only in evaluation / unit tests).
        The server does NOT call this; it just groups by fake_id.
        """
        return self._inv[fake_id]

    def refresh(self, seed: int = None):
        """Re-randomise the mapping (called every round for stronger privacy)."""
        n = len(self._map)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        self._map = {i: int(perm[i]) for i in range(n)}
        self._inv = {v: k for k, v in self._map.items()}


# ---------------------------------------------------------------------------
# Client-side: build triples and simulate random delays
# ---------------------------------------------------------------------------

def _flatten_indexed(weights: Weights) -> List[Tuple[int, float]]:
    """
    Returns [(global_index, value), …] for all weights, preserving order.
    """
    pairs = []
    idx = 0
    for W in weights.values():
        for val in W.ravel():
            pairs.append((idx, float(val)))
            idx += 1
    return pairs


def shuffle_and_label(
    perturbed_weights: Weights,
    cluster_id: int,
    distributor: IDDistributor,
    T: int = 100,
    seed: int = None,
) -> List[Triple]:
    """
    Algorithm 3 – ParameterShuffling (client side).

    1. Get a randomised cluster identity from the distributor.
    2. Label every weight with (fake_id, location).
    3. Assign a uniform random delay ∈ [0, T] to each weight.
    4. Return the list of triples sorted by delay (transmission order).

    Parameters
    ----------
    perturbed_weights : LDP-perturbed local model weights
    cluster_id        : the client's true cluster id
    distributor       : IDDistributor shared among all clients
    T                 : maximum transmission delay (communication rounds)
    seed              : random seed

    Returns
    -------
    List of (fake_id, location, value, delay) sorted by delay.
    """
    rng     = np.random.default_rng(seed)
    fake_id = distributor.encode(cluster_id)
    indexed = _flatten_indexed(perturbed_weights)

    triples: List[Triple] = []
    for loc, val in indexed:
        delay = float(rng.uniform(0.0, T))
        triples.append((fake_id, loc, val, delay))

    # Sort by delay so the server receives them in randomised order
    triples.sort(key=lambda t: t[3])
    return triples


# ---------------------------------------------------------------------------
# Server-side: collect triples and aggregate per fake cluster
# ---------------------------------------------------------------------------

def server_collect(
    all_triples: List[Triple],
    n_total_weights: int,
) -> Dict[int, Dict[int, List[float]]]:
    """
    Collect all triples from all clients and organise them as:
        {fake_id: {location: [values from different clients]}}

    Parameters
    ----------
    all_triples      : merged list of triples from all clients
    n_total_weights  : total number of scalar weights in the model

    Returns
    -------
    Nested dict ready for aggregation.
    """
    collected: Dict[int, Dict[int, List[float]]] = {}
    for fake_id, loc, val, _ in all_triples:
        if fake_id not in collected:
            collected[fake_id] = {}
        if loc not in collected[fake_id]:
            collected[fake_id][loc] = []
        collected[fake_id][loc].append(val)
    return collected


def aggregate_triples(
    collected: Dict[int, Dict[int, List[float]]],
    n_total_weights: int,
    template_weights: Dict[str, np.ndarray],
    previous_weights: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    distributor: Optional[IDDistributor] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Server-side aggregation.

    For each fake cluster id:
      • Average received values per location.
      • Positions with no received value are padded with 0.
      • Reshape back to the model's layer structure.
      • (Optional) Average with the previous round's model to smooth
        information loss from DCT compression / partial reception.

    Parameters
    ----------
    collected         : output of server_collect()
    n_total_weights   : total scalar weights
    template_weights  : a weight dict used only for shape information
    previous_weights  : {cluster_id: weights} from last round (for smoothing)
    distributor       : if provided, translates fake_id → real cluster_id

    Returns
    -------
    {cluster_id: aggregated_weights}
    """
    # Build shape info from template
    layer_shapes = [(k, v.shape, v.size) for k, v in template_weights.items()]

    result: Dict[int, Dict[str, np.ndarray]] = {}

    for fake_id, loc_dict in collected.items():
        # Build flat aggregated vector
        flat = np.zeros(n_total_weights, dtype=np.float32)
        for loc, vals in loc_dict.items():
            flat[loc] = float(np.mean(vals))

        # Reshape to weight dict
        agg_weights: Dict[str, np.ndarray] = {}
        idx = 0
        for layer_name, shape, size in layer_shapes:
            agg_weights[layer_name] = flat[idx: idx + size].reshape(shape)
            idx += size

        # Smooth with previous round (counteracts information loss)
        if previous_weights is not None:
            real_id = distributor.decode(fake_id) if distributor else fake_id
            if real_id in previous_weights:
                prev = previous_weights[real_id]
                for k in agg_weights:
                    agg_weights[k] = 0.5 * (agg_weights[k] + prev[k])

        # Map fake_id back to real cluster id
        real_id = distributor.decode(fake_id) if distributor else fake_id
        result[real_id] = agg_weights

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: shuffle → collect → aggregate in one call
# ---------------------------------------------------------------------------

def apply_shuffling(
    client_updates: List[Tuple[int, Weights]],   # [(cluster_id, weights), …]
    distributor: IDDistributor,
    template_weights: Weights,
    previous_weights: Optional[Dict[int, Weights]] = None,
    T: int = 100,
    seed: int = None,
) -> Dict[int, Weights]:
    """
    Full shuffling pipeline for one FL round.

    Parameters
    ----------
    client_updates   : [(real_cluster_id, perturbed_weights), …]
    distributor      : shared IDDistributor
    template_weights : weight dict used only for shape reconstruction
    previous_weights : last round's cluster models (for smoothing)
    T                : max delay
    seed             : random seed

    Returns
    -------
    Aggregated cluster models {real_cluster_id: weights}
    """
    rng = np.random.default_rng(seed)
    n_weights = sum(v.size for v in template_weights.values())

    # --- Client side: each client shuffles and labels its weights
    all_triples: List[Triple] = []
    for i, (cid, w) in enumerate(client_updates):
        client_seed = int(rng.integers(0, 2**31)) + i
        triples = shuffle_and_label(w, cid, distributor, T=T, seed=client_seed)
        all_triples.extend(triples)

    # Merge and sort all triples by delay (simulates network arrival order)
    all_triples.sort(key=lambda t: t[3])

    # --- Server side: collect and aggregate
    collected = server_collect(all_triples, n_weights)
    return aggregate_triples(
        collected, n_weights, template_weights,
        previous_weights=previous_weights,
        distributor=distributor,
    )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.mlp import init_weights

    n_clusters = 4
    dist = IDDistributor(n_clusters=n_clusters, seed=0)

    # Simulate 8 clients (2 per cluster)
    template = init_weights(n_features=7)
    updates = []
    for uid in range(8):
        cid = uid % n_clusters
        w = init_weights(n_features=7, seed=uid)
        updates.append((cid, w))

    agg = apply_shuffling(updates, dist, template, T=100, seed=42)
    print(f"Aggregated {len(agg)} cluster models via shuffling.")
    for cid, w in sorted(agg.items()):
        total = sum(v.size for v in w.values())
        print(f"  Cluster {cid}: {total} parameters")
