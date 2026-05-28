"""
clustered_fl.py
---------------
Implements clustered Federated Learning – the FL backbone of ACS-FL.

Cluster assignment follows the IFCA protocol (Ghosh et al. 2020):
    Each client independently picks the cluster model that achieves
    the lowest loss on its local data (equation 3 in the paper).

The server maintains m separate models (one per cluster) and performs
FedAvg aggregation within each cluster.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from models.mlp import (
    init_weights, copy_weights, average_weights,
    evaluate_loss, local_train, layer_stats, predict,
)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights     = Dict[str, np.ndarray]
ClientData  = Dict                    # {X_train, y_train, X_test, y_test, cluster}


# ---------------------------------------------------------------------------
# Cluster assignment (equation 3 in the paper)
# ---------------------------------------------------------------------------

def assign_cluster(
    client_data: ClientData,
    cluster_models: Dict[int, Weights],
) -> int:
    """
    Pick the cluster whose model has the lowest loss on this client's data.
    Returns the cluster id (integer key in cluster_models).
    """
    best_cid  = None
    best_loss = float("inf")
    X, y = client_data["X_train"], client_data["y_train"]

    for cid, w in cluster_models.items():
        loss = evaluate_loss(w, X, y)
        if loss < best_loss:
            best_loss = best_cid = None  # reset
            best_loss = loss
            best_cid  = cid

    return best_cid


# ---------------------------------------------------------------------------
# Server: initialise cluster models
# ---------------------------------------------------------------------------

def init_cluster_models(
    m: int,
    n_features: int,
    hidden: int = 128,
    n_classes: int = 2,
    seed: int = 42,
) -> Dict[int, Weights]:
    """Randomly initialise m independent cluster models."""
    return {
        i: init_weights(n_features, hidden, n_classes, seed=seed + i)
        for i in range(m)
    }


# ---------------------------------------------------------------------------
# Client: local training + cluster assignment
# ---------------------------------------------------------------------------

def client_update(
    client_data: ClientData,
    cluster_models: Dict[int, Weights],
    lr: float = 0.001,
    local_epochs: int = 5,
    batch_size: int = 16,
    seed: int = 0,
) -> Tuple[int, Weights]:
    """
    1. Assign client to the best cluster.
    2. Run local SGD on the chosen cluster model.
    3. Return (assigned_cluster_id, updated_weights).
    """
    cid      = assign_cluster(client_data, cluster_models)
    w_local  = local_train(
        weights   = cluster_models[cid],
        X         = client_data["X_train"],
        y         = client_data["y_train"],
        lr        = lr,
        epochs    = local_epochs,
        batch_size= batch_size,
        seed      = seed,
    )
    return cid, w_local


# ---------------------------------------------------------------------------
# Server: aggregate per-cluster
# ---------------------------------------------------------------------------

def server_aggregate(
    updates: List[Tuple[int, Weights]],   # [(cid, weights), …]
    current_models: Dict[int, Weights],
    min_clients: int = 1,
) -> Dict[int, Weights]:
    """
    FedAvg aggregation within each cluster.
    If a cluster receives no updates, its model is left unchanged.

    Parameters
    ----------
    updates         : list of (cluster_id, local_weights) from clients
    current_models  : existing cluster models (used as fallback)
    min_clients     : minimum updates needed to trigger aggregation

    Returns
    -------
    Updated cluster models dict.
    """
    # Group updates by cluster id
    buckets: Dict[int, List[Weights]] = {cid: [] for cid in current_models}
    for cid, w in updates:
        if cid in buckets:
            buckets[cid].append(w)

    new_models = {}
    for cid, w_list in buckets.items():
        if len(w_list) >= min_clients:
            new_models[cid] = average_weights(w_list)
        else:
            new_models[cid] = copy_weights(current_models[cid])

    return new_models


# ---------------------------------------------------------------------------
# Server: compute per-cluster, per-layer clipping ranges
# ---------------------------------------------------------------------------

def compute_clipping_ranges(
    cluster_models: Dict[int, Weights],
) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """
    For each cluster model, compute (c_l, r_l) for every layer.
    These are broadcast to clients before their local updates.

    Returns  {cluster_id: {layer_name: (centre, radius)}}
    """
    ranges = {}
    for cid, w in cluster_models.items():
        ranges[cid] = layer_stats(w)
    return ranges


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_all_clusters(
    cluster_models: Dict[int, Weights],
    clients: Dict[int, ClientData],
) -> Dict[int, float]:
    """
    For each client, find its best cluster model and compute test accuracy.
    Returns {uid: accuracy}.
    """
    per_client_acc = {}
    for uid, cd in clients.items():
        # Pick best model for this client
        best_cid = assign_cluster(cd, cluster_models)
        w  = cluster_models[best_cid]
        X, y = cd["X_test"], cd["y_test"]
        from models.mlp import predict
        if len(X) == 0:
            per_client_acc[uid] = float('nan')
            continue
        acc = float((predict(w, X) == y).mean())
        per_client_acc[uid] = acc
    return per_client_acc


def mean_accuracy(
    cluster_models: Dict[int, Weights],
    clients: Dict[int, ClientData],
) -> float:
    """Global mean test accuracy across all clients."""
    accs = evaluate_all_clusters(cluster_models, clients)
    return float(np.mean(list(accs.values())))


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.preprocess import prepare_data

    clients, nf, nc, nk = prepare_data()
    models = init_cluster_models(m=nk, n_features=nf, n_classes=nc)

    # One round of plain FedAvg (no LDP)
    updates = []
    for uid, cd in clients.items():
        cid, w_local = client_update(cd, models, lr=0.001, local_epochs=3)
        updates.append((cid, w_local))

    models = server_aggregate(updates, models)
    acc = mean_accuracy(models, clients)
    print(f"After 1 round (no LDP): mean test accuracy = {acc:.4f}")
