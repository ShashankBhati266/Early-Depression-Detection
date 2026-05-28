"""
run_acsfl.py
------------
Full ACS-FL training loop combining:
  • Clustered FL with IFCA-style cluster assignment
  • Adaptive layer-wise LDP perturbation (Algorithm 2)
  • Parameter shuffling (Algorithm 3)
  • DCT-based compression (Algorithm 4)

Replicates the four experiments from He et al. (IEEE IoT J. 2024)
on the StudentLife dataset:
  A. Effect of cluster size on accuracy
  B. Adaptive vs. fixed clipping range
  C. Generalised PM vs. Duchi's mechanism
  D. Effect of DCT compression ratio

Usage
-----
    python experiments/run_acsfl.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

# --- Project modules
from data.preprocess     import prepare_data
from models.mlp          import (init_weights, copy_weights, average_weights,
                                  local_train, evaluate_loss, layer_stats,
                                  predict, weight_dimension)
from ldp.perturbation    import (adaptive_perturbation, fixed_range_perturbation,
                                  duchi_perturbation, compute_layer_stats,
                                  compute_all_cluster_stats)
from fl.clustered_fl     import (init_cluster_models, client_update,
                                  server_aggregate, assign_cluster,
                                  compute_clipping_ranges, mean_accuracy)
from fl.shuffling        import IDDistributor, apply_shuffling
from fl.dct_compression  import dct_compress, server_reconstruct


# ===========================================================================
# Configuration dataclass (plain dict for simplicity)
# ===========================================================================

def default_config() -> dict:
    return dict(
        # Data
        n_clusters       = 4,
        data_dir         = "data/raw",
        test_size        = 0.2,
        seed             = 42,
        # Model
        hidden           = 128,
        # Training
        lr               = 0.001,
        local_epochs     = 5,
        batch_size       = 16,
        n_rounds         = 100,
        # Privacy
        epsilon          = 9.0,
        # DCT
        compression_ratio= 1.0,       # 1.0 = no compression
        # Shuffling
        use_shuffling    = True,
        # Perturbation mode: 'adaptive' | 'fixed' | 'duchi' | 'none'
        perturbation_mode= "adaptive",
        # Fixed-range params (used when perturbation_mode == 'fixed')
        fixed_c          = 0.0,
        fixed_r          = 0.015,
        # Logging
        log_every        = 5,
        verbose          = True,
    )


# ===========================================================================
# Single FL round
# ===========================================================================

def run_one_round(
    cluster_models: Dict[int, dict],
    clients: Dict[int, dict],
    clipping_ranges: Dict[int, Dict[str, Tuple[float, float]]],
    distributor: IDDistributor,
    previous_models: Optional[Dict[int, dict]],
    cfg: dict,
    round_idx: int,
    rng: np.random.Generator,
) -> Tuple[Dict[int, dict], List[Tuple[int, dict]]]:
    """
    Execute one ACS-FL round.  Returns (updated_cluster_models, raw_updates).
    """
    all_updates_for_shuffling: List[Tuple[int, dict]] = []   # (cid, perturbed_w)
    # Bucket for plain FedAvg path (when shuffling disabled)
    plain_buckets: Dict[int, List[Tuple[int, dict]]] = {cid: [] for cid in cluster_models}
    # For DCT path per cluster: List[perturbed_weights]
    dct_buckets: Dict[int, List[dict]] = {cid: [] for cid in cluster_models}

    for uid, cd in clients.items():
        client_seed = int(rng.integers(0, 2**31)) + uid

        # ── 1. Cluster assignment
        cid = assign_cluster(cd, cluster_models)

        # ── 2. Local training
        w_local = local_train(
            weights    = cluster_models[cid],
            X          = cd["X_train"],
            y          = cd["y_train"],
            lr         = cfg["lr"],
            epochs     = cfg["local_epochs"],
            batch_size = cfg["batch_size"],
            seed       = client_seed,
        )

        # ── 3. LDP perturbation
        mode = cfg["perturbation_mode"]
        if mode == "adaptive":
            stats  = clipping_ranges.get(cid, compute_layer_stats(w_local))
            w_pert = adaptive_perturbation(
                w_local, stats, cfg["epsilon"], seed=client_seed
            )
        elif mode == "fixed":
            w_pert = fixed_range_perturbation(
                w_local, cfg["epsilon"],
                c_fixed=cfg["fixed_c"], r_fixed=cfg["fixed_r"],
                seed=client_seed,
            )
        elif mode == "duchi":
            w_pert = duchi_perturbation(w_local, cfg["epsilon"], seed=client_seed)
        else:   # "none" – no perturbation (noise-free upper bound)
            w_pert = w_local

        # ── 4. DCT compression  (if η < 1)
        if cfg["compression_ratio"] < 1.0:
            w_pert, win_info = dct_compress(
                w_pert, cfg["compression_ratio"], round_number=round_idx
            )
            # Store compressed update + window info for server reconstruction
            dct_buckets[cid].append((w_pert, win_info))
        else:
            plain_buckets[cid].append(w_pert)

        all_updates_for_shuffling.append((cid, w_pert))

    # ── 5. Server aggregation
    if cfg["use_shuffling"] and cfg["compression_ratio"] >= 1.0:
        # Shuffling path  (Algorithm 3)
        distributor.refresh(seed=int(rng.integers(0, 2**31)))
        template = list(cluster_models.values())[0]
        new_models = apply_shuffling(
            all_updates_for_shuffling, distributor, template,
            previous_weights=previous_models,
            T=cfg["n_rounds"], seed=int(rng.integers(0, 2**31)),
        )
        # Fill in any clusters that received no updates
        for cid in cluster_models:
            if cid not in new_models:
                new_models[cid] = copy_weights(cluster_models[cid])

    elif cfg["compression_ratio"] < 1.0:
        # DCT reconstruction path  (Algorithm 4)
        template  = list(cluster_models.values())[0]
        new_models = {}
        for cid in cluster_models:
            prev = previous_models.get(cid) if previous_models else None
            if dct_buckets[cid]:
                new_models[cid] = server_reconstruct(
                    dct_buckets[cid], template, prev
                )
            else:
                new_models[cid] = copy_weights(cluster_models[cid])

    else:
        # Plain FedAvg aggregation (no shuffling, no compression)
        updates_flat = []
        for cid, w_list in plain_buckets.items():
            for w in w_list:
                updates_flat.append((cid, w))
        new_models = server_aggregate(updates_flat, cluster_models)

    return new_models, all_updates_for_shuffling


# ===========================================================================
# Full training loop
# ===========================================================================

def train_acsfl(cfg: dict = None) -> Dict[str, List]:
    """
    Run the full ACS-FL training.

    Returns
    -------
    history dict with keys:
        round           : [0, 1, 2, …]
        mean_acc        : mean test accuracy across all clients
        cluster_acc     : {cluster_id: [accuracy per round]}
        privacy_budget  : cumulative privacy budget per round
    """
    if cfg is None:
        cfg = default_config()

    rng = np.random.default_rng(cfg["seed"])

    # ── Data
    clients, n_features, n_classes, n_clusters = prepare_data(
        data_dir   = cfg["data_dir"],
        n_clusters = cfg["n_clusters"],
        seed       = cfg["seed"],
    )
    cfg["n_clusters"] = n_clusters    # update in case prepare_data adjusted it

    # ── Initialise cluster models
    cluster_models = init_cluster_models(
        m          = n_clusters,
        n_features = n_features,
        hidden     = cfg["hidden"],
        n_classes  = n_classes,
        seed       = cfg["seed"],
    )

    # ── ID distributor for shuffling
    distributor = IDDistributor(n_clusters=n_clusters, seed=cfg["seed"])

    # ── Logging
    history: Dict[str, list] = dict(
        round       = [],
        mean_acc    = [],
        cluster_acc = {cid: [] for cid in cluster_models},
        privacy_budget = [],
    )

    # Cumulative privacy budget: ε × T × d  (composition theorem)
    d             = weight_dimension(list(cluster_models.values())[0])
    total_budget  = 0.0
    previous_models = None

    if cfg["verbose"]:
        mode_str = cfg["perturbation_mode"]
        eta_str  = f"η={cfg['compression_ratio']}"
        print(f"\n{'='*60}")
        print(f" ACS-FL Training")
        print(f" Clients={len(clients)} | Clusters={n_clusters} | "
              f"Rounds={cfg['n_rounds']}")
        print(f" Perturbation={mode_str} | ε={cfg['epsilon']} | {eta_str}")
        print(f" Shuffling={'ON' if cfg['use_shuffling'] else 'OFF'}")
        print(f"{'='*60}")

    for t in range(cfg["n_rounds"]):
        # Server computes per-cluster clipping ranges (broadcast to clients)
        clipping_ranges = compute_clipping_ranges(cluster_models)

        # Run one federated round
        cluster_models, _ = run_one_round(
            cluster_models  = cluster_models,
            clients         = clients,
            clipping_ranges = clipping_ranges,
            distributor     = distributor,
            previous_models = previous_models,
            cfg             = cfg,
            round_idx       = t,
            rng             = rng,
        )
        previous_models = {cid: copy_weights(w)
                           for cid, w in cluster_models.items()}

        # ── Privacy budget accumulation
        if cfg["perturbation_mode"] != "none":
            # Each round each weight is perturbed once with budget ε
            # Budget per round = ε × d (composition)
            total_budget += cfg["epsilon"] * d * cfg["compression_ratio"]
        history["privacy_budget"].append(total_budget)

        # ── Evaluate
        if (t + 1) % cfg["log_every"] == 0 or t == cfg["n_rounds"] - 1:
            per_client_acc = {}
            for uid, cd in clients.items():
                best_cid = assign_cluster(cd, cluster_models)
                preds    = predict(cluster_models[best_cid], cd["X_test"])
                per_client_acc[uid] = float((preds == cd["y_test"]).mean())

            # Mean per cluster
            cluster_client_map: Dict[int, list] = {cid: [] for cid in cluster_models}
            for uid, cd in clients.items():
                cid = assign_cluster(cd, cluster_models)
                cluster_client_map[cid].append(per_client_acc[uid])

            vals = [v for v in per_client_acc.values() if not np.isnan(v)]
            m_acc = float(np.mean(vals)) if vals else 0.0

            history["round"].append(t + 1)
            history["mean_acc"].append(m_acc)
            for cid in cluster_models:
                c_accs = cluster_client_map.get(cid, [0.0])
                history["cluster_acc"][cid].append(
                    float(np.mean(c_accs)) if c_accs else 0.0
                )

            if cfg["verbose"]:
                c_str = " | ".join(
                    f"C{cid}={np.mean(cluster_client_map.get(cid,[0])):.3f}"
                    for cid in sorted(cluster_models)
                )
                print(f"[Round {t+1:3d}] mean_acc={m_acc:.4f} | {c_str} | "
                      f"budget={total_budget:.1f}")

    return history


# ===========================================================================
# Experiment runners (mirror Sections IV-A through IV-D of the paper)
# ===========================================================================

def experiment_cluster_size(n_rounds: int = 100) -> Dict:
    """Section IV-A: effect of cluster / client count on accuracy."""
    print("\n" + "="*60)
    print("EXPERIMENT A: Cluster Size Effect")
    print("="*60)
    results = {}
    for n_clusters in [2, 3, 4]:
        cfg = default_config()
        cfg.update(n_clusters=n_clusters, n_rounds=n_rounds,
                   perturbation_mode="adaptive", epsilon=9.0,
                   compression_ratio=1.0, use_shuffling=True,
                   log_every=10, verbose=True)
        print(f"\n--- n_clusters={n_clusters} ---")
        results[n_clusters] = train_acsfl(cfg)
    return results


def experiment_adaptive_vs_fixed(n_rounds: int = 100) -> Dict:
    """Section IV-B: adaptive range vs. fixed clipping range."""
    print("\n" + "="*60)
    print("EXPERIMENT B: Adaptive vs Fixed LDP Range")
    print("="*60)
    results = {}
    for eps in [6.0, 10.0]:
        for mode in ["adaptive", "fixed"]:
            cfg = default_config()
            cfg.update(n_rounds=n_rounds, epsilon=eps,
                       perturbation_mode=mode,
                       compression_ratio=1.0, use_shuffling=False,
                       log_every=10, verbose=True)
            key = f"eps={eps}_mode={mode}"
            print(f"\n--- ε={eps}, mode={mode} ---")
            results[key] = train_acsfl(cfg)
    return results


def experiment_gpm_vs_duchi(n_rounds: int = 50) -> Dict:
    """Section IV-C: generalised PM vs Duchi's mechanism."""
    print("\n" + "="*60)
    print("EXPERIMENT C: Generalised PM vs Duchi")
    print("="*60)
    results = {}
    for eps in [7.0, 8.0]:
        for mode in ["adaptive", "duchi"]:
            cfg = default_config()
            cfg.update(n_rounds=n_rounds, epsilon=eps,
                       perturbation_mode=mode,
                       compression_ratio=1.0, use_shuffling=False,
                       log_every=5, verbose=True)
            key = f"eps={eps}_mode={mode}"
            print(f"\n--- ε={eps}, mode={mode} ---")
            results[key] = train_acsfl(cfg)
    return results


def experiment_dct_compression(n_rounds: int = 50) -> Dict:
    """Section IV-D: DCT compression ratio effect."""
    print("\n" + "="*60)
    print("EXPERIMENT D: DCT Compression Ratios")
    print("="*60)
    results = {}
    for eps in [1.0, 2.0]:
        for eta in [0.25, 0.50, 0.70, 0.90, 1.00]:
            cfg = default_config()
            cfg.update(n_rounds=n_rounds, epsilon=eps,
                       perturbation_mode="adaptive",
                       compression_ratio=eta, use_shuffling=False,
                       log_every=5, verbose=True)
            key = f"eps={eps}_eta={eta}"
            print(f"\n--- ε={eps}, η={eta} ---")
            results[key] = train_acsfl(cfg)
    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACS-FL on StudentLife dataset")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["all", "A", "B", "C", "D", "quick"],
        help="Which experiment to run (default: all)"
    )
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of rounds")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override epsilon (privacy budget)")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Directory to save JSON results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.exp == "quick":
        # Fast sanity check: one round, all modes
        print("\n[QUICK TEST] Running a 10-round ACS-FL sanity check …")
        cfg = default_config()
        cfg.update(n_rounds=10, log_every=2, verbose=True,
                   epsilon=args.epsilon or 9.0)
        h = train_acsfl(cfg)
        print(f"\nFinal mean accuracy: {h['mean_acc'][-1]:.4f}")

    else:
        rounds_override = args.rounds

        if args.exp in ("all", "A"):
            r = experiment_cluster_size(n_rounds=rounds_override or 100)
            with open(f"{args.outdir}/exp_A_cluster_size.json", "w") as f:
                json.dump({str(k): v["mean_acc"] for k, v in r.items()}, f, indent=2)

        if args.exp in ("all", "B"):
            r = experiment_adaptive_vs_fixed(n_rounds=rounds_override or 100)
            with open(f"{args.outdir}/exp_B_adaptive_vs_fixed.json", "w") as f:
                json.dump({k: v["mean_acc"] for k, v in r.items()}, f, indent=2)

        if args.exp in ("all", "C"):
            r = experiment_gpm_vs_duchi(n_rounds=rounds_override or 50)
            with open(f"{args.outdir}/exp_C_gpm_vs_duchi.json", "w") as f:
                json.dump({k: v["mean_acc"] for k, v in r.items()}, f, indent=2)

        if args.exp in ("all", "D"):
            r = experiment_dct_compression(n_rounds=rounds_override or 50)
            with open(f"{args.outdir}/exp_D_dct_compression.json", "w") as f:
                json.dump({k: v["mean_acc"] for k, v in r.items()}, f, indent=2)

    print("\nDone.")
