"""
preprocess.py
-------------
Loads and preprocesses the StudentLife (Dartmouth) dataset for use in
Clustered Federated Learning (ACS-FL).

Each student → one FL client with their own local dataset.
We:
  1. Load raw CSV(s)
  2. Engineer per-student feature vectors
  3. Normalise features to [-1, 1]  (required by LDP clipping math)
  4. Assign cluster labels via K-Means (simulating non-IID heterogeneity)
  5. Split each student's data into train / test

If the real StudentLife CSVs are not yet downloaded, a synthetic dataset
that mimics its structure is generated automatically so the whole pipeline
can be exercised immediately.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 1.  Synthetic data generator (fallback when real data is absent)
# ---------------------------------------------------------------------------

def generate_synthetic_studentlife(
    n_students: int = 48,
    n_weeks: int = 10,
    n_clusters: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mirrors the StudentLife structure.

    Features per student-week row
    ─────────────────────────────
    uid              student id  (0 … n_students-1)
    week             week number (1 … n_weeks)
    sleep_hours      avg nightly sleep
    activity_level   step-count proxy  (0-1 normalised)
    phone_usage      daily screen-on minutes
    social_duration  duration of social-proximity events (min/day)
    stress_level     EMA self-report  (0 = low … 3 = high)
    mood             EMA self-report  (1 = very bad … 5 = very good)
    deadline_count   number of academic deadlines that week
    phq_score        PHQ-9 depression proxy (0-27)
    cluster_true     ground-truth cluster (used only for evaluation)
    label            binary stress: 0 = low (stress<2), 1 = high (stress≥2)
    """
    rng = np.random.default_rng(seed)

    # Each cluster has a distinct behavioural profile
    cluster_profiles = {
        0: dict(sleep_mu=7.5, sleep_sig=0.5, act_mu=0.7, act_sig=0.1,
                mood_mu=4.0, mood_sig=0.4, stress_mu=1.0, phq_mu=5),
        1: dict(sleep_mu=6.0, sleep_sig=0.7, act_mu=0.4, act_sig=0.15,
                mood_mu=3.0, mood_sig=0.5, stress_mu=2.0, phq_mu=12),
        2: dict(sleep_mu=5.0, sleep_sig=1.0, act_mu=0.3, act_sig=0.2,
                mood_mu=2.5, mood_sig=0.6, stress_mu=2.5, phq_mu=18),
        3: dict(sleep_mu=8.0, sleep_sig=0.4, act_mu=0.8, act_sig=0.08,
                mood_mu=4.5, mood_sig=0.3, stress_mu=0.8, phq_mu=3),
    }

    students_per_cluster = n_students // n_clusters
    rows = []
    for uid in range(n_students):
        c = uid // students_per_cluster
        c = min(c, n_clusters - 1)
        p = cluster_profiles[c]
        for week in range(1, n_weeks + 1):
            sleep   = rng.normal(p["sleep_mu"],  p["sleep_sig"])
            act     = np.clip(rng.normal(p["act_mu"],  p["act_sig"]), 0, 1)
            phone   = rng.uniform(60, 300)
            social  = rng.uniform(10, 180)
            stress  = int(np.clip(round(rng.normal(p["stress_mu"], 0.6)), 0, 3))
            mood    = int(np.clip(round(rng.normal(p["mood_mu"],   p["mood_sig"])), 1, 5))
            dead    = rng.integers(0, 5)
            phq     = int(np.clip(round(rng.normal(p["phq_mu"], 3)), 0, 27))
            rows.append(dict(uid=uid, week=week, sleep_hours=sleep,
                             activity_level=act, phone_usage=phone,
                             social_duration=social, stress_level=stress,
                             mood=mood, deadline_count=dead,
                             phq_score=phq, cluster_true=c,
                             label=int(stress >= 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2.  Feature engineering helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "sleep_hours", "activity_level", "phone_usage",
    "social_duration", "deadline_count", "phq_score", "mood",
]
LABEL_COL = "label"


def load_or_generate(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Tries to load real StudentLife CSVs from *data_dir*.
    Falls back to synthetic generation if files are not found.

    Expected real CSV columns (subset):
        uid, week, sleep_hours, activity_level, phone_usage,
        social_duration, stress_level, mood, deadline_count, phq_score
    """
    csv_path = os.path.join(data_dir, "studentlife.csv")
    if os.path.exists(csv_path):
        print(f"[preprocess] Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)
        # Derive binary stress label if not present
        if LABEL_COL not in df.columns:
            df[LABEL_COL] = (df["stress_level"] >= 2).astype(int)
        if "cluster_true" not in df.columns:
            df["cluster_true"] = -1          # unknown; will be set by K-Means
    else:
        print("[preprocess] Real data not found – using synthetic StudentLife data.")
        df = generate_synthetic_studentlife()
    return df


def normalise_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scales all feature columns to [-1, 1] in-place."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].fillna(0))
    return df, scaler


def assign_clusters_kmeans(
    df: pd.DataFrame, n_clusters: int = 4, seed: int = 42
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Runs K-Means on per-student mean feature vectors to assign cluster ids.
    Returns the augmented dataframe and the fitted KMeans object.
    """
    per_student = df.groupby("uid")[FEATURE_COLS].mean()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    per_student["cluster"] = km.fit_predict(per_student)
    df = df.merge(per_student[["cluster"]], on="uid")
    return df, km


# ---------------------------------------------------------------------------
# 3.  Build per-client (per-student) data splits
# ---------------------------------------------------------------------------

ClientData = Dict[str, np.ndarray]   # keys: X_train, y_train, X_test, y_test


def build_client_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, ClientData], int, int]:
    """
    Returns
    -------
    clients     : {uid: {X_train, y_train, X_test, y_test, cluster}}
    n_features  : number of input features
    n_classes   : number of output classes
    """
    clients: Dict[int, ClientData] = {}
    for uid, group in df.groupby("uid"):
        X = group[FEATURE_COLS].values.astype(np.float32)
        y = group[LABEL_COL].values.astype(np.int64)
        cluster = int(group["cluster"].iloc[0])

        if len(X) < 4:
            # Too few samples – duplicate rows to allow a split
            X = np.tile(X, (4, 1))
            y = np.tile(y, 4)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )
        clients[int(uid)] = dict(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            cluster=cluster,
        )

    n_features = len(FEATURE_COLS)
    n_classes  = int(df[LABEL_COL].nunique())
    return clients, n_features, n_classes


# ---------------------------------------------------------------------------
# 4.  One-stop preprocessing function
# ---------------------------------------------------------------------------

def prepare_data(
    data_dir: str = "data/raw",
    n_clusters: int = 4,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, ClientData], int, int, int]:
    """
    Full pipeline.

    Returns
    -------
    clients     : per-client train/test splits + cluster assignment
    n_features  : int
    n_classes   : int
    n_clusters  : int
    """
    df = load_or_generate(data_dir)
    df, _scaler = normalise_features(df)
    df, _km     = assign_clusters_kmeans(df, n_clusters=n_clusters, seed=seed)
    clients, n_features, n_classes = build_client_datasets(
        df, test_size=test_size, seed=seed
    )

    # Summary
    cluster_counts = {}
    for cd in clients.values():
        c = cd["cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    print(f"[preprocess] {len(clients)} clients | "
          f"{n_features} features | {n_classes} classes | "
          f"{n_clusters} clusters")
    print(f"[preprocess] Clients per cluster: {dict(sorted(cluster_counts.items()))}")
    return clients, n_features, n_classes, n_clusters


# ---------------------------------------------------------------------------
# 5.  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clients, nf, nc, nk = prepare_data()
    uid0 = list(clients.keys())[0]
    cd   = clients[uid0]
    print(f"\nClient {uid0}: cluster={cd['cluster']}  "
          f"train={cd['X_train'].shape}  test={cd['X_test'].shape}")
