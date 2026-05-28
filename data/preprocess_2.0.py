"""
preprocess.py  —  Depression Detection edition
------------------------------------------------
Loads and preprocesses the StudentLife (Dartmouth) dataset for
depression detection using the PHQ-9 scale as the target label.

PHQ-9 severity buckets (standard clinical thresholds)
──────────────────────────────────────────────────────
  0 –  4  →  class 0  (minimal / none)
  5 –  9  →  class 1  (mild)
 10 – 14  →  class 2  (moderate)
 15 – 27  →  class 3  (moderately-severe / severe)

Features used (all passive behavioural signals — PHQ-9 is the TARGET,
not a predictor)
────────────────────────────────────────────────────────────────────────
  sleep_hours      avg nightly sleep hours
  sleep_onset      time of sleep onset (24-h decimal, e.g. 23.5 = 11:30 pm)
  activity_level   normalised step-count proxy (0–1)
  sedentary_mins   daily sedentary minutes
  phone_usage      daily screen-on minutes
  social_duration  duration of social-proximity events (min/day)
  conversation_ct  number of detected conversation episodes
  mood             EMA self-report (1=very bad … 5=very good)
  energy_level     EMA self-report (1=very low … 5=very high)
  isolation_score  derived: low social_duration + low conversation_ct
  sleep_irregularity  std-dev of sleep_onset across the week

Pipeline
────────
  1. Load real CSV  or  auto-generate synthetic StudentLife-like data
  2. Derive depression label from PHQ-9 score
  3. Engineer per-student weekly aggregates
  4. Normalise features to [-1, 1]  (required by LDP clipping math)
  5. Cluster students by behaviour via K-Means  (non-IID FL simulation)
  6. Split each student's data into train / test
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


# ════════════════════════════════════════════════════════════════════════════
# 0.  Constants
# ════════════════════════════════════════════════════════════════════════════

# PHQ-9 clinical severity thresholds
# 0=minimal(0-4), 1=mild(5-9), 2=moderate(10-14), 3=severe(15-27)
PHQ_BINS   = [0, 5, 10, 15, 28]
PHQ_LABELS = [0, 1, 2, 3]          # 4-class depression severity

# Behavioural features (PHQ-9 score is NOT here — it is the target)
FEATURE_COLS = [
    "sleep_hours",
    "sleep_onset",
    "activity_level",
    "sedentary_mins",
    "phone_usage",
    "social_duration",
    "conversation_ct",
    "mood",
    "energy_level",
    "isolation_score",
    "sleep_irregularity",
]

LABEL_COL   = "depression_label"    # 0 / 1 / 2 / 3
PHQ_COL     = "phq_score"


# ════════════════════════════════════════════════════════════════════════════
# 1.  PHQ-9 → depression label
# ════════════════════════════════════════════════════════════════════════════

def phq_to_label(phq: float) -> int:
    """
    Map a PHQ-9 score (0-27) to a 4-class depression severity label.

    Class  PHQ-9 range  Clinical meaning
    ──────────────────────────────────────
      0      0 –  4     Minimal / none
      1      5 –  9     Mild
      2     10 – 14     Moderate
      3     15 – 27     Moderately-severe / Severe
    """
    phq = int(np.clip(round(phq), 0, 27))
    if phq < 5:
        return 0
    elif phq < 10:
        return 1
    elif phq < 15:
        return 2
    else:
        return 3


# ════════════════════════════════════════════════════════════════════════════
# 2.  Synthetic data generator  (auto-used when real CSV is absent)
# ════════════════════════════════════════════════════════════════════════════

def generate_synthetic_studentlife(
    n_students: int = 48,
    n_weeks:    int = 10,
    n_clusters: int = 4,
    seed:       int = 42,
) -> pd.DataFrame:
    """
    Synthesise a StudentLife-like dataset with realistic behavioural
    profiles tied to depression severity.

    Each cluster represents a distinct depression severity group so that
    the FL heterogeneity is clinically meaningful:

    Cluster 0 — Minimal depression   (PHQ ~4,  good sleep, active, social)
    Cluster 1 — Mild depression       (PHQ ~8,  moderate activity, some isolation)
    Cluster 2 — Moderate depression   (PHQ ~12, poor sleep, sedentary, withdrawn)
    Cluster 3 — Severe depression     (PHQ ~20, very poor sleep, inactive, isolated)
    """
    rng = np.random.default_rng(seed)

    # Profile: (sleep_mu, sleep_sig, sleep_onset_mu, act_mu, sed_mu,
    #           phone_mu, social_mu, conv_mu, mood_mu, energy_mu, phq_mu)
    profiles = {
        0: dict(sleep_mu=7.8, sleep_sig=0.4, onset_mu=23.0,  # good sleeper, early
                act_mu=0.72, sed_mu=200, phone_mu=90,
                social_mu=140, conv_mu=8,
                mood_mu=4.2, energy_mu=4.0, phq_mu=3),
        1: dict(sleep_mu=6.8, sleep_sig=0.6, onset_mu=23.8,
                act_mu=0.50, sed_mu=280, phone_mu=160,
                social_mu=90,  conv_mu=5,
                mood_mu=3.2, energy_mu=3.0, phq_mu=7),
        2: dict(sleep_mu=5.8, sleep_sig=0.9, onset_mu=1.2,   # late nights
                act_mu=0.30, sed_mu=380, phone_mu=240,
                social_mu=45,  conv_mu=3,
                mood_mu=2.4, energy_mu=2.0, phq_mu=12),
        3: dict(sleep_mu=4.8, sleep_sig=1.1, onset_mu=2.5,   # very late / fragmented
                act_mu=0.15, sed_mu=470, phone_mu=320,
                social_mu=15,  conv_mu=1,
                mood_mu=1.6, energy_mu=1.2, phq_mu=20),
    }

    students_per_cluster = n_students // n_clusters
    rows = []
    onset_history: Dict[int, list] = {uid: [] for uid in range(n_students)}

    for uid in range(n_students):
        c = min(uid // students_per_cluster, n_clusters - 1)
        p = profiles[c]

        for week in range(1, n_weeks + 1):
            sleep  = float(np.clip(rng.normal(p["sleep_mu"], p["sleep_sig"]),
                                   2.0, 12.0))
            onset  = float(np.clip(rng.normal(p["onset_mu"], 0.8),
                                   20.0, 5.0 + 24) % 24)   # wrap 24-h clock
            act    = float(np.clip(rng.normal(p["act_mu"],  0.12), 0.0, 1.0))
            sed    = float(np.clip(rng.normal(p["sed_mu"],  60),   0.0, 720))
            phone  = float(np.clip(rng.normal(p["phone_mu"], 50),  0.0, 600))
            social = float(np.clip(rng.normal(p["social_mu"], 30), 0.0, 360))
            conv   = int  (np.clip(rng.normal(p["conv_mu"],   2),  0,   20 ))
            mood   = int  (np.clip(round(rng.normal(p["mood_mu"],   0.6)),
                                   1, 5))
            energy = int  (np.clip(round(rng.normal(p["energy_mu"], 0.6)),
                                   1, 5))
            phq    = int  (np.clip(round(rng.normal(p["phq_mu"],    3)),
                                   0, 27))

            # Derived features
            isolation = float(
                1.0 - (social / 360.0) * 0.5 - (conv / 20.0) * 0.5
            )
            onset_history[uid].append(onset)
            sleep_irreg = float(np.std(onset_history[uid]))

            rows.append(dict(
                uid=uid, week=week,
                sleep_hours=sleep, sleep_onset=onset,
                activity_level=act, sedentary_mins=sed,
                phone_usage=phone, social_duration=social,
                conversation_ct=conv, mood=mood, energy_level=energy,
                isolation_score=isolation, sleep_irregularity=sleep_irreg,
                phq_score=phq, cluster_true=c,
                depression_label=phq_to_label(phq),
            ))

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 3.  Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_or_generate(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load real StudentLife CSV from *data_dir/studentlife.csv*.
    Falls back to synthetic generation if the file is not found.

    Real CSV must contain at minimum:
        uid, week, phq_score
    Plus as many of the FEATURE_COLS as are available.
    Any missing feature column is filled with 0.
    """
    csv_path = os.path.join(data_dir, "studentlife.csv")
    if os.path.exists(csv_path):
        print(f"[preprocess] Loading real data from {csv_path}")
        df = pd.read_csv(csv_path)

        # ── Derive depression label from PHQ-9
        if PHQ_COL not in df.columns:
            raise ValueError(
                f"Real CSV must contain a '{PHQ_COL}' column for depression detection."
            )
        df[LABEL_COL] = df[PHQ_COL].apply(phq_to_label)

        # ── Derive isolation_score if raw columns exist
        if "isolation_score" not in df.columns:
            soc = df.get("social_duration", pd.Series(0, index=df.index))
            conv = df.get("conversation_ct", pd.Series(0, index=df.index))
            df["isolation_score"] = (
                1.0 - (soc / soc.max().clip(1)) * 0.5
                    - (conv / conv.max().clip(1)) * 0.5
            )

        # ── Derive sleep_irregularity (per-student std of sleep_onset)
        if "sleep_irregularity" not in df.columns and "sleep_onset" in df.columns:
            df["sleep_irregularity"] = df.groupby("uid")["sleep_onset"].transform("std").fillna(0)
        elif "sleep_irregularity" not in df.columns:
            df["sleep_irregularity"] = 0.0

        # ── Fill any missing feature columns with 0
        for col in FEATURE_COLS:
            if col not in df.columns:
                print(f"[preprocess] WARNING: column '{col}' not found – filling with 0")
                df[col] = 0.0

        if "cluster_true" not in df.columns:
            df["cluster_true"] = -1

    else:
        print("[preprocess] Real data not found – using synthetic StudentLife data "
              "(depression-detection mode).")
        df = generate_synthetic_studentlife()

    return df


# ════════════════════════════════════════════════════════════════════════════
# 4.  Feature normalisation
# ════════════════════════════════════════════════════════════════════════════

def normalise_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale all FEATURE_COLS to [-1, 1].
    This is required by the LDP clipping math (weights must be bounded).
    PHQ-9 score and depression label are NOT normalised.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS].fillna(0))
    return df, scaler


# ════════════════════════════════════════════════════════════════════════════
# 5.  K-Means clustering  (simulates FL non-IID heterogeneity)
# ════════════════════════════════════════════════════════════════════════════

def assign_clusters_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster students by their mean behavioural feature vector.

    In the depression-detection context this naturally groups students by
    severity level (e.g. severe-depressed students share similar sleep and
    activity patterns), creating a clinically meaningful non-IID split.
    """
    per_student = df.groupby("uid")[FEATURE_COLS].mean()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    per_student["cluster"] = km.fit_predict(per_student.values)
    df = df.merge(per_student[["cluster"]], on="uid")
    return df, km


# ════════════════════════════════════════════════════════════════════════════
# 6.  Per-client dataset splits
# ════════════════════════════════════════════════════════════════════════════

ClientData = Dict[str, np.ndarray]


def build_client_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, ClientData], int, int]:
    """
    Build per-student (per-client) train / test splits.

    Returns
    -------
    clients    : {uid: {X_train, y_train, X_test, y_test, cluster,
                        phq_mean, depression_class_dist}}
    n_features : int  (= len(FEATURE_COLS))
    n_classes  : int  (= 4  — PHQ severity levels)
    """
    clients: Dict[int, ClientData] = {}

    for uid, group in df.groupby("uid"):
        X       = group[FEATURE_COLS].values.astype(np.float32)
        y       = group[LABEL_COL].values.astype(np.int64)
        cluster = int(group["cluster"].iloc[0])
        phq_mean = float(group[PHQ_COL].mean())

        # Class distribution (useful for reporting)
        class_dist = {int(k): int(v)
                      for k, v in zip(*np.unique(y, return_counts=True))}

        # If too few samples, tile to allow a split
        if len(X) < 4:
            X = np.tile(X, (4, 1))
            y = np.tile(y, 4)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=None
        )

        clients[int(uid)] = dict(
            X_train            = X_tr,
            y_train            = y_tr,
            X_test             = X_te,
            y_test             = y_te,
            cluster            = cluster,
            phq_mean           = phq_mean,
            depression_class_dist = class_dist,
        )

    n_features = len(FEATURE_COLS)
    n_classes  = 4      # always 4 PHQ severity levels
    return clients, n_features, n_classes


# ════════════════════════════════════════════════════════════════════════════
# 7.  One-stop entry point
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(
    data_dir:   str   = "data/raw",
    n_clusters: int   = 4,
    test_size:  float = 0.2,
    seed:       int   = 42,
) -> Tuple[Dict[int, ClientData], int, int, int]:
    """
    Full preprocessing pipeline for depression detection.

    Returns
    -------
    clients     : per-client dict (X_train, y_train, X_test, y_test,
                  cluster, phq_mean, depression_class_dist)
    n_features  : 11   (behavioural features)
    n_classes   : 4    (PHQ-9 severity levels)
    n_clusters  : int  (as requested)
    """
    df = load_or_generate(data_dir)
    df, _scaler = normalise_features(df)
    df, _km     = assign_clusters_kmeans(df, n_clusters=n_clusters, seed=seed)

    clients, n_features, n_classes = build_client_datasets(
        df, test_size=test_size, seed=seed
    )

    # ── Summary statistics
    cluster_counts: Dict[int, int] = {}
    cluster_phq:    Dict[int, list] = {}
    for cd in clients.values():
        c = cd["cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        cluster_phq.setdefault(c, []).append(cd["phq_mean"])

    label_names = {0: "Minimal(0-4)", 1: "Mild(5-9)",
                   2: "Moderate(10-14)", 3: "Severe(15-27)"}

    print(f"\n[preprocess] Task          : Depression Detection (PHQ-9 4-class)")
    print(f"[preprocess] Clients       : {len(clients)}")
    print(f"[preprocess] Features      : {n_features}  {FEATURE_COLS}")
    print(f"[preprocess] Classes       : {n_classes}   {label_names}")
    print(f"[preprocess] Clusters      : {n_clusters}")
    print(f"[preprocess] Clients/cluster: {dict(sorted(cluster_counts.items()))}")
    print(f"[preprocess] Mean PHQ-9 per cluster:")
    for c in sorted(cluster_phq):
        mean_phq = np.mean(cluster_phq[c])
        label    = label_names.get(phq_to_label(mean_phq), "?")
        print(f"             Cluster {c}: PHQ={mean_phq:.1f}  → {label}")
    print()

    return clients, n_features, n_classes, n_clusters


# ════════════════════════════════════════════════════════════════════════════
# 8.  Smoke-test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    clients, nf, nc, nk = prepare_data()

    uid0 = list(clients.keys())[0]
    cd   = clients[uid0]
    print(f"Client {uid0}:")
    print(f"  cluster           = {cd['cluster']}")
    print(f"  mean PHQ-9        = {cd['phq_mean']:.1f}")
    print(f"  class distribution= {cd['depression_class_dist']}")
    print(f"  X_train shape     = {cd['X_train'].shape}")
    print(f"  X_test  shape     = {cd['X_test'].shape}")
    print(f"  y_train sample    = {cd['y_train'][:5]}")

    # Verify PHQ → label mapping
    print("\nPHQ-9 → label mapping check:")
    for phq, expected in [(2, 0), (7, 1), (12, 2), (18, 3), (25, 3)]:
        got = phq_to_label(phq)
        status = "✓" if got == expected else "✗"
        print(f"  PHQ={phq:2d} → label={got}  {status}")
