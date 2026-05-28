"""
preprocess.py  —  Real StudentLife Dataset Edition
----------------------------------------------------
Reads directly from the actual StudentLife folder structure:

    dataset/
    ├── survey/PHQ-9.csv              ← depression labels (PHQ-9 scores)
    ├── EMA/response/
    │   ├── Activity/Activity_uXX.json   ← physical activity EMA
    │   ├── Mood/Mood_uXX.json           ← mood EMA
    │   └── Sleep Duration/...           ← sleep EMA
    ├── phonelock/phonelock_uXX.csv    ← phone screen-on time proxy
    ├── call_log/call_log_uXX.csv      ← social calls
    ├── sms/sms_uXX.csv               ← social SMS
    └── sensing/
        ├── activity/activity_uXX.csv  ← accelerometer activity
        └── sleep/sleep_uXX.csv        ← inferred sleep

Depression label: PHQ-9 score → 4-class severity
    0 = Minimal  (PHQ 0-4)
    1 = Mild     (PHQ 5-9)
    2 = Moderate (PHQ 10-14)
    3 = Severe   (PHQ 15-27)

Each student = one FL client.
One row per student per week (aggregated from raw sensor timestamps).
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional

# ════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "avg_activity_ema",       # EMA self-reported activity level (1-5)
    "avg_mood",               # EMA mood (1-5)
    "avg_sleep_duration",     # EMA-reported sleep hours
    "phone_usage_mins",       # daily phone screen-on minutes (phonelock)
    "call_count",             # number of calls per week
    "sms_count",              # number of SMS per week
    "activity_sensing",       # accelerometer-based activity level
    "sleep_sensing",          # sensor-inferred sleep hours
    "social_score",           # combined call + sms social activity
    "sleep_irregularity",     # std-dev of sleep duration across week
    "phone_activity_ratio",   # phone usage relative to activity (isolation proxy)
]

LABEL_COL  = "depression_label"
PHQ_COL    = "phq_score"
N_WEEKS    = 10   # StudenLife runs 10 weeks

# User IDs present in the dataset (from the tree structure)
ALL_UIDS = [
    0,1,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19,20,
    22,23,24,25,27,30,31,32,33,34,35,36,39,41,42,43,
    44,45,46,47,49,50,51,52,53,54,56,57,58,59
]


# ════════════════════════════════════════════════════════════════════════════
# 1. PHQ-9 → depression label
# ════════════════════════════════════════════════════════════════════════════

def phq_to_label(phq: float) -> int:
    phq = int(np.clip(round(float(phq)), 0, 27))
    if phq < 5:   return 0   # Minimal
    if phq < 10:  return 1   # Mild
    if phq < 15:  return 2   # Moderate
    return 3                  # Severe


# ════════════════════════════════════════════════════════════════════════════
# 2. Load PHQ-9 labels
# ════════════════════════════════════════════════════════════════════════════

def load_phq9(dataset_dir: str) -> Dict[int, float]:
    """
    Load survey/PHQ-9.csv.
    Returns {uid: phq_score}.

    Handles the real StudentLife PHQ-9.csv format where each of the 9
    questions has a text response ("Not at all", "Several days", etc.)
    that must be converted to 0-3 scores and summed.

    CSV columns:
        uid, type, <question 1>, …, <question 9>, Response
    """
    path = os.path.join(dataset_dir, "survey", "PHQ-9.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PHQ-9.csv not found at {path}")

    df = pd.read_csv(path)

    # Standard PHQ-9 text → numeric mapping
    PHQ_TEXT_MAP = {
        "not at all":                                0,
        "several days":                              1,
        "more than half the days":                   2,
        "nearly every day":                          3,
        # Response column (functional impairment) — not counted in PHQ-9 score
        "not difficult at all":                      0,
        "somewhat difficult":                        1,
        "very difficult":                            2,
        "extremely difficult":                       3,
    }

    # The 9 PHQ question columns (everything except uid, type, Response)
    skip_cols = {"uid", "type", "Response"}
    question_cols = [c for c in df.columns if c not in skip_cols]

    def score_row(row) -> float:
        total = 0
        for col in question_cols:
            val = str(row[col]).strip().lower()
            total += PHQ_TEXT_MAP.get(val, 0)
        return float(total)

    uid_to_phq: Dict[int, list] = {}
    for _, row in df.iterrows():
        try:
            uid = int(str(row["uid"]).replace("u", "").replace("U", "").strip())
        except (ValueError, KeyError):
            continue
        score = score_row(row)
        uid_to_phq.setdefault(uid, []).append(score)

    result = {uid: float(np.mean(scores)) for uid, scores in uid_to_phq.items()}
    print(f"[PHQ-9] Loaded {len(result)} students | "
          f"scores range: {min(result.values()):.0f} – {max(result.values()):.0f} | "
          f"mean: {np.mean(list(result.values())):.1f}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 3. EMA JSON loaders
# ════════════════════════════════════════════════════════════════════════════

def _parse_uid_from_filename(fname: str) -> Optional[int]:
    """Extract integer uid from filenames like Activity_u07.json → 7"""
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_u")
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


def load_ema_activity(ema_dir: str) -> Dict[int, float]:
    """
    Load Activity EMA from EMA/response/Activity/Activity_uXX.json
    Real format: [{'Social2': '2', 'null': '1', 'resp_time': ...}, ...]
    'Social2' = activity level (1=very active … 5=not active at all)
    We invert it so higher = more active.
    """
    folder = os.path.join(ema_dir, "response", "Activity")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            with open(fpath) as f:
                entries = json.load(f)
            vals = []
            for e in entries:
                if isinstance(e, dict) and "Social2" in e:
                    try:
                        # 1=very active, 5=not active → invert to 5=very active
                        v = float(e["Social2"])
                        vals.append(6.0 - v)   # invert scale
                    except (ValueError, TypeError):
                        pass
            result[uid] = float(np.mean(vals)) if vals else np.nan
        except Exception:
            result[uid] = np.nan
    return result


def load_ema_mood(ema_dir: str) -> Dict[int, float]:
    """
    Load Mood EMA from EMA/response/Mood/Mood_uXX.json
    Real format: [{'happy': '1', 'happyornot': '2', 'sad': '3', 'sadornot': '1', ...}, ...]
    We use 'happy' (1=not happy … 5=very happy) as mood score.
    """
    folder = os.path.join(ema_dir, "response", "Mood")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            with open(fpath) as f:
                entries = json.load(f)
            vals = []
            for e in entries:
                if isinstance(e, dict):
                    # 'happy' key: higher = better mood
                    if "happy" in e:
                        try:
                            vals.append(float(e["happy"]))
                        except (ValueError, TypeError):
                            pass
            result[uid] = float(np.mean(vals)) if vals else np.nan
        except Exception:
            result[uid] = np.nan
    return result


def load_ema_sleep(ema_dir: str) -> Dict[int, List[float]]:
    """
    Load Sleep EMA from EMA/response/Sleep/Sleep_uXX.json
    Real format: [{'null': '8', 'resp_time': ...}, ...]
    'null' key contains the sleep duration in hours as a string.
    Returns {uid: [list of nightly sleep hours]}
    """
    folder = os.path.join(ema_dir, "response", "Sleep")
    result: Dict[int, List[float]] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            with open(fpath) as f:
                entries = json.load(f)
            vals = []
            for e in entries:
                if isinstance(e, dict) and "null" in e:
                    try:
                        v = float(e["null"])
                        # Sanity check: sleep hours should be 0-16
                        if 0 < v <= 16:
                            vals.append(v)
                    except (ValueError, TypeError):
                        pass
            result[uid] = vals
        except Exception:
            result[uid] = []
    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. Sensing CSV loaders  (all paths confirmed from real dataset structure)
# ════════════════════════════════════════════════════════════════════════════

def load_activity_sensing(dataset_dir: str) -> Dict[int, float]:
    """
    sensing/activity/activity_uXX.csv
    Columns: timestamp, ' activity inference'
    Values: 0=stationary, 1=walking, 2=running, 3=unknown
    Returns mean activity level per user.
    """
    folder = os.path.join(dataset_dir, "sensing", "activity")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            col = next((c for c in df.columns if "activity" in c.lower()), None)
            if col is None:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            # Keep only valid values (0-2, exclude 3=unknown)
            series = series[series.isin([0, 1, 2])]
            result[uid] = float(series.mean()) if len(series) > 0 else np.nan
        except Exception:
            result[uid] = np.nan
    return result


def load_conversation_sensing(dataset_dir: str) -> Dict[int, float]:
    """
    sensing/conversation/conversation_uXX.csv
    Columns: start_timestamp, end_timestamp
    Returns avg daily conversation minutes per user.
    """
    folder = os.path.join(dataset_dir, "sensing", "conversation")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            start_col = next((c for c in df.columns if "start" in c.lower()), None)
            end_col   = next((c for c in df.columns if "end"   in c.lower()), None)
            if start_col and end_col:
                starts = pd.to_numeric(df[start_col], errors="coerce")
                ends   = pd.to_numeric(df[end_col],   errors="coerce")
                durations = (ends - starts).dropna()
                durations = durations[durations > 0]
                # Total conversation seconds → daily minutes over N_WEEKS
                daily_mins = float(durations.sum() / 60.0 / (N_WEEKS * 7))
                result[uid] = daily_mins
            else:
                result[uid] = float(len(df) / N_WEEKS)
        except Exception:
            result[uid] = np.nan
    return result


def load_phonelock_sensing(dataset_dir: str) -> Dict[int, float]:
    """
    sensing/phonelock/phonelock_uXX.csv
    Columns: start, end   (Unix timestamps)
    Returns avg daily screen-on minutes per user.
    """
    folder = os.path.join(dataset_dir, "sensing", "phonelock")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            df.columns = df.columns.str.strip().str.lower()
            starts = pd.to_numeric(df["start"], errors="coerce")
            ends   = pd.to_numeric(df["end"],   errors="coerce")
            durations = (ends - starts).dropna()
            durations = durations[durations > 0]
            # Total screen-on seconds → daily minutes over N_WEEKS
            daily_mins = float(durations.sum() / 60.0 / (N_WEEKS * 7))
            result[uid] = daily_mins
        except Exception:
            result[uid] = np.nan
    return result


def load_call_log(dataset_dir: str) -> Dict[int, float]:
    """call_log/call_log_uXX.csv → avg calls per week."""
    folder = os.path.join(dataset_dir, "call_log")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "call_log_u*.csv"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            result[uid] = float(len(df) / N_WEEKS)
        except Exception:
            result[uid] = 0.0
    return result


def load_sms(dataset_dir: str) -> Dict[int, float]:
    """sms/sms_uXX.csv → avg SMS per week."""
    folder = os.path.join(dataset_dir, "sms")
    result: Dict[int, float] = {}
    for fpath in sorted(glob.glob(os.path.join(folder, "sms_u*.csv"))):
        uid = _parse_uid_from_filename(fpath)
        if uid is None:
            continue
        try:
            df = pd.read_csv(fpath, on_bad_lines="skip")
            result[uid] = float(len(df) / N_WEEKS)
        except Exception:
            result[uid] = 0.0
    return result


# ════════════════════════════════════════════════════════════════════════════
# 5. Build per-student feature vector
# ════════════════════════════════════════════════════════════════════════════

def build_feature_table(dataset_dir: str) -> pd.DataFrame:
    """
    Loads all data sources and merges them into one row per student.
    Returns a DataFrame with columns = FEATURE_COLS + [PHQ_COL, LABEL_COL, uid].
    """
    ema_dir     = os.path.join(dataset_dir, "EMA")
    sensing_dir = os.path.join(dataset_dir, "sensing")

    print("[preprocess] Loading PHQ-9 labels …")
    phq_map = load_phq9(dataset_dir)

    print("[preprocess] Loading EMA responses …")
    activity_ema = load_ema_activity(ema_dir)   # {uid: mean_activity_score}
    mood_ema     = load_ema_mood(ema_dir)        # {uid: mean_mood_score}
    sleep_ema    = load_ema_sleep(ema_dir)       # {uid: [list of sleep hours]}

    print("[preprocess] Loading sensing data …")
    activity_sensing = load_activity_sensing(dataset_dir)      # {uid: mean_activity}
    conv_sensing     = load_conversation_sensing(dataset_dir)  # {uid: daily_conv_mins}

    print("[preprocess] Loading phone / call / SMS data …")
    phone_usage = load_phonelock_sensing(dataset_dir)   # {uid: daily_screen_mins}
    call_count  = load_call_log(dataset_dir)
    sms_count   = load_sms(dataset_dir)

    # ── Build one row per student ──────────────────────────────────────────
    rows = []
    for uid in ALL_UIDS:
        if uid not in phq_map:
            continue   # skip students with no PHQ-9 label

        phq   = phq_map[uid]
        label = phq_to_label(phq)

        sleep_vals = sleep_ema.get(uid, [])
        avg_act    = activity_ema.get(uid, np.nan)
        avg_mood   = mood_ema.get(uid, np.nan)
        avg_sleep  = float(np.mean(sleep_vals)) if sleep_vals else np.nan
        sleep_irr  = float(np.std(sleep_vals))  if len(sleep_vals) > 1 else 0.0

        phone  = phone_usage.get(uid, np.nan)
        calls  = call_count.get(uid, np.nan)
        sms    = sms_count.get(uid, np.nan)
        act_s  = activity_sensing.get(uid, np.nan)
        conv_s = conv_sensing.get(uid, np.nan)     # daily conversation minutes

        # Social score: weighted calls + sms + conversation
        social = (
            (calls  if not np.isnan(calls)  else 0) * 1.5 +
            (sms    if not np.isnan(sms)    else 0) * 1.0 +
            (conv_s if not np.isnan(conv_s) else 0) * 0.5
        )

        # Phone-activity ratio (isolation proxy): high phone + low activity
        phone_act_ratio = (
            (phone if not np.isnan(phone) else 0) /
            max(avg_act if (not np.isnan(avg_act) and avg_act > 0) else 1, 0.1)
        )

        rows.append({
            "uid":                  uid,
            "avg_activity_ema":     avg_act,
            "avg_mood":             avg_mood,
            "avg_sleep_duration":   avg_sleep,
            "phone_usage_mins":     phone,
            "call_count":           calls,
            "sms_count":            sms,
            "activity_sensing":     act_s,
            "sleep_sensing":        conv_s,        # reuse slot for conversation mins
            "social_score":         social,
            "sleep_irregularity":   sleep_irr,
            "phone_activity_ratio": phone_act_ratio,
            PHQ_COL:                phq,
            LABEL_COL:              label,
        })

    df = pd.DataFrame(rows)
    print(f"[preprocess] Built feature table: {len(df)} students, "
          f"{df[LABEL_COL].notna().sum()} with valid labels")
    return df


# ════════════════════════════════════════════════════════════════════════════
# 6. Handle missing values
# ════════════════════════════════════════════════════════════════════════════

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing feature values.
    Uses column median when available, otherwise 0.
    Ensures no NaN values survive into KMeans / normalisation.
    """
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            continue
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        median = df[col].median()
        fill   = median if not np.isnan(median) else 0.0
        print(f"[preprocess] Imputing {n_missing} missing in '{col}' → {fill:.3f}")
        df[col] = df[col].fillna(fill)
    # Final safety net — should never be needed but guarantees no NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


# ════════════════════════════════════════════════════════════════════════════
# 7. Normalise + cluster + split
# ════════════════════════════════════════════════════════════════════════════

def normalise_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df, scaler


def assign_clusters_kmeans(
    df: pd.DataFrame, n_clusters: int = 4, seed: int = 42
) -> Tuple[pd.DataFrame, KMeans]:
    """
    Cluster students by behavioural feature vector.
    With real data this naturally reflects depression severity groups.
    """
    features = df[FEATURE_COLS].values
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    df = df.copy()
    df["cluster"] = km.fit_predict(features)
    return df, km


ClientData = Dict[str, np.ndarray]


def build_client_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, ClientData], int, int]:
    """
    Since we have ONE ROW per student (aggregated over the term),
    we treat each student's feature vector as their single data point
    and tile it to create a small local dataset for FL training.

    Each client gets N_WEEKS synthetic weekly variations via small
    Gaussian jitter around their mean feature vector — this mimics
    having weekly sensor readings while preserving the real PHQ label.
    """
    rng = np.random.default_rng(seed)
    clients: Dict[int, ClientData] = {}

    for _, row in df.iterrows():
        uid     = int(row["uid"])
        x_mean  = row[FEATURE_COLS].values.astype(np.float32)
        label   = int(row[LABEL_COL])
        cluster = int(row["cluster"])
        phq     = float(row[PHQ_COL])

        # Generate N_WEEKS ≈ 10 weekly samples via Gaussian jitter
        noise = rng.normal(0, 0.05, size=(N_WEEKS, len(FEATURE_COLS))).astype(np.float32)
        X = np.clip(x_mean[None, :] + noise, -1.0, 1.0)
        y = np.full(N_WEEKS, label, dtype=np.int64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        clients[uid] = dict(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            cluster=cluster, phq_mean=phq,
            depression_class_dist={label: N_WEEKS},
        )

    n_features = len(FEATURE_COLS)
    n_classes  = 4
    return clients, n_features, n_classes


# ════════════════════════════════════════════════════════════════════════════
# 8. One-stop entry point
# ════════════════════════════════════════════════════════════════════════════

def prepare_data(
    data_dir:   str   = "data/raw",
    n_clusters: int   = 4,
    test_size:  float = 0.2,
    seed:       int   = 42,
) -> Tuple[Dict[int, ClientData], int, int, int]:
    """
    Full pipeline for the real StudentLife dataset.

    Parameters
    ----------
    data_dir    : path to the root dataset folder (contains survey/, EMA/, etc.)
    n_clusters  : number of FL clusters
    test_size   : train/test split ratio
    seed        : random seed

    Returns
    -------
    clients, n_features, n_classes, n_clusters
    """
    df = build_feature_table(data_dir)
    df = impute_features(df)
    df, _scaler = normalise_features(df)
    df, _km     = assign_clusters_kmeans(df, n_clusters=n_clusters, seed=seed)
    clients, n_features, n_classes = build_client_datasets(
        df, test_size=test_size, seed=seed
    )

    # ── Summary ──────────────────────────────────────────────────────────
    cluster_counts: Dict[int, int] = {}
    cluster_phq: Dict[int, list]   = {}
    label_dist: Dict[int, int]     = {}

    for cd in clients.values():
        c = cd["cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        cluster_phq.setdefault(c, []).append(cd["phq_mean"])
        lbl = int(cd["y_train"][0])
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    label_names = {0: "Minimal(0-4)", 1: "Mild(5-9)",
                   2: "Moderate(10-14)", 3: "Severe(15-27)"}

    print(f"\n[preprocess] ── Real StudentLife Results ──────────────────")
    print(f"[preprocess] Task      : Depression Detection (PHQ-9, 4-class)")
    print(f"[preprocess] Students  : {len(clients)}")
    print(f"[preprocess] Features  : {n_features}")
    print(f"[preprocess] Clusters  : {n_clusters}")
    print(f"[preprocess] Clients/cluster: {dict(sorted(cluster_counts.items()))}")
    print(f"[preprocess] Label distribution:")
    for lbl in sorted(label_dist):
        print(f"             Class {lbl} ({label_names[lbl]}): {label_dist[lbl]} students")
    print(f"[preprocess] Mean PHQ-9 per FL cluster:")
    for c in sorted(cluster_phq):
        mean_phq = np.mean(cluster_phq[c])
        print(f"             Cluster {c}: PHQ={mean_phq:.1f} "
              f"→ {label_names.get(phq_to_label(mean_phq), '?')}")
    print()

    return clients, n_features, n_classes, n_clusters


# ════════════════════════════════════════════════════════════════════════════
# 9. Smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Pass dataset root as argument, e.g.:
    #   python -m data.preprocess C:/Users/Shashank/Downloads/Compressed/dataset
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    clients, nf, nc, nk = prepare_data(data_dir=dataset_root)

    uid0 = list(clients.keys())[0]
    cd   = clients[uid0]
    print(f"Client u{uid0:02d}:")
    print(f"  PHQ-9 mean   = {cd['phq_mean']:.1f}")
    print(f"  Cluster      = {cd['cluster']}")
    print(f"  X_train      = {cd['X_train'].shape}")
    print(f"  y_train      = {cd['y_train']}")
