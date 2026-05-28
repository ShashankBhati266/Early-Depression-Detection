# ACS-FL on StudentLife Dataset

Replication of:
> **"Clustered Federated Learning With Adaptive Local Differential Privacy on Heterogeneous IoT Data"**
> He, Wang & Cai — IEEE Internet of Things Journal, Vol. 11, No. 1, January 2024

---

## Project Structure

```
acsfl_studentlife/
├── data/
│   └── preprocess.py        # Data loading, feature engineering, cluster assignment
├── models/
│   └── mlp.py               # 2-layer MLP (NumPy, no framework dependency)
├── ldp/
│   ├── generalized_pm.py    # Algorithm 1 – Generalised Piecewise Mechanism
│   └── perturbation.py      # Algorithm 2 – Adaptive Layer-wise Perturbation
├── fl/
│   ├── clustered_fl.py      # IFCA-style cluster assignment + FedAvg
│   ├── shuffling.py         # Algorithm 3 – Parameter Shuffling
│   └── dct_compression.py   # Algorithm 4 – DCT Compression
├── experiments/
│   └── run_acsfl.py         # Main training loop + 4 experiment functions
└── results/
    └── plot_results.py      # Matplotlib figures mirroring paper's Fig 1-5
```

---

## Setup

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

---

## Data

### Option 1 – Use synthetic data (immediate)
No download needed. The pipeline auto-generates a StudentLife-like dataset
with 48 clients and 4 behavioural clusters.

### Option 2 – Real StudentLife data
1. Download from [Kaggle](https://www.kaggle.com/datasets/dartweichen/student-life)
2. Place the CSV as `data/raw/studentlife.csv`
3. Ensure columns include:
   `uid, week, sleep_hours, activity_level, phone_usage,`
   `social_duration, stress_level, mood, deadline_count, phq_score`

---

## Quick Start

```bash
# Sanity check (10 rounds, all components enabled)
python experiments/run_acsfl.py --exp quick

# Run all 4 paper experiments
python experiments/run_acsfl.py --exp all

# Run a single experiment
python experiments/run_acsfl.py --exp A   # cluster size
python experiments/run_acsfl.py --exp B   # adaptive vs fixed range
python experiments/run_acsfl.py --exp C   # generalised PM vs Duchi
python experiments/run_acsfl.py --exp D   # DCT compression ratios

# Generate all plots from saved results
python results/plot_results.py
```

---

## ACS-FL Components

| Component | File | Paper Section |
|---|---|---|
| Generalised Piecewise Mechanism | `ldp/generalized_pm.py` | §III-B-1 |
| Adaptive layer-wise perturbation | `ldp/perturbation.py` | §III-B-2, Alg. 2 |
| Clustered FL (IFCA) | `fl/clustered_fl.py` | §III-A, eq. (3) |
| Parameter shuffling | `fl/shuffling.py` | §III-C, Alg. 3 |
| DCT compression | `fl/dct_compression.py` | §III-D, Alg. 4 |

---

## Experiments Replicated

| # | Paper Section | Variable | ε | η |
|---|---|---|---|---|
| A | §IV-A | Number of clusters (2 / 3 / 4) | 9 | 1.0 |
| B | §IV-B | Adaptive vs fixed clipping | 6, 10 | 1.0 |
| C | §IV-C | Generalised PM vs Duchi | 7, 8 | 1.0 |
| D | §IV-D | DCT ratio (25%–100%) | 1, 2 | varied |

---

## Key Implementation Notes

- **No deep-learning framework** — pure NumPy throughout; easy to inspect every
  computation and verify it against paper equations.
- **Synthetic data fallback** — the entire pipeline runs without downloading
  anything; synthetic data mirrors the 4-cluster heterogeneous structure of the
  paper's MNIST rotation experiments.
- **Privacy budget tracking** — cumulative budget ε·T·d is logged every round;
  the shuffling mechanism prevents explosion by not allocating additional budget.
- **DCT sliding window** — each round extracts a different slice of the DCT
  spectrum so all weights participate in training over time.

---

## Tuning Suggestions

| Goal | Parameter | Suggested range |
|---|---|---|
| Stronger privacy | `epsilon` | 1 – 5 |
| Better utility (weaker privacy) | `epsilon` | 6 – 10 |
| Lower communication overhead | `compression_ratio` | 0.25 – 0.70 |
| Fewer clusters | `n_clusters` | 2 – 4 (48 students) |
| Faster convergence | `local_epochs` | 5 – 10 |
