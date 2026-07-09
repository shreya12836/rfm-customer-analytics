# Customer Value Prediction: Deep Clustering + Leakage-Safe Classification

A config-driven pipeline that segments retail customers by purchase behavior and predicts which ones are high-value — built to be correct under evaluation, not just to produce a plot.

## Problem

Retailers need to know which customers are worth retaining before they churn, but naive segmentation (e.g., manual RFM thresholds) misses non-linear structure in purchase behavior, and naive classification pipelines routinely leak information during cross-validation, producing scores that don't hold up in production.

## Approach

The pipeline runs in five stages: preprocess → cluster → classify → explain → report.

1. **RFM feature engineering** — Recency, Frequency, Monetary computed from raw transactions, log-transformed and scaled per a YAML-configurable strategy.
2. **Unsupervised segmentation** — four clustering approaches are run and compared on the same scaled features, so the best method is chosen by evidence, not assumption.
3. **Supervised classification** — three models predict top-quartile spenders from Recency and Frequency alone (Monetary is excluded because it defines the label).
4. **Explainability** — SHAP quantifies which behavioral signal actually drives the prediction.

## Results

**Clustering** (5,675 customers, evaluated on Silhouette, Davies-Bouldin, Calinski-Harabasz):

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|:---:|:---:|:---:|
| K-Means | 0.437 | 0.873 | 6,018 |
| Gaussian Mixture (BIC-selected k) | 0.309 | 0.977 | 3,232 |
| DBSCAN | collapsed at default eps | n/a | n/a |
| **Autoencoder + K-Means** | **0.574** | **0.622** | **11,748** |

The autoencoder (3-16-8-4-8-16-3) learns a non-linear latent representation before clustering, and wins on all three metrics. It resolves two actionable segments: **Champions** (n=2,267, recent and frequent buyers with high spend) and **Loyal-but-lapsing** (n=3,408, ~300 days since last purchase).

**Classification** (top-25% Monetary, 75/25 split, test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.875 | 0.720 | 0.817 | 0.765 | 0.931 |
| **XGBoost** | **0.885** | 0.715 | 0.899 | **0.797** | 0.952 |
| MLP | 0.865 | 0.665 | 0.924 | 0.774 | **0.956** |

**SHAP** (XGBoost): Frequency drives the prediction roughly 4x more than Recency (mean \|SHAP\| 2.73 vs 0.66) — customers are flagged as high-value primarily because they buy often, not because they bought recently. Monetary is excluded from the feature set since it defines the label.

## Engineering Highlights

- **Leakage-safe evaluation by construction.** SMOTE sits inside an `imblearn.pipeline.Pipeline` alongside the scaler, so it is refit on each cross-validation fold rather than applied once to the full training set — a common source of inflated CV scores in imbalanced classification that this design rules out structurally. After moving SMOTE inside CV, test-set numbers were unchanged but CV F1 estimates dropped closer to test F1, consistent with an unbiased CV.
- **Config-driven, not hardcoded.** Every pipeline decision — data schema, cleaning rules, feature scaling, label definition, model selection, hyperparameters — is declared in YAML and validated at load time (`src/config.py`). Pointing the pipeline at a different retail dataset is a config change, not a code change; see [configs/online_retail_ii.yaml](configs/online_retail_ii.yaml) for the commented baseline.
- **Reproducible end-to-end.** All stochastic steps (clustering, splitting, model training) are seeded (`SEED = 42`); a `python main.py --config <path>` run regenerates the exact tables and figures in `outputs/`.
- **Explainability wired into the pipeline, not bolted on.** SHAP's TreeExplainer runs on the best tree model with the pipeline automatically unwrapped so it sees the raw estimator and pre-scaled features.

## Usage

```bash
pip install -r requirements.txt
python main.py --config configs/online_retail_ii.yaml
```

Downloads `online_retail_II.xlsx` into `data/` on first run (~45 MB), runs the full pipeline, and writes results to `outputs/figures/` and `outputs/tables/`. End-to-end runtime is roughly 3-5 minutes on a laptop.

To run on a different dataset, write a new YAML config mapping your source columns to the canonical schema (`customer_id`, `transaction_id`, `transaction_date`, `quantity`, `unit_price`, `revenue`) — no code changes needed:

```bash
python examples/generate_synthetic.py
python main.py --config configs/sample_synthetic.yaml
```

## Data

[UCI Online Retail II](https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx) — real transaction-level e-commerce data, 5,675 customers after cleaning (cancellations, missing customer IDs, and non-positive quantities/prices removed).

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `PyTorch` · `imbalanced-learn` · `SHAP` · `matplotlib` · `PyYAML`

## Author

**Shreya Mishra** · Centre of Quantitative Economics and Data Science, BIT Mesra
