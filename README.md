# Customer Segmentation and High-Value Spender Prediction

A YAML-driven pipeline for any transactional customer dataset. It computes per-customer RFM features, segments customers with four clustering algorithms (one of which is a PyTorch autoencoder followed by KMeans), and predicts top-quartile spenders with Random Forest, XGBoost, and an MLP. SMOTE is used inside cross-validation to handle class imbalance. SHAP explains the best tree model.

---

## Running on the UCI Online Retail II dataset

```bash
pip install -r requirements.txt
python main.py --config configs/online_retail_ii.yaml
```

The script downloads `online_retail_II.xlsx` into `data/` on first run (around 45 MB), runs the full pipeline, and writes outputs to `outputs/figures/` and `outputs/tables/`. End-to-end time is roughly 3 to 5 minutes on a laptop.

## Running on a different dataset

Every dataset-specific choice lives in a YAML config. To work on a new dataset, write a new config that maps your source columns to the canonical ones the pipeline uses.

```bash
python examples/generate_synthetic.py
python main.py --config configs/sample_synthetic.yaml
```

A config has six sections. See [configs/online_retail_ii.yaml](configs/online_retail_ii.yaml) for the commented baseline.

```yaml
dataset:    # loader (csv / excel / parquet), path, optional download_url
schema:     # source -> canonical: customer_id, transaction_id,
            # transaction_date, quantity, unit_price, revenue
cleaning:   # cancellation rule, missing/non-positive/duplicate drops,
            # outlier method (iqr / zscore / none) + columns
features:   # log1p toggle, scaler choice, snapshot date for Recency
target:     # top_quantile (default 0.75) / threshold / top_n
modeling:   # test_size, smote, cv_folds, which models to run,
            # k_range, dbscan params, autoencoder hyperparams
output:     # figures_dir, tables_dir
```

To add a new dataset: copy `configs/online_retail_ii.yaml`, edit the schema, path, and cleaning toggles, then run `python main.py --config configs/your_dataset.yaml`.

---

## Pipeline

1. **Preprocessing**. Load source with the configured loader. Rename source columns to canonical names. Apply the configured cleaning and outlier rules. Aggregate to per-customer Recency, Frequency, Monetary. Apply log1p (optional) and the chosen scaler.
2. **Clustering**. K-Means with an elbow plus silhouette sweep to pick `k`. DBSCAN. Gaussian Mixture with a BIC sweep that picks its own `k`. An MLP autoencoder (3-16-8-4-8-16-3) followed by KMeans on the latent codes.
3. **Classification**. Predict the configured high-value label. Random Forest, XGBoost, and an MLP. Each model is wrapped in an `imblearn.pipeline.Pipeline` so the StandardScaler and SMOTE run inside each CV fold rather than once on the whole training set. Running SMOTE before `cross_val_score` causes information from held-out folds to leak into training and inflates CV scores.
4. **Explainability**. SHAP TreeExplainer on the best tree model. The pipeline is unwrapped automatically so SHAP sees the raw tree estimator and the pre-scaled feature matrix.

All randomness is seeded with `SEED = 42`.

---

## Results on UCI Online Retail II

### Clustering, 5,675 customers, two-cluster solution

| Algorithm                         | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|-----------------------------------|:---:|:---:|:---:|
| K-Means                           | 0.437 | 0.873 | 6,018 |
| DBSCAN                            | collapsed at default eps | n/a | n/a |
| Gaussian Mixture (BIC-selected k) | 0.309 | 0.977 | 3,232 |
| **Autoencoder + KMeans**          | **0.574** | **0.622** | **11,748** |

The two segments are Champions (n=2,267, recent and frequent buyers with high spend) and Loyal-but-lapsing (n=3,408, around 300 days since last purchase).

### Classification, top-25 percent Monetary, 75/25 split, test set

| Model           | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-----------------|:---:|:---:|:---:|:---:|:---:|
| Random Forest   | 0.875 | 0.720 | 0.817 | 0.765 | 0.931 |
| **XGBoost**     | **0.885** | 0.715 | 0.899 | **0.797** | 0.952 |
| MLP             | 0.865 | 0.665 | 0.924 | 0.774 | **0.956** |

After moving SMOTE inside CV, the test numbers above are unchanged but the CV F1 estimates drop closer to test F1, which is what an unbiased CV should look like.

### SHAP attribution on the XGBoost model

| Feature   | mean \|SHAP\| |
|-----------|:---:|
| Frequency | 2.73 |
| Recency   | 0.66 |

Frequency dominates Recency by roughly four to one. Monetary is held out of the feature set because it defines the label.

---

## Project layout

```
sml_project/
├── configs/
│   ├── online_retail_ii.yaml      # UCI baseline (commented)
│   └── sample_synthetic.yaml      # demo using non-UCI column names
├── examples/
│   └── generate_synthetic.py      # writes data/synthetic_transactions.csv
├── data/                          # datasets live here
├── src/
│   ├── config.py                  # YAML loader and dataclass validation
│   ├── preprocess.py              # loader, schema mapping, RFM
│   ├── clustering.py              # KMeans, DBSCAN, GMM (BIC), AE+KMeans
│   ├── classification.py          # RF, XGB, MLP, SMOTE inside CV
│   ├── evaluation.py              # metric tables and SHAP
│   ├── visualisation.py           # figures
│   └── utils.py
├── outputs/
│   ├── figures/                   # 9 PNGs at 300 dpi
│   └── tables/                    # 9 CSVs
├── main.py                        # python main.py --config <path>
├── requirements.txt
└── README.md
```

---

## Tech stack

`Python 3.11`, `pandas`, `numpy`, `scikit-learn`, `XGBoost`, `PyTorch`, `imbalanced-learn`, `SHAP`, `matplotlib`, `PyYAML`.

---

## Author

**Shreya Mishra** &nbsp;·&nbsp; 
Centre of Quantitative Economics and Data Science, BIT Mesra



