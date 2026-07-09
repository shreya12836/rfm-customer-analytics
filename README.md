# RFM Customer Analytics & High-Value Spender Prediction

A customer analytics pipeline for customer segmentation and high-value spender prediction, built with deep clustering, explainable machine learning, and reproducible, leakage-safe evaluation.

---

## Overview

Customer segmentation is fundamental to targeted marketing, personalized engagement, and customer retention. While traditional RFM thresholding provides a simple segmentation strategy, it often overlooks complex purchasing behaviors. Likewise, many customer value prediction pipelines unintentionally introduce data leakage during evaluation, resulting in overly optimistic performance.

This project implements a configurable machine learning pipeline that combines **RFM feature engineering**, **deep clustering**, **supervised learning**, and **SHAP explainability** to identify high-value customers while following robust evaluation practices.

---

## Key Features

- RFM feature engineering from transaction-level retail data
- Deep customer segmentation using **Autoencoder + K-Means**
- High-value spender prediction using **Random Forest**, **XGBoost**, and **MLP**
- SHAP-based model explainability
- Leakage-safe evaluation with **SMOTE** inside cross-validation
- Configuration-driven pipeline using YAML
- Reproducible experiments with deterministic training

---

## Pipeline Overview

```text
Raw Transactions
        │
        ▼
RFM Feature Engineering
        │
        ▼
Scaling & Preprocessing
        │
        ▼
Customer Segmentation
(K-Means | DBSCAN | Gaussian Mixture | Autoencoder + K-Means)
        │
        ▼
High-Value Spender Prediction
(Random Forest | XGBoost | MLP)
        │
        ▼
SHAP Explainability
        │
        ▼
Reports & Visualizations
```

---

## Results

### Customer Segmentation

The clustering pipeline was evaluated on **5,675 customers** using **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Score**.

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|-------------:|-----------------:|--------------------:|
| K-Means | 0.437 | 0.873 | 6,018 |
| Gaussian Mixture | 0.309 | 0.977 | 3,232 |
| DBSCAN | Collapsed at default `eps` | — | — |
| **Autoencoder + K-Means** | **0.574** | **0.622** | **11,748** |

The Autoencoder learned a non-linear latent representation before clustering and consistently outperformed traditional clustering methods across all evaluation metrics. The resulting customer segments captured two actionable behavioral groups:

- **Champions** — Recent, frequent, and high-spending customers.
- **Loyal-but-Lapsing** — Historically valuable customers with declining purchasing activity.

---

### High-Value Spender Prediction

Customers in the **top 25% of Monetary value** were classified using three supervised learning models.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------:|----------:|-------:|---:|--------:|
| Random Forest | 0.875 | 0.720 | 0.817 | 0.765 | 0.931 |
| **XGBoost** | **0.885** | 0.715 | 0.899 | **0.797** | 0.952 |
| MLP | 0.865 | 0.665 | 0.924 | 0.774 | **0.956** |

**XGBoost** achieved the best balance between precision and recall, while the **MLP** obtained the highest ROC-AUC.

---

### Model Explainability

SHAP analysis revealed that **Frequency** contributes approximately **4× more** than **Recency** when predicting high-value customers, indicating that purchase frequency is the strongest behavioral signal for customer value.

---

## Engineering Highlights

- **Leakage-safe evaluation** — SMOTE is applied inside every cross-validation fold using an `imblearn.pipeline.Pipeline`, preventing synthetic samples from leaking into validation data.
- **Configuration-driven pipeline** — Preprocessing, feature engineering, model selection, and hyperparameters are managed through YAML configuration files.
- **Reproducible experiments** — All stochastic processes are seeded to ensure deterministic training and evaluation.
- **Integrated explainability** — SHAP explanations are automatically generated for the best-performing tree-based model.

---

## Repository Structure

```text
.
├── configs/
├── data/
├── outputs/
├── src/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/shreya12836/rfm-customer-analytics.git
cd rfm-customer-analytics
pip install -r requirements.txt
```

---

## Usage

Run the complete pipeline:

```bash
python main.py --config configs/online_retail_ii.yaml
```

To evaluate a different retail dataset, create a new YAML configuration that maps your dataset to the required schema and run:

```bash
python main.py --config path/to/config.yaml
```

---

## Dataset

This project uses the **UCI Online Retail II** dataset.

- Transaction-level e-commerce data
- **5,675 customers** after preprocessing
- Cancellations, missing customer IDs, and invalid transactions removed

Dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python |
| **Machine Learning** | Scikit-learn, XGBoost, PyTorch, SHAP, imbalanced-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Configuration** | PyYAML |

---


