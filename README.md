# Credit Card Fraud Detection

End-to-end ML/DL pipeline for binary fraud detection on a highly imbalanced tabular dataset. The project walks through EDA, class-imbalance mitigation, classical ML model selection and hyperparameter search, and a comparative deep learning section using MLPs and a Feature-Tokenizer Transformer (FT-Transformer).

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group, Kaggle.

284,807 transactions (492 frauds, ~0.17% positive rate). Features V1–V28 are PCA-derived anonymous components; `Time` and `Amount` are the only interpretable raw features.

> **Note:** the CSV is not included in this repository. Download `creditcard.csv` from the link above and place it under `Data/creditcard.csv`.

---

## Project structure

```
├── Credit_Card_Fraud_Detection.ipynb   # Main notebook
├── Data/
│   └── creditcard.csv                  # (not tracked — download from Kaggle)
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of `Time` and `Amount` by class, with outlier handling (z-score filtering and IQR-based clipping).
- Feature correlation analysis on both the raw imbalanced dataset and the undersampled subset to surface the divergence introduced by undersampling.
- KDE plots for the top-8 most class-discriminative features.

### 2. Preprocessing
- `RobustScaler` applied to `Amount` and `Time` (appropriate given the heavy-tailed distributions).
- Manual random undersampling to create a 50/50 balanced subset for correlation and initial model exploration.
- Outlier removal on fraud-class feature distributions using a 2×IQR multiplier.

### 3. Dimensionality reduction (visualisation)
- t-SNE, PCA, and TruncatedSVD applied to the balanced subset to assess cluster separability.

### 4. Classical ML — model selection
Six classifiers benchmarked on the balanced subset:
- Logistic Regression
- K-Nearest Neighbours
- Support Vector Classifier
- Decision Tree
- Random Forest
- XGBoost

Cross-validation scoring uses **AUPRC (Average Precision)** as the primary metric — appropriate for severe class imbalance where ROC-AUC can be misleadingly optimistic.

### 5. Hyperparameter tuning
`GridSearchCV` / `RandomizedSearchCV` with `StratifiedKFold` across all six classifiers. Best estimators retained for downstream evaluation.

### 6. Imbalance handling — NearMiss undersampling
`NearMiss` applied as an alternative to random undersampling; evaluated via `StratifiedShuffleSplit` to measure the recall/precision trade-off.

### 7. Evaluation on the full imbalanced dataset
Best estimators re-evaluated on a stratified 80/20 split of the **full** dataset (no undersampling) to get honest generalisation estimates. Metrics reported: Precision, Recall, F1, ROC-AUC, AUPRC. Confusion matrices and Precision-Recall curves plotted for all models.

### 8. Deep Learning comparison
Two MLP architectures (swish activations, BatchNorm, Dropout) trained with `BinaryFocalCrossentropy` (γ=2, α=0.25 / α=0.75), `AdamW`, and cosine annealing with warm restarts. An FT-Transformer variant is also benchmarked. Class-weight balancing used in place of resampling.

---

## Key results

| Model | AUPRC |
|---|---|
| XGBoost (scale_pos_weight) | 0.7133 |
| Random Forest | 0.8618 |
| MLP (BFC α=0.75) | 0.8634 |
| FT-Transformer | 0.6381 (REQUIRES MORE TRAINING TIME) |

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
imbalanced-learn
xgboost
tensorflow >= 2.x
matplotlib
seaborn
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Clone the repo
git clone https://github.com/bouhlet0/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection

# 2. Drop the dataset in place
mkdir -p Data
# → copy creditcard.csv into Data/

# 3. Launch the notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## License

MIT — see [LICENSE](LICENSE).
