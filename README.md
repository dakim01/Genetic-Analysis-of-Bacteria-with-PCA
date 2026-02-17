# 🧬 Multi-Class Bacterial Species Identification from High-Dimensional Gene Expression Data

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-interactive-blueviolet?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning pipeline for identifying bacterial species from high-dimensional gene expression data (286 genetic features, 10 species). This project applies **Principal Component Analysis (PCA)** for dimensionality reduction and uses an **Extra Trees Classifier** for multi-class identification, comparing model performance with and without PCA compression.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline Summary](#-pipeline-summary)
- [Key Results](#-key-results)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Methods in Detail](#-methods-in-detail)
- [Contributing](#-contributing)

---

## 🔬 Project Overview

Bacterial identification from genetic data is a critical task in microbiology, diagnostics, and bioinformatics. This notebook tackles a multi-class classification problem where each sample contains **286 gene expression features** measured across multiple bacterial species.

The core questions explored:
1. Which genes are most informative for distinguishing bacterial species?
2. How much of the genetic variance can be explained by a reduced set of principal components?
3. Does PCA-based dimensionality reduction maintain predictive performance compared to using all 286 features?

---

## 📊 Dataset

| Split | Samples (raw) | Features | After dedup |
|-------|--------------|----------|-------------|
| Train | 200,000 | 287 (286 + target) | 123,993 |
| Test  | 100,000 | 286 | — |

- **Features:** 286 numerical gene expression variables
- **Target:** 10 bacterial species (multi-class classification)
- **No missing values** in either split
- **Duplicates removed** from training data (~38% were duplicates)

> **Note:** The data files (`train.csv`, `test.csv`, `sample_submission.csv`) are not included in this repository. Place them in a local `data/` directory and update the file paths in the notebook accordingly.

---

## 🔄 Pipeline Summary

```
Raw Data
   │
   ├── EDA & Quality Checks
   │     ├── Duplicate removal
   │     ├── Class distribution analysis
   │     └── Inter-gene correlation analysis
   │
   ├── Feature Engineering
   │     ├── Box-Cox transformation (skewed features, skew > 0.75)
   │     └── StandardScaler normalization
   │
   ├── Dimensionality Reduction
   │     ├── Full PCA (286 components) — variance analysis
   │     └── Reduced PCA (100 components) — for modeling
   │
   ├── Modeling
   │     ├── ExtraTreesClassifier (PCA features, 100 components)
   │     └── ExtraTreesClassifier (all 286 features)
   │
   └── Evaluation
         ├── Accuracy, F1-Score, AUC
         ├── Per-class classification report
         └── Multiclass ROC curves
```

---

## 📈 Key Results

### Model Comparison

| Model | Features Used | Accuracy | F1-Score | AUC |
|-------|--------------|----------|----------|-----|
| ExtraTrees + PCA | 100 principal components | — | — | — |
| ExtraTrees (Full) | 286 original features | — | — | — |

> *Fill in your results after running the notebook.*

### PCA Variance Analysis

- **PC1** explains ~31.9% of variance
- **PC2** explains ~20.4% of variance
- Top 100 components capture the vast majority of cumulative variance

---

## 📊 Visualizations

The notebook produces the following interactive (Plotly) and static charts:

| Visualization | Description |
|--------------|-------------|
| Species distribution bar chart | Class balance across 10 bacterial species |
| Gene correlation heatmap | Pairwise correlations across all 286 genes |
| Most correlated gene pairs | Gene pairs with absolute correlation > 0.75 |
| Per-species top correlated genes | Highest intra-species gene correlation for each species |
| PCA explained variance (animated) | Individual and cumulative variance per principal component |
| Gene importance per PC | Top gene loadings in the first 10 principal components |
| PC loadings subplot (PC1–PC4) | Top 5 gene contributors to each of the first 4 PCs |
| 2D PCA scatter (PC1 vs PC2) | Species clusters projected onto the first two components |
| Multiclass ROC curves | Per-species AUC curves for the PCA-based model |
| Prediction distribution | Predicted species breakdown on the test set |

---

## 📁 Project Structure

```
bacterial-species-identification-gene-expression/
│
├── Genetic_Analysis_of_Bacteria_with_PCA.ipynb   # Main notebook
├── README.md                                      # This file
├── requirements.txt                               # Python dependencies
│
└── data/                                          # ⚠️ Not included — add your own
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## ⚙️ Requirements

```
numpy
pandas
matplotlib
seaborn
plotly
scipy
scikit-learn
```

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/bacterial-species-identification-gene-expression.git
cd bacterial-species-identification-gene-expression
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data

Place `train.csv`, `test.csv`, and `sample_submission.csv` in a `data/` folder, then update the file paths in the first notebook cell:

```python
# Update these paths to match your local setup
train = pd.read_csv('data/train.csv', index_col=0)
test  = pd.read_csv('data/test.csv',  index_col=0)
sub   = pd.read_csv('data/sample_submission.csv')
```

### 4. Launch the notebook

```bash
jupyter notebook Genetic_Analysis_of_Bacteria_with_PCA.ipynb
```

---

## 📖 Usage

Run the cells in order from top to bottom. The notebook is organized into the following sections:

1. **Data Loading & Quality Checks** — Load CSVs, check shapes, missing values, and duplicates
2. **Summary Statistics** — Grouped stats by bacterial species
3. **Species Distribution** — Interactive bar chart of class balance
4. **Gene Correlations** — Correlation matrix and most correlated gene pairs
5. **Per-Species Correlations** — Top correlated gene pair for each species
6. **Skewness Correction** — Box-Cox transformation for skewed gene features
7. **Standardization** — StandardScaler applied to training features
8. **PCA Analysis** — Full 286-component PCA with animated variance chart
9. **Gene Importance in PCA** — Weighted gene importance per principal component
10. **2D PCA Projection** — Scatter plot of species clusters in PC1 vs PC2 space
11. **Modeling with PCA (100 components)** — Train, validate, and evaluate ExtraTrees on reduced features
12. **ROC Curves** — Per-species multiclass ROC and AUC
13. **Test Predictions (PCA)** — Generate and save PCA-based submission
14. **Modeling with All Features** — Train and evaluate ExtraTrees on full 286-feature space
15. **Comparison & Final Predictions** — Full-feature submission and species distribution plot

---

## 🔬 Methods in Detail

### Box-Cox Transformation
Features with skewness > 0.75 are transformed using the Box-Cox power transformation (`scipy.special.boxcox1p`) with the optimal lambda estimated per feature (`scipy.stats.boxcox_normmax`). This normalizes the feature distributions before scaling.

### Principal Component Analysis (PCA)
PCA is applied to the standardized feature matrix. Two variants are used:
- **Full PCA (n=286):** Used purely for variance analysis and visualization — to understand how many components are needed to explain the data.
- **Reduced PCA (n=100):** Used as input features for the classifier, balancing information retention with dimensionality reduction.

### Extra Trees Classifier
`ExtraTreesClassifier` with 500 estimators and `class_weight='balanced'` is used to handle potential class imbalance. The model is evaluated on a stratified 80/20 train-validation split using accuracy, macro-averaged F1-score, and weighted OvR AUC.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
