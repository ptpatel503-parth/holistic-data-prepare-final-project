🏦 Customer Credit Risk — End-to-End Feature Engineering & Data Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%26%20Transformers-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-Tensor%20%26%20Array%20Ops-013243.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, leakage-free data acquisition, exploratory profiling, preprocessing, and feature engineering framework built for **consumer credit risk scoring** and **loan default prediction** ($y \in \{0, 1\}$).

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Repository Structure](#-repository-structure)
- [Dataset Schema](#-dataset-schema)
- [Methodology & Key Modules](#-methodology--key-modules)
- [Production Scikit-Learn Pipeline](#-production-scikit-learn-pipeline)
- [Getting Started](#-getting-started)
- [Generated Outputs](#-generated-outputs)
- [License](#-license)

---

## 🎯 Overview

In financial risk modeling, predicting loan default probability requires combining heterogeneous customer data sources while strictly preventing train-test data leakage. This repository implements a modular pipeline that transforms raw tabular, JSON, SQL, and macroeconomic feeds into a clean, scaled, and model-ready feature matrix.

### Core Capabilities
- **Multi-Source Ingestion:** Merges flat files (`.csv`), semi-structured feeds (`.json`), relational ledgers (`SQLite`), and external macroeconomic API indicators.
- **Comparative Missingness Resolution:** Evaluates Univariate Median/Mode, Missing Indicator + Random Sampling (preserving MNAR signals), Distance-Weighted KNN, and Multivariate Imputation by Chained Equations (MICE).
- **Robust Outlier Management:** Implements Z-Score filtering ($|z| > 3.0$), Tukey's IQR fences, Percentile Clipping, and 5th/95th Percentile Winsorization.
- **Domain-Driven Feature Construction:** Engineers financial solvency ratios ($DTI$), transaction velocity, customer tenure, and composite financial stress indices.
- **Production `ColumnTransformer`:** Encapsulates Yeo-Johnson power transformations, robust/standard scaling, ordinal ranking, and multicollinearity-safe one-hot encoding (`drop='first'`).

---

## 🏗 Pipeline Architecture

```text
Heterogeneous Data Sources
 ├── customer_credit_risk_dataset.csv  (Core Demographics & Financials)
 ├── customer_metadata.json            (KYC & Digital Onboarding Channel)
 ├── SQLite Loan Ledger                (Prior Settlements & Bureau Inquiries)
 └── Macroeconomic API                 (Central Bank Repo Rate & CPI)
                  │
                  ▼
   [1. Relational Join & Schema Alignment]
                  │
                  ▼
   [2. Missing Value Imputation (Median / Mode / KNN / MICE)]
                  │
                  ▼
   [3. Outlier Capping & Winsorization (IQR & 5%-95% Limits)]
                  │
                  ▼
   [4. Temporal Extraction, Binning & Domain Ratios (DTI, Velocity)]
                  │
                  ▼
   [5. Production Scikit-Learn ColumnTransformer]
    ├── Skewed Numerical  ──▶ Median Impute ──▶ Yeo-Johnson ──▶ RobustScaler
    ├── Standard Numerical ──▶ Median Impute ──▶ StandardScaler
    ├── Ordinal Categorical ──▶ Mode Impute   ──▶ OrdinalEncoder
    └── Nominal Categorical ──▶ Mode Impute   ──▶ OneHotEncoder (drop='first')
                  │
                  ▼
    Model-Ready Feature Matrix (Zero Missing Values)
```

---

## 📂 Repository Structure

```text
├── Holistic Data Preparer (Final Project)dpfe.ipynb  # Main Jupyter Notebook (Parts A–H)
├── customer_credit_risk_dataset.csv                   # Raw credit risk dataset (1,000+ records)
├── customer_metadata.json                             # Customer KYC and channel metadata
├── customer_credit_risk_final_transformed(data set).xlsx # Final processed reference dataset
├── customer_credit_risk_pipeline_documentation.md     # Detailed technical documentation
├── standard_scaled.csv                                # Z-score standardized features
├── minmax_scaled.csv                                  # Min-Max normalized features [0, 1]
├── robust_scaled.csv                                  # Median/IQR scaled features
├── maxabs_scaled.csv                                  # Max-Absolute scaled features [-1, 1]
└── README.md                                          # Repository overview & setup guide
```

---

## 📋 Dataset Schema

| Feature | Data Type | Source | Description |
| :--- | :--- | :--- | :--- |
| `customer_id` | Alphanumeric | CSV / JSON / SQL | Primary customer key (e.g., `CUST_10000`) |
| `age` | Numerical | Base CSV | Applicant age in years |
| `annual_income` | Continuous | Base CSV | Gross annual income |
| `loan_amount` | Continuous | Base CSV | Requested loan principal |
| `credit_score` | Numerical | Base CSV | Credit bureau score ($300 - 850$) |
| `spending_ratio` | Continuous | Base CSV | Monthly expenditure relative to income |
| `transaction_count`| Discrete | Base CSV | Total transaction volume over 6 months |
| `repayment_history`| Discrete | Base CSV | Count of historical delinquency / missed payments |
| `education_level` | Ordinal | Base CSV | `Primary` < `Secondary` < `Graduate` < `Post-Graduate` |
| `employment_type` | Nominal | Base CSV | `Salaried`, `Self-Employed`, `Unemployed` |
| `region` | Nominal | Base CSV | Operational geographic region |
| `loan_purpose` | Nominal | Base CSV | Stated purpose of loan (`Personal`, `Auto`, `Home`, `Education`) |
| `kyc_status` | Nominal | Metadata JSON | Identity verification state (`Verified`, `Pending`) |
| `channel` | Nominal | Metadata JSON | Acquisition channel (`Mobile_App`, `Web`, `Branch`) |
| `prior_settlements`| Integer | Bureau SQL | Count of past settled credit facilities |
| `bureau_inquiries` | Integer | Bureau SQL | Credit checks in the trailing 6 months |
| `macro_repo_rate` | Float | External API | Benchmark policy interest rate |
| `default_flag` | Binary Target | Target | Loan default outcome ($0 = \text{No Default}, 1 = \text{Default}$) |

---

## ⚙ Methodology & Key Modules

The primary notebook (`Holistic Data Preparer (Final Project)dpfe.ipynb`) is organized into eight structured stages:

1. **Part A — Conceptual & Tensor Foundations:** Frames binary credit risk classification under asymmetric cost constraints ($FN > FP$) and demonstrates NumPy tensor ranks ($0\text{D}$ scalars through $4\text{D}$ batches).
2. **Part B — Multi-Source Data Acquisition:** Integrates CSV, JSON, in-memory SQLite queries, and external API contextual indicators on `customer_id`.
3. **Part C — Data Understanding & Imputation:** Benchmarks Complete Case Analysis (CCA), `SimpleImputer`, Random Sample Imputation with missingness flags, `KNNImputer`, and `IterativeImputer` (MICE).
4. **Part D — Outlier Detection & Treatment:** Applies parametric Z-score thresholds ($|z| > 3.0$), non-parametric Tukey IQR bounds ($[Q_1 - 1.5 \times \text{IQR},\ Q_3 + 1.5 \times \text{IQR}]$), percentile clipping, and 5%–95% Winsorization.
5. **Part E — Feature Engineering & Discretization:** Extracts regex numeric IDs, decomposes dates into tenure days/weekdays, binarizes prime credit tiers ($\text{credit\_score} \ge 700$), and applies Equal-Width, Quantile, and 1D K-Means discretization (`KBinsDiscretizer`).
6. **Part F — Feature Scaling Comparison:** Generates and exports comparative feature matrices across `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, and `RobustScaler`.
7. **Part G — Power Transforms & Domain Features:** Applies `log1p`, `sqrt`, `Box-Cox`, and `Yeo-Johnson` transformations alongside engineered risk ratios:
   - **Debt-to-Income Ratio:** $\text{DTI} = \frac{\text{loan\_amount}}{\text{annual\_income} + 1.0}$
   - **Monthly Velocity:** $\text{Avg Monthly Transactions} = \frac{\text{transaction\_count}}{6.0}$
   - **Financial Stress Index:** $\text{Stress Index} = \text{spending\_ratio} \times (\text{repayment\_history} + 1.0)$
8. **Part H — Summary & ML Readiness Report:** Consolidates analytical findings and outputs the final 26-column encoded matrix.

---

## 🛠 Production Scikit-Learn Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

num_skewed_features = ['annual_income', 'loan_amount', 'spending_ratio']
num_standard_features = ['age', 'transaction_count', 'repayment_history']
cat_ordinal_features = ['education_level']
cat_nominal_features = ['gender', 'region', 'employment_type', 'loan_purpose']

preprocessor = ColumnTransformer(
    transformers=[
        ('skewed_num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('yeo_johnson', PowerTransformer(method='yeo-johnson', standardize=False)),
            ('robust_scaler', RobustScaler())
        ]), num_skewed_features),
        ('std_num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ]), num_standard_features),
        ('ord_cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal_enc', OrdinalEncoder(
                categories=[['Primary', 'Secondary', 'Graduate', 'Post-Graduate']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ]), cat_ordinal_features),
        ('nom_cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), cat_nominal_features)
    ],
    remainder='drop'
)

preprocessor.set_output(transform="pandas")
```

---

## 🚀 Getting Started

### Prerequisites
- Python `3.10+`
- Jupyter Notebook / JupyterLab

### Installation & Execution
```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-credit-risk-pipeline.git
cd customer-credit-risk-pipeline

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install numpy pandas scikit-learn scipy openpyxl requests jupyter

# 4. Launch the pipeline notebook
jupyter notebook "Holistic Data Preparer (Final Project)dpfe.ipynb"
```

---

## 📦 Generated Outputs

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `customer_credit_risk_final_transformed(data set).xlsx` | Excel | Complete cleaned, imputed, and feature-engineered dataset |
| `standard_scaled.csv` | CSV | Numerical features scaled to $\mu = 0, \sigma = 1$ |
| `minmax_scaled.csv` | CSV | Numerical features scaled to range $[0, 1]$ |
| `robust_scaled.csv` | CSV | Numerical features centered on median and scaled by IQR |
| `maxabs_scaled.csv` | CSV | Numerical features scaled by maximum absolute value $[-1, 1]$ |
