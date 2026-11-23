# Customer Credit Risk: Data Preprocessing & Feature Engineering

## Executive summary
This project prepares a synthetic customer credit-risk dataset for a binary loan-default prediction task. The target is `default_flag` (0 = No Default, 1 = Default). The source CSV contains 1,500 customer rows and 15 columns before supplementary joins. The merged acquisition demonstration combines CSV customer records, JSON metadata, SQL repayment history, and an offline dummy API table.

## Dataset fields
customer_id, age, gender, region, education_level, employment_type, annual_income, loan_amount, loan_purpose, credit_score, repayment_history, transaction_count, spending_ratio, join_date, default_flag

## Workflow completed in the notebook
1. Data acquisition from CSV, JSON, SQLite SQL table and dummy API payload.
2. Data understanding with `info()`, `describe()`, missingness, duplicates and target distribution.
3. Missing data comparisons: median/mode SimpleImputer, most-frequent category, missing indicator + random-sample imputation, KNN, MICE/IterativeImputer, complete-case analysis.
4. Outlier detection and handling: Z-score, IQR, percentile capping and Winsorization.
5. Date extraction: year, month, day and weekday.
6. Encoding: ordinal education encoding, gender label-encoding demonstration, one-hot region and loan purpose.
7. Numerical encoding: quantile binning, payment-history bands, credit-score binarization (>700), K-Means transaction bins.
8. Scaling: Standard, Min-Max, MaxAbs, Robust and row-wise Normalizer comparison.
9. Transformations: log, reciprocal, square root, Box-Cox and Yeo-Johnson.
10. Feature construction: debt-to-income ratio, average monthly transactions and spending-to-income ratio.
11. Mixed-type `ColumnTransformer` with train/test split and train-only fitting.
12. Export of cleaned engineered data and model-ready train/test matrices.

## Interpretation guide
- Median imputation is a robust baseline for skewed numerical data; mode is appropriate for categorical data.
- KNN and MICE exploit multivariate relationships but need careful scaling, validation and computational consideration.
- IQR and percentile capping are useful for skewed financial features; outliers should be reviewed, not automatically discarded.
- Ordinal encoding is appropriate only when categories have a meaningful order; one-hot encoding is safer for nominal categories.
- Robust scaling is less sensitive to extreme values; StandardScaler is useful for many distance- or gradient-based methods.
- Log/power transformations can reduce right skew, but compare distributions and preserve interpretability.
- The ColumnTransformer prevents inconsistent treatment of numeric and categorical variables and is fitted on training data only.

## Reproducibility and outputs
Run all notebook cells from top to bottom. It writes:
- `outputs/01_data_quality.png`
- `outputs/02_outlier_boxplots.png`
- `outputs/03_income_transformation.png`
- `outputs/customer_credit_risk_cleaned_engineered.csv`
- `outputs/model_train_processed.csv`
- `outputs/model_test_processed.csv`

## Limitations and responsible use
The dataset is synthetic and cannot establish real-world credit-risk relationships. The exercise is not a validated lending model. Real deployment would require representative data, temporal validation, leakage prevention, fairness and bias assessment, explainability, calibration, monitoring, legal review and governance. Protected or sensitive attributes must not be used without a lawful, justified and audited basis.

## Notebook execution status
Notebook executed successfully; outputs embedded
