# Fraud Detection – E-Commerce and Bank Transactions

**Project Overview:**
Adey Innovations Inc. aims to improve fraud detection in both **e-commerce transactions** and **bank credit transactions**. This project builds accurate machine learning models, integrates geolocation data, leverages transaction patterns, and applies explainability techniques to identify fraudulent activities effectively.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [Task 1: Data Analysis and Preprocessing](#task-1-data-analysis-and-preprocessing)
4. [Task 2: Model Building and Training](#task-2-model-building-and-training)
5. [Task 3: Model Explainability](#task-3-model-explainability)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Key Insights & Business Recommendations](#key-insights--business-recommendations)

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/            # Original datasets
│   └── processed/      # Cleaned & feature-engineered data
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── __init__.py
├── src/                # Optional utility scripts
│   └── __init__.py
├── scripts/            # Optional automation scripts
│   └── __init__.py
├── models/             # Saved model artifacts
├── tests/              # Optional test scripts
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## Datasets

1. **Fraud_Data.csv** – E-commerce transactions

   * Key columns: `user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`
   * Challenge: highly imbalanced classes

2. **IpAddress_to_Country.csv** – IP-to-country mapping

3. **creditcard.csv** – Bank credit transactions

   * Key columns: `Time`, `V1`–`V28` (PCA features), `Amount`, `Class`
   * Challenge: extremely imbalanced classes

---

## Task 1: Data Analysis and Preprocessing

**Objective:** Prepare clean, feature-rich datasets ready for modeling.

**Key Steps:**

1. **Data Cleaning:**

   * Removed duplicates, imputed missing `age` with median, converted datetime fields, verified critical columns.

2. **Exploratory Data Analysis (EDA):**

   * Class distributions reveal severe imbalance.
   * Country-level fraud patterns identified.
   * Numerical and categorical features analyzed.

3. **Geolocation Integration:**

   * IP addresses converted to integers.
   * Merged with `IpAddress_to_Country.csv` via range-based lookup.
   * Fraud patterns visualized by country.

4. **Feature Engineering:**

   * Time-based: `hour_of_day`, `day_of_week`, `is_weekend`
   * Duration: `time_since_signup`
   * Transaction frequency: `tx_count_24h`
   * Device/IP risk: `users_per_device`, `users_per_ip`
   * Categorical One-Hot Encoding (`browser`, `source`, `sex`)
   * Standard scaling for numerical features (`purchase_value`, `age`, `Amount`)

5. **Handling Class Imbalance:**

   * **SMOTE** applied to training data only
   * Ensures minority class is well-represented without leaking synthetic data

**Outcome:** Feature-engineered datasets saved in `data/processed/fraud_processed.csv`.

---

## Task 2: Model Building and Training

**Objective:** Train and evaluate models to detect fraudulent transactions.

**Steps:**

1. **Data Preparation:**

   * Stratified train-test split preserves class distribution.
   * Features separated from target (`class` for E-commerce, `Class` for credit data).
   * Scaled numeric features and one-hot encoded categorical features.

2. **Baseline Model:** Logistic Regression

   * Interpretable, evaluated using **F1-score**, **AUC-PR**, and **confusion matrix**.

3. **Ensemble Model:** Random Forest (best-performing)

   * Hyperparameters tuned (`n_estimators=200`, `max_depth=10`)
   * Class weight balanced
   * Evaluated using same metrics

4. **Cross-Validation:**

   * Stratified K-Fold (k=5) for reliable performance estimation
   * Metrics reported as mean ± std

**Outcome:** Random Forest selected as the final model due to superior performance and interpretability.

---

## Task 3: Model Explainability

**Objective:** Understand model decisions using **SHAP**.

**Steps:**

1. **Global Feature Importance:**

   * Top 10 features identified, including time-of-day, transaction frequency, device/IP risk.

2. **Local Explanations:**

   * SHAP force plots created for:

     * True Positive (correctly detected fraud)
     * False Positive (legitimate flagged as fraud)
     * False Negative (missed fraud)

3. **Insights:**

   * High transaction frequency and new accounts are strong fraud indicators.
   * Certain countries show higher risk.
   * Unexpected patterns detected in specific browsers or sources.

4. **Business Recommendations:**

   * Transactions within the first few hours of signup should receive extra verification.
   * High-risk countries or devices should trigger additional checks.
   * Monitor high-value transactions with unusual frequency or time patterns.

---

## Installation

```bash
# Clone repository
git clone <your_repo_url>
cd fraud-detection

# Create virtual environment
python -m venv .venv

.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Place raw datasets in `data/raw/`.
2. Run notebooks in the following order:

```text
1. eda-fraud-data.ipynb
2. eda-creditcard.ipynb
3. feature-engineering.ipynb
4. modeling.ipynb
5. shap-explainability.ipynb
```

3. Processed data will be saved in `data/processed/`.
4. Trained models can be saved/loaded from `models/`.

---

## Key Insights & Business Impact

* Fraud concentrated in specific **countries, devices, and time windows**.
* High transaction frequency and new users are strong fraud indicators.
* Explainability (SHAP) enables **actionable business recommendations**.
* Proper handling of **class imbalance** ensures models can detect minority fraud cases reliably.

---

