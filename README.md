# Fraud Detection – Task 1: Data Analysis and Preprocessing

**Objective:**  
Prepare clean, feature-rich datasets for both E-commerce and bank transaction fraud detection. This includes data cleaning, exploratory data analysis (EDA), geolocation integration, feature engineering, and handling class imbalance.

## 1. Loading Datasets

We load the following datasets:

1. `Fraud_Data.csv` – E-commerce transactions  
2. `IpAddress_to_Country.csv` – Maps IP ranges to countries  
3. `creditcard.csv` – Bank transaction data

**Goal:** Inspect the data for structure, missing values, and data types before preprocessing.

## 2. Data Cleaning

**Steps:**

- Removed duplicate rows to avoid biased modeling.
- Filled missing `age` values with median (robust against outliers).
- Converted `signup_time` and `purchase_time` to `datetime`.
- Checked data types and corrected as needed.
- Verified there are no missing values in critical columns.

**Result:** Cleaned datasets ready for EDA.

## 3. Exploratory Data Analysis (EDA)

**E-commerce Fraud Data:**

- Class distribution shows severe imbalance: far fewer fraud cases than legitimate transactions.
- Country-level fraud patterns: some countries have higher fraud percentages (Turkmenistan, Namibia, Sri Lanka, Luxembourg, Virgin Islands).
- Examined distributions of key numerical features (`purchase_value`, `age`).
- Checked categorical variables (`browser`, `source`, `sex`) for patterns.

**Bank Credit Card Data:**

- Class distribution: 0 = 99.8%, 1 = 0.17% → extremely imbalanced.
- PCA features (`V1`–`V28`) are ready for modeling.
- Transaction `Amount` shows higher values for fraudulent transactions.

## 4. Geolocation Integration

**Process:**

1. Converted IP addresses to integers for merging.
2. Merged `Fraud_Data.csv` with `IpAddress_to_Country.csv` using range-based lookup.
3. Analyzed fraud patterns by country.

**Insights:**
- Fraudulent transactions are concentrated in specific countries.
- Geolocation is a valuable predictive feature.

## 5. Feature Engineering

**E-commerce Features:**

- Time-based: `hour_of_day`, `day_of_week` from `purchase_time`.
- Duration: `time_since_signup` = purchase_time – signup_time.
- Transaction frequency per user.
- One-Hot Encoding for categorical features (`browser`, `source`, `sex`).
- Standard scaling for numerical features (`purchase_value`, `age`).

**Bank Features:**

- PCA features (`V1`–`V28`) retained.
- `Amount` scaled.

## 6. Handling Class Imbalance

**Technique:** SMOTE (Synthetic Minority Oversampling Technique) applied to **training data only**.

**Reasoning:**
- Synthetic oversampling improves model learning for minority class.
- Avoids leakage of synthetic data into validation or test sets.

**Result:** Balanced datasets for model training.

## 7. Summary

- Data cleaning completed for both datasets.
- Exploratory data analysis revealed fraud patterns by country, browser, and transaction time.
- Geolocation features added.
- Feature engineering produced time-based and transaction frequency features.
- Class imbalance handled with SMOTE.
- Datasets are now ready for **model training and evaluation** (Task 2).
