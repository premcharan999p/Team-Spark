# Team-Spark
# Rossmann Store Sales Forecasting (Monthly 3-Step Forecasting)

## Problem Statement
Build a **monthly time series forecasting system** to predict **future sales** using historical store sales data and store metadata.  
We forecast **3 months ahead (multi-horizon)**:
- **H1** = Sales at **t+1**
- **H2** = Sales at **t+2**
- **H3** = Sales at **t+3**

## Dataset Used (Kaggle Rossmann)
We used only:
- `train.csv` (daily sales + promo/holiday signals)
- `store.csv` (store metadata + competition/promo2 info)

Not used:
- `test.csv`, `sample_submission.csv`

---

## Tech Stack
- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**: OneHotEncoder, TruncatedSVD, StandardScaler
- **XGBoost**: XGBRegressor
- **Optuna**: hyperparameter tuning
- **SHAP**: interpretability
- **Matplotlib/Seaborn**: EDA plots

---

## End-to-End Workflow (What We Did)

### 1) Data Loading + Basic Checks
- Loaded `train.csv`, `store.csv`
- Checked shapes, columns, dtypes
- Checked missing values:
  - `train.csv`: no missing
  - `store.csv`: missing in competition and promo2-related columns

### 2) Merge
- Merged `train` and `store` on `Store` → created `df`

### 3) EDA
- Viewed random samples
- Plotted distributions (histograms) for key numeric columns
- Built correlation matrix (with values)

### 4) Missing Value Handling
Imputed missing values for:
- `CompetitionDistance`
- `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`
- `Promo2SinceWeek`, `Promo2SinceYear`
- `PromoInterval`

Verified after imputation: **all missing values = 0**

### 5) Duplicate Check
- Verified **0 duplicate rows**

### 6) Outlier Detection + Handling
- Detected outliers using **IQR**
- Capped required numeric columns (winsorization)
- Verified using **before vs after** distributions + boxplots

### 7) Monthly Forecast Setup (Core)
- Converted daily → **monthly per store** (`monthly_df`)
- Created multi-horizon targets using groupby-shift:
  - `y_t+1`, `y_t+2`, `y_t+3`
- Used **time-based split** with a cutoff:
  - Train = past months
  - Validation = future months  
(Avoids time leakage)

### 8) Baselines
Built baselines on monthly series:
- 1-Month Naive
- Seasonal Naive (lag-12, fixed)
- MA-3 rolling mean (fixed)

### 9) Feature Processing for ML
- **One-Hot Encoding** for categorical features
- **TruncatedSVD** to reduce high-dimensional sparse OHE features into `svd_0..svd_n`
- **StandardScaler** on SVD features

### 10) Modeling
Trained and compared:
- Linear Regression
- ARIMA / SARIMA (monthly total sales reference)
- XGBoost (3 horizons)
- BiLSTM experiments (performed worse than XGB)

### 11) Hyperparameter Tuning (Optuna)
- Tuned XGBoost with **Optuna (10 trials)**
- Objective: minimize **average validation wMAPE** across H1/H2/H3
- Trained final XGB models using best parameters

### 12) Explainability (SHAP)
- SHAP summary plots for H1/H2/H3
- Observed `svd_0` as dominant driver across horizons (SVD components are latent compressed features)

### 13) Deployment Artifacts Saved
Saved for deployment:
- Models: `xgb_h1_opt.pkl`, `xgb_h2_opt.pkl`, `xgb_h3_opt.pkl`
- Preprocessors: `ohe.pkl`, `svd.pkl`, `scaler.pkl`
- Results: `xgb_optuna_results.json`, `xgb_optuna_results.csv`
- Zipped outputs for deployment

---

## Final Results (Optuna-Tuned XGBoost)
Validation metrics (lower is better for wMAPE/MAPE; **val_acc = 1 - wMAPE**)

- **H1**:  
  - wMAPE = **0.0764069779632946**  
  - val_acc = **0.9235930220367055**  
  - MAPE = **0.07516675352798571**

- **H2**:  
  - wMAPE = **0.06990470275142902**  
  - val_acc = **0.930095297248571**  
  - MAPE = **0.07483583159168539**

- **H3**:  
  - wMAPE = **0.06314324152308845**  
  - val_acc = **0.9368567584769115**  
  - MAPE = **0.06779279824081506**

- **AVG**:  
  - avg wMAPE = **0.06981830741260402**  
  - avg val_acc = **0.930181692587396**  
  - avg MAPE = **0.07259846112016205**

✅ Best overall model: **Optuna-tuned XGBoost**

---

## Deliverables
- End-to-end notebook (EDA → FE → Modeling → Evaluation → SHAP)
- Deployment-ready `.pkl` models + preprocessors
- Results files (CSV/JSON)
- SHAP plots

---

## How to Run (Colab)
1. Upload/unzip dataset into `/content/rossmann_data/`
2. Run notebook cells in order:
   - Load → Merge → EDA → Missing/Outliers → Monthly setup
   - OHE → SVD → Scaling
   - Baselines → Linear/XGB → Optuna tuning
   - SHAP → Save artifacts
3. Download deployment zip from `/content/rossmann_outputs.zip`
