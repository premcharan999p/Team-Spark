# Team-Spark  
# Rossmann Store Sales Forecasting (Monthly 3-Step Multi-Horizon Forecasting)

## Team Contributions
1) **Rahul Naik (22CS01068):** Data preprocessing + Feature engineering  
2) **Prem Charan (22CS01062):** Baseline models + ARIMA + SARIMA + model training  
3) **G. Sivaji (22CS01066):** Dimensionality reduction + Forecasting data to embeddings + Optuna fine-tuning + BiLSTM tuned model  
4) **R. Nikhilesh (22CS01064):** Data understanding, justification of dataset suitability. 

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Problem Statement](#problem-statement)  
3. [Dataset](#dataset)  
4. [Output Definition](#output-definition)  
5. [Exogenous (External) Features Used](#exogenous-external-features-used)  
6. [Tech Stack](#tech-stack)  
7. [End-to-End Workflow](#end-to-end-workflow)  
   - 7.1 [Data Loading + Checks](#71-data-loading--checks)  
   - 7.2 [Merge Strategy](#72-merge-strategy)  
   - 7.3 [EDA](#73-eda)  
   - 7.4 [Missing Value Handling](#74-missing-value-handling)  
   - 7.5 [Duplicate Check](#75-duplicate-check)  
   - 7.6 [Outlier Handling](#76-outlier-handling)  
8. [Forecasting Design](#forecasting-design)  
   - 8.1 [Daily → Monthly Aggregation](#81-daily--monthly-aggregation)  
   - 8.2 [Target Creation: y_t+1, y_t+2, y_t+3](#82-target-creation-y_t1-y_t2-y_t3)  
   - 8.3 [Time Split (No Leakage)](#83-time-split-no-leakage)  
9. [Models Implemented (All Models)](#models-implemented-all-models)  
10. [Model Outputs & Results (All Models)](#model-outputs--results-all-models)  
11. [Feature Processing](#feature-processing)  
12. [Evaluation Metrics](#evaluation-metrics)  
13. [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)  
14. [Final Results](#final-results)  
15. [SHAP Explainability](#shap-explainability)  
16. [Artifacts Saved for Deployment](#artifacts-saved-for-deployment)  
17. [How to Run (Google Colab)](#how-to-run-google-colab)  
18. [Deployment (FastAPI)](#deployment-fastapi)  
19. [Deliverables](#deliverables)  

---

<img width="1900" height="967" alt="Screenshot 2025-12-12 224303" src="https://github.com/user-attachments/assets/dbeb39f2-49ca-4d95-912e-a2ba5f255fa1" />


## Project Overview
This project builds a complete **time series forecasting system** for Rossmann stores using historical sales and store-level information.  
We convert daily sales to **monthly sales per store** and forecast **3 months ahead** (multi-horizon forecasting).  

We tested multiple models (baselines, linear, ARIMA/SARIMA, XGBoost, BiLSTM) and finalized the best-performing approach.

✅ Final best model: **Optuna-tuned XGBoost** (trained separately for H1/H2/H3)

---

## Problem Statement
Predict future monthly store sales using:
- historical store sales trends,
- store metadata,
- and external drivers (exogenous signals) like promotions and holidays.

**Why it matters:**  
Accurate forecasting helps in:
- inventory planning,
- staffing decisions,
- promotion impact estimation,
- budgeting and revenue planning.

---

## Dataset
We used the Kaggle Rossmann dataset (only 2 files):

### Used
- `train.csv`  
  Daily store sales and signals including:
  - `Store`, `Date`, `Sales`, `Customers`, `Open`
  - `Promo`, `StateHoliday`, `SchoolHoliday`, `DayOfWeek`

- `store.csv`  
  Store metadata and market context including:
  - `StoreType`, `Assortment`
  - `CompetitionDistance`, `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear`
  - `Promo2`, `Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`

### Not Used
- `test.csv`, `sample_submission.csv`

---

## Output Definition
For each **store-month** input row, our system returns three predictions (monthly sales):

- **H1 / pred_h1:** forecast monthly sales for next month (**t+1**)  
- **H2 / pred_h2:** forecast monthly sales for (**t+2**)  
- **H3 / pred_h3:** forecast monthly sales for (**t+3**)  

**Units:** money amount (monthly aggregated `Sales`).

---

## Exogenous (External) Features Used
Primary exogenous drivers:
1. **Promo**: short-term promotion flag (0/1)  
2. **StateHoliday**: holiday type (categorical)  
3. **DayOfWeek**: weekly seasonality signal (1–7)  

These features help the model learn demand changes beyond sales history alone.

---

## Tech Stack
- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**: OneHotEncoder, TruncatedSVD, StandardScaler
- **XGBoost**: XGBRegressor
- **Optuna**: hyperparameter tuning
- **SHAP**: model interpretability
- **Matplotlib/Seaborn**: EDA visualizations
- **Joblib**: saving models and preprocessors
- **FastAPI / Uvicorn**: deployment API

---

## Models Implemented (All Models)
We implemented and compared the following model families:

1) **Baselines (Monthly)**
- Naive (Lag-1)
- Seasonal Naive (Lag-12)
- MA-3 (Moving Average)

2) **Linear Model**
- Linear Regression / Ridge (3 horizon models)

3) **Classical Forecasting**
- ARIMA / SARIMA (benchmark reference, monthly series)

4) **Tree-Based ML (Best)**
- XGBoost (Normal)
- XGBoost (Optuna-tuned) ✅

5) **Deep Learning**
- BiLSTM (tested; did not beat tuned XGBoost)

---

## Model Outputs & Results (All Models)

### Metrics Used for Reporting
- **wMAPE (main)**: lower is better  
- **val_acc = 1 - wMAPE** (accuracy-style)
- MAE/RMSE/MAPE were also computed, but below table focuses on wMAPE/val_acc for direct comparison.

### A) Baseline Results (Monthly)

**Naive (Lag-1)**
- H1: wMAPE **0.1114** | val_acc **0.8886**
- H2: wMAPE **0.1015** | val_acc **0.8985**
- H3: wMAPE **0.0812** | val_acc **0.9188**

**Seasonal Naive (Lag-12, fixed)**
- H1: wMAPE **0.1141** | val_acc **0.8859**
- H2: wMAPE **0.1104** | val_acc **0.8896**
- H3: wMAPE **0.0991** | val_acc **0.9009**

**MA-3 (fixed)**
- H1: wMAPE **0.0904** | val_acc **0.9096**
- H2: wMAPE **0.0731** | val_acc **0.9269**
- H3: wMAPE **0.0800** | val_acc **0.9200**

✅ Best baseline: **MA-3**

---

### B) XGBoost (Normal, before Optuna)
- H1: wMAPE **0.0761** | val_acc **0.9239**
- H2: wMAPE **0.0711** | val_acc **0.9289**
- H3: wMAPE **0.0634** | val_acc **0.9366**

---

### C) XGBoost (Optuna-Tuned) ✅ Final Best
Validation results:

- **H1**
  - wMAPE = **0.0764069779632946**
  - val_acc = **0.9235930220367055**
  - MAPE  = **0.07516675352798571**

- **H2**
  - wMAPE = **0.06990470275142902**
  - val_acc = **0.930095297248571**
  - MAPE  = **0.07483583159168539**

- **H3**
  - wMAPE = **0.06314324152308845**
  - val_acc = **0.9368567584769115**
  - MAPE  = **0.06779279824081506**

- **AVG**
  - avg wMAPE = **0.06981830741260402**
  - avg val_acc = **0.930181692587396**
  - avg MAPE = **0.07259846112016205**

✅ Best overall: **Optuna-tuned XGBoost** (lowest avg wMAPE)

---

### D) Linear Regression / Ridge
Trained as a baseline ML model (3 horizon models).  
**Observation:** underperformed vs XGBoost because it cannot capture non-linear promo/holiday effects and feature interactions.

*(If you want, paste your linear model metrics and we can add exact wMAPE here.)*

---

### E) ARIMA / SARIMA
Used as classical forecasting reference on monthly series.  
**Observation:** not best for multi-store + exogenous + interaction-heavy forecasting.

*(If you want, paste ARIMA/SARIMA metrics and we can add exact values here.)*

---

### F) BiLSTM
Tested on sequence-based setup.  
**Observation:** did not beat tuned XGBoost in our monthly + engineered-feature pipeline (short sequences + tabular/exogenous dominance).

---

## Final Results
✅ **Winner: Optuna-tuned XGBoost**  
Reason:
- best avg validation wMAPE (~ **0.0698**)
- stable performance across all horizons H1/H2/H3
- scalable across 1115 stores
- strong with tabular + exogenous features

---
