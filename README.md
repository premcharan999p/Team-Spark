# Team-Spark  
# Rossmann Store Sales Forecasting (Monthly 3-Step Multi-Horizon Forecasting)

## Team Contributions
1) **Rahul Naik (22CS01068):** Data preprocessing + Feature engineering  
2) **Prem Charan (22CS01062):** Baseline models + XGBoost model training  
3) **G. Sivaji (22CS01066):** Dimensionality reduction + Forecasting data to embeddings + Optuna fine-tuning + BiLSTM tuned model  
4) **R. Nikhilesh(22cs01064):** Data understanding, justification of dataset suitability, and explaining why the selected approach is best for the hackathon use case  

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
   - 9.1 [Baselines](#91-baselines)  
   - 9.2 [Linear Regression / Ridge](#92-linear-regression--ridge)  
   - 9.3 [ARIMA / SARIMA](#93-arima--sarima)  
   - 9.4 [XGBoost (Final Winner)](#94-xgboost-final-winner)  
   - 9.5 [BiLSTM Experiments](#95-bilstm-experiments)  
10. [Feature Processing](#feature-processing)  
    - 10.1 [Why One-Hot Encoding](#101-why-one-hot-encoding)  
    - 10.2 [Why TruncatedSVD (Not PCA)](#102-why-truncatedsvd-not-pca)  
    - 10.3 [Scaling](#103-scaling)  
11. [Evaluation Metrics](#evaluation-metrics)  
    - 11.1 [MAPE vs wMAPE](#111-mape-vs-wmape)  
12. [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)  
13. [Final Results](#final-results)  
14. [SHAP Explainability](#shap-explainability)  
15. [Artifacts Saved for Deployment](#artifacts-saved-for-deployment)  
16. [How to Run (Google Colab)](#how-to-run-google-colab)  
17. [Deployment (FastAPI)](#deployment-fastapi)  
18. [Deliverables](#deliverables)  

---

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
For each **store-month** input row, our system returns three predictions:

- **H1 / pred_h1:** forecast monthly sales for next month (**t+1**)  
- **H2 / pred_h2:** forecast monthly sales for (**t+2**)  
- **H3 / pred_h3:** forecast monthly sales for (**t+3**)  

**Units:** money amount (monthly aggregated `Sales`).

---

## Exogenous (External) Features Used
In this project, the primary **exogenous drivers** (external influences) are:

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
