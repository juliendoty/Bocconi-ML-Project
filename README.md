# Bocconi ML Project — Player Market Value Prediction  
**Università Bocconi — Machine Learning Coursework (Spring 2025)**

**Short summary**  
This repository contains a **sanitized excerpt** of a coursework machine-learning project that predicts professional footballers’ market values (EUR). The full project used a 15K+ observation, 75-feature dataset; to protect course data and group deliverables, this repo exposes (1) a concise technical summary, (2) selected code excerpts that illustrate the preprocessing and modeling pipelines (Ridge and Random Forest), and (3) sample outputs (RMSE comparison and feature importance). The full dataset and original group submission are not published here.

---

## Table of contents
- [Overview](#overview)  
- [Data preprocessing](#data-preprocessing)  
- [Models evaluated](#models-evaluated)  
- [Training pipelines — technical design](#training-pipelines---technical-design)  
  - [Ridge (linear regularized) pipeline](#ridge-linear-regularized-pipeline)  
  - [Random Forest (ensemble) pipeline](#random-forest-ensemble-pipeline)  
- [Model comparison & results](#model-comparison--results)  
- [Authorship, ethics & data disclaimer](#authorship-ethics--data-disclaimer)  
- [Key takeaways](#key-takeaways)

---

## Overview
The goal of the project was to produce accurate, interpretable predictions of player market value (in €) using player attributes and metadata. Exploratory analysis revealed position-dependent feature availability (notably features present only for Goalkeepers or only for Outfielders), strong right skew in the `value_eur` target, and extensive multicollinearity among technical attributes. To address these issues we applied: targeted preprocessing, log-transformations of the target, role-based model segmentation (Goalkeepers vs Outfielders), regularized linear models, and nonlinear ensemble learners.

---

## Data preprocessing
Key preprocessing steps applied prior to modeling:

- **Irrelevant column removal:** Drop identifiers and high-cardinality textual columns (IDs, names, free-text tags, etc.) that do not generalize to a predictive pipeline.  
- **Categorical encoding:** One-hot / dummy encoding of remaining categorical features (e.g., `preferred_foot`, `work_rate`, `body_type`).  
- **Target transformation:** `value_eur` is heavily right-skewed (skewness ≈ 7.8, kurtosis ≈ 84); apply `log1p(value_eur)` to stabilize variance and improve model fit. Reported RMSE is later back-transformed to euros for interpretability.  
- **Missingness & imputation:** Numeric missing values imputed with the **median** (robust to outliers). Position-specific features (GK-only or OF-only) were handled within each role’s pipeline to avoid introducing artefacts.  
- **Segmentation (role-specific modeling):** Because Goalkeepers and Outfielders have distinct feature sets (e.g., `goalkeeping_speed` only for GK; `pace`, `shooting`, `dribbling`, etc. only for OF), the dataset was partitioned into **Goalkeepers** and **Outfielders**, and models were trained independently on each subset.  
- **Scaling:** Features were standardized (z-score) prior to linear models and optionally prior to tree-based learners depending on the pipeline.

---

## Models evaluated
Compared a set of linear and ensemble methods to balance interpretability and predictive power:

- **Ordinary Least Squares (OLS)** — interpretability baseline.  
- **Ridge regression** — L2-regularized linear model to mitigate multicollinearity.  
- **Random Forest (RF)** — bagged decision-tree ensemble to capture nonlinearities and interactions.  
- **XGBoost (GB)** — gradient-boosted trees with regularization and shrinkage for high predictive capacity.

All models were assessed using cross-validation and held-out evaluation metrics; RMSE was reported on the log-transformed scale and then converted back to euros for interpretability.

---

## Training pipelines — technical design

### Ridge (linear, regularized) pipeline
- **Segmentation:** Ridge models were trained **separately** on the Goalkeeper and Outfielder subsets to respect position-specific feature availability.  
- **Pipeline:** `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])` — scaling followed by Ridge regression.  
- **Hyperparameter tuning:** `GridSearchCV` over `alpha` ∈ `linspace(0.05, 5.0, 100)` using **5-fold cross-validation**.  
- **Scoring & selection:** Models were selected by **maximizing R²** across CV folds (ensures the model explains variance consistently across splits).  
- **Final fit & evaluation:** After selecting the best `alpha` for each subset, the final model was re-fit on that subset’s full training data. In-sample RMSE (on the log target) was computed and then converted back to euro scale using the appropriate back-transformation (mean-based adjustment) to produce interpretable RMSE estimates reported below.

### Random Forest (ensemble) pipeline
- **Segmentation:** Two independent Random Forests were trained — one for **Goalkeepers**, one for **Outfielders** — so each model could fully leverage role-specific features without noisy missingness from the other group.  
- **Model configuration & rationale:** Each forest used bootstrap sampling and tree averaging (bagging) to reduce estimator variance and to increase robustness to multicollinearity and noisy predictors. Feature subsampling (`max_features='sqrt'`) was used to decorrelate trees and reduce overfitting.  
- **Out-of-Bag (OOB) validation:** OOB estimation was enabled (`oob_score=True`) to obtain an unbiased estimate of model generalization without holding out a separate validation set. OOB predictions were used to compute per-subset RMSE and a combined RMSE (by concatenating OOB predictions from both subset models).  
- **Hyperparameter tuning:** Grid search was used to explore `n_estimators`, `max_depth`, and `max_features` (and other sensible ranges) — tuning targeted RMSE reduction while controlling complexity to avoid overfitting. When OOB was used, OOB errors informed selection and early stopping of aggressive parameter configurations.  
- **Bootstrapping & aggregation:** Each tree in the forest is trained on a bootstrap sample (with replacement); final predictions are averages across trees, which stabilizes predictions and captures nonlinear feature interactions naturally.

---

## Model comparison & results

**Validation RMSE (log-scale) and combined normal-scale RMSE (project results):**

| Model | Goalkeepers RMSE (log scale) | Outfielders RMSE (log scale) | Combined RMSE (normal scale) |
| --- | ---: | ---: | ---: |
| **OLS** | 0.2882 | 0.1848 | €628,212.56 |
| **Ridge** | 0.2763 | 0.1836 | €618,417.56 |
| **Random Forest** | **0.2306** | **0.1470** | **€489,948.22** |
| **XGBoost (GB)** | 0.2458 | 0.1917 | €626,127.40 |

**R² (log-space) — final models (reported):**

| Model | Goalkeepers (R², log) | Outfielders (R², log) | Combined (R², normal) |
| --- | ---: | ---: | ---: |
| **Random Forest** | 0.965 | **0.986** | **0.996** |
| **XGBoost** | 0.960 | 0.976 | 0.993 |
| **OLS** | 0.945 | 0.978 | 0.993 |
| **Ridge** | 0.949 | 0.978 | 0.993 |

**Interpretation**  
- Random Forest produced the *lowest* RMSE (best predictive accuracy) and the *highest* R², particularly on the Outfielder subset where nonlinearities and feature interactions are more pronounced.  
- Ridge offered modest improvements over OLS in coefficient stability, but predictive gains were limited — suggesting that in this dataset the primary performance gains come from capturing nonlinear structure (which tree ensembles do naturally).  
- XGBoost performed comparably to RF in many respects, but in this specific training environment Random Forest combined better robustness and training-time convenience (OOB evaluation allowed efficient model assessment without an explicit validation split).

---

## Authorship, ethics & data disclaimer

- Course context: This work was completed as part of a group coursework assignment at Università Bocconi (Spring 2025).
- Authorship: The excerpted code and analyses shown here reflect **code authored exclusively by the repository owner** and analysis authored primarily by the repository owner; full group contributions are acknowledged in the original course submission.
- Data & confidentiality: The complete course dataset and the full group deliverables are not published in this repository. The excerpted notebook is intended to demonstrate methodology, coding style, and pipeline design without exposing proprietary or course-controlled data.

---

## Key takeaways

- Analytical and quantitative problem-solving: This project helped me connect the mathematical foundation of machine learning — from linear regression assumptions to regularization and ensemble variance reduction — to actual code that performs effectively on real-world data. The project demanded deep analytical thinking, especially in understanding why certain models (e.g., Random Forests) handled multicollinearity and nonlinearity more effectively than others (e.g., OLS). This reinforced my ability to reason mathematically about algorithm behavior and to make model choices grounded in both data structure and theory.
- Significant growth in Python and ML workflow design: Working hands-on with libraries like pandas, scikit-learn, and xgboost greatly expanded my confidence in Python-based data science. I gained a stronger sense of how to structure end-to-end workflows — from data cleaning and transformation to model evaluation and visualization — with efficiency and clarity.
- Acknowledgements: I am incredibly grateful to Prof. Andrea Celli at Univerità Bocconi for his excellent instruction and enthusiasm for machine learning. It was an absolute privilege to enroll in such an impactful course during my semester exchange. It is truly exhilirating to have such a diverse data analysis toolkit that I am eager to leverage to probe deeper into contemporary issues spanning environmental science, economics, and sustainability.
