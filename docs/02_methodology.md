# 02 — Methodology

## Data

| Key field | Role |
|-----------|------|
| `Churn` | Binary target (`Yes`/`No`) |
| `tenure` | Tenure in months and **time** in survival |
| `MonthlyCharges` / `TotalCharges` | Billing |
| Services / contract / payment | Categorical predictors |

Data-quality notes:

- `TotalCharges` arrives as a string; converted to numeric.
- Customers with `tenure = 0` have empty `TotalCharges` → imputed to `0`.
- Churn rate ≈ **26.5%** (moderate imbalance).

## Feature engineering

### Encoding

- Yes/No → `0/1`
- Multi-category → one-hot (`drop_first=True`)
- `customerID` kept for scoring (not used as a model feature)

### Business features

- Service indicators `has_*`
- Aggregates: `num_additional_services`, protection, streaming
- Billing:
  - `charge_ratio`
  - `deviation_from_expected`
  - `avg_monthly_charge`
  - `log_monthly_charges` / `log_total_charges`

## Validation: train / val / test

```text
100% data
 ├── 60% train   → fit + hyperparameter tuning
 ├── 20% val     → calibration + thresholds
 └── 20% test    → final metrics + lift
```

All splits are stratified by churn.  
This avoids the classic mistake of “tuning while looking at test”.

## Case 1 — Churn classification

### Models

- Logistic Regression (interpretable baseline, with scaler)
- Random Forest
- XGBoost (when available)

### Class imbalance

- `class_weight="balanced"` (sklearn)
- `scale_pos_weight ≈ n_neg / n_pos` (XGBoost)
- **No SMOTE** by default: moderate imbalance + many categorical variables

### Calibration

- `CalibratedClassifierCV` (sigmoid / Platt scaling)
- Probability-quality metrics: **Brier**, **ECE**

### Thresholds

1. **F1-optimal** (with recall constraint on val)
2. **Cost-based:** minimize `FP × cost_fp + FN × cost_fn`  
   (default: `cost_fn = 5 × cost_fp`)
3. **Risk tiers** aligned to recall targets on val  
   (not hard-coded 0.3/0.6 cuts)

### Primary metrics

| Metric | Why |
|--------|-----|
| F1 | Balances FP/FN under imbalance |
| PR-AUC | More informative than ROC when positives are minority |
| ROC-AUC | Overall discrimination |
| Recall | Churner capture (business) |
| Brier / ECE | Reliability of the score as a probability |

## Case 2 — Commercial potential (regression)

Constructed target:

```text
monthly_potential = P75(MonthlyCharges | Contract, InternetService) − MonthlyCharges
```

(clipped at ≥ 0)

Models: Ridge (baseline) + XGBoost Regressor.  
Metrics: MAE, RMSE, R².

## Case 3 — Billing anomalies

- **Isolation Forest** (unsupervised; does not use `Churn` as input)
- Features oriented to billing rarity / price vs services
- Exploratory check: Precision@K vs churn base rate  
  (useful as a sanity check, **not** the main objective)

## Case 4 — Survival analysis

| Field | Role |
|-------|------|
| `tenure` | duration (time to event) |
| `Churn` | event (1) / censored (0) |

### Models

- **Kaplan–Meier:** global survival curve and by `Contract` (+ log-rank)
- **Cox PH:** hazard ratios + concordance
- Scoring: `P(churn before t) = 1 − S(t | X)` for t = 6, 12, 24 months

### Caveats (important to narrate)

- Dataset is a **snapshot**, not a longitudinal panel with calendar dates
- Non-churners are **right-censored** at their current tenure
- Interprets relative risk over observed lifetime, not pure calendar forecasting

## Unified commercial score

Default weights:

| Signal | Weight |
|--------|-------:|
| Churn score | 0.50 |
| Upsell potential | 0.30 |
| Anomaly | 0.20 |

Produces:

- `commercial_priority_score`
- `commercial_segment` (playbook)
- `recommended_action`

Segment detail: [04 — Business playbook](04_business_playbook.md).

## Interpretability

- Feature importance (tree models)
- SHAP summary (when applicable)

## Navigation

← Previous: [01 — Overview](01_overview.md)  
→ Next: [03 — Results](03_results.md)
