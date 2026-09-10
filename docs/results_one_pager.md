# One-pager — Telco Churn Scoring

> Full documentation: [`docs/README.md`](README.md)

## The pitch

**Problem:** you cannot visit all ~7k customers; calling at random burns retention budget.  
**Result:** a commercial ranking where the **top 20%** captures ~**51%** of churners (**2.5×** vs random).

---

## Problem

Prioritize a sales force over ~7k telecom customers with **26.5% churn**.  
Accuracy misleads: always predicting “No Churn” gets ~73% without detecting anyone.

## Approach

| Layer | What it does |
|-------|--------------|
| Case 1 — Classification | Calibrated churn probability + dynamic tiers |
| Case 2 — Regression | Upsell potential (€/month) |
| Case 3 — Anomalies | Billing rarity (Isolation Forest) |
| Case 4 — Survival | Timing risk (KM + Cox) |
| Unified | Commercial ranking with playbooks |

Validation: **train / val / test**. Tuning on train only. Calibration/thresholds on val. Final metrics on test.

## Results (holdout)

**Best model:** RandomForest — F1 **0.66** | ROC-AUC **0.86** | PR-AUC **0.68** | ECE **0.016**

**Lift (business):**

| If you contact… | You capture… of churners | vs random |
|-----------------|-------------------------:|----------:|
| Top 10% | 28% | **2.85x** |
| Top 20% | 51% | **2.55x** |
| Top 30% | 69% | **2.30x** |

**Costs:** with `cost(FN)=5 × cost(FP)`, optimal threshold ~**0.21** (not 0.5).

**Case 2:** XGBoost Regressor MAE ≈ 0.33 | R² ≈ 0.997  
**Survival:** Cox concordance ≈ 0.83; Month-to-month HR ≈ 9.2x; 12-month risk per customer  
**Unified:** segments `Retain_HighValue`, `Retain_Urgent`, `Grow_Upsell`, etc.

## Why it matters

The output is not only a model: it is an **actionable queue** (retention, upsell, billing review) with a score readable as a probability — and survival adds *when* (6/12/24m horizon), not only *if*.

## Limitations

- Snapshot data (not month-by-month history) → survival uses `tenure` + censoring
- Anomalies ≠ churn (Precision@K ~ base rate)
- No A/B experiment for causal impact

## Survival layer (implemented)

Kaplan–Meier (global + by Contract) and Cox PH on `tenure` / `Churn`.  
Artifacts: `survival_kaplan_meier.png`, `survival_cox_hazard_ratios.csv`, `survival_risk_scoring.csv`.
