# 03 — Results

Numbers from the latest reproducible pipeline run (`python3 main.py`).  
Figures live in `output/figures/`. Reports live in `output/reports/`.

---

## Case 1 — Classification (holdout test)

| Model | F1 | ROC-AUC | PR-AUC | Precision | Recall | Brier | ECE |
|-------|---:|--------:|-------:|----------:|-------:|------:|----:|
| **RandomForest** | **0.661** | 0.856 | **0.680** | 0.573 | 0.781 | 0.131 | **0.016** |
| XGBoost | 0.659 | **0.858** | 0.669 | 0.562 | 0.797 | 0.132 | 0.025 |
| LogisticRegression | 0.643 | 0.853 | 0.678 | 0.534 | **0.810** | 0.131 | 0.026 |

**Best by F1:** RandomForest.  
Logistic Regression maximizes recall (more FPs): useful when visit cost is low.

### Thresholds (RandomForest, from val)

| Concept | Value |
|---------|------:|
| `threshold_opt` (F1) | ~0.27 |
| `risk_threshold_medium` | ~0.32 |
| `risk_threshold_high` | ~0.47 |
| Cost-based threshold (`FN=5×FP`) | ~0.21 |

Takeaway: with imbalance + expensive FNs, the business-optimal threshold **drops** below 0.5.

### Figures

- `output/figures/roc_curves.png`
- `output/figures/pr_curves.png`
- `output/figures/calibration_RandomForest.png`
- `output/figures/cost_curve.png`
- `output/figures/shap_summary_RandomForest.png`

---

## Lift / Gains (business)

Table: `output/reports/lift_table_RandomForest.csv`  
Figure: `output/figures/lift_gains.png`

| Contact | % of churners captured | Cumulative lift |
|---------|-----------------------:|----------------:|
| Top ~10% | ~28% | **2.85×** |
| Top ~20% | ~51% | **2.55×** |
| Top ~30% | ~69% | **2.30×** |

**Interpretation:** under limited visit capacity, the ranking multiplies effectiveness vs random contact.

---

## Case 2 — Commercial potential

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Ridge | 2.64 | 3.25 | 0.896 |
| **XGBoost Regressor** | **0.33** | **0.58** | **0.997** |

Output: `output/reports/case2_commercial_scoring.csv`  
Figure: `output/figures/case2_potential_distribution.png`

---

## Case 3 — Anomalies

- 5% contamination → ~353 anomalies
- Precision@K ≈ churn base rate (~0.26)  
  → the model detects **billing rarity**, not churn directly (expected)

Output: `output/reports/case3_anomaly_scoring.csv`  
Figure: `output/figures/case3_anomaly_distribution.png`

---

## Case 4 — Survival

| Result | Value |
|--------|------:|
| Concordance (Cox) | ~0.83 |
| HR `Month-to-month` | ~9.2× |
| Median survival Month-to-month | ~35 months |
| One/Two year | median not reached in window (`inf`) |

Per-customer output: `P(churn within 6/12/24m)`  
Files:

- `output/reports/survival_cox_hazard_ratios.csv`
- `output/reports/survival_risk_scoring.csv`
- `output/figures/survival_kaplan_meier.png`
- `output/figures/survival_cox_hazard_ratios.png`

---

## Unified score

Weights: churn 0.50 / potential 0.30 / anomaly 0.20  
File: `output/reports/unified_commercial_scoring.csv`  
Figure: `output/figures/unified_commercial_scoring.png`

Typical segments: `Maintain`, `Retain_Urgent`, `Retain_HighValue`, `Grow_Upsell`, …

---

## Honest limitations

1. Cross-sectional snapshot → survival is approximate (not month-by-month panel data)
2. Anomalies ≠ churn predictor
3. No A/B test: lift is targeting potential, not measured causal impact
4. Telecom domain: re-validate features for other industries

---

## Next

→ [04 — Business playbook](04_business_playbook.md)
