# 03 — Results

Números de la última ejecución reproducible del pipeline (`python3 main.py`).  
Figuras en `output/figures/`. Reportes en `output/reports/`.

---

## Case 1 — Clasificación (holdout test)

| Modelo | F1 | ROC-AUC | PR-AUC | Precision | Recall | Brier | ECE |
|--------|---:|--------:|-------:|----------:|-------:|------:|----:|
| **RandomForest** | **0.661** | 0.856 | **0.680** | 0.573 | 0.781 | 0.131 | **0.016** |
| XGBoost | 0.659 | **0.858** | 0.669 | 0.562 | 0.797 | 0.132 | 0.025 |
| LogisticRegression | 0.643 | 0.853 | 0.678 | 0.534 | **0.810** | 0.131 | 0.026 |

**Mejor por F1:** RandomForest.  
Logistic Regression maximiza recall (más FP): útil si el coste de visita es bajo.

### Umbrales (RandomForest, desde val)

| Concepto | Valor |
|----------|------:|
| `threshold_opt` (F1) | ~0.27 |
| `risk_threshold_medium` | ~0.32 |
| `risk_threshold_high` | ~0.47 |
| Umbral por costes (`FN=5×FP`) | ~0.21 |

Lectura: con desbalanceo + FN caro, el umbral óptimo de negocio **baja** respecto a 0.5.

### Figuras

- `output/figures/roc_curves.png`
- `output/figures/pr_curves.png`
- `output/figures/calibration_RandomForest.png`
- `output/figures/cost_curve.png`
- `output/figures/shap_summary_RandomForest.png`

---

## Lift / Gains (negocio)

Tabla: `output/reports/lift_table_RandomForest.csv`  
Figura: `output/figures/lift_gains.png`

| Contacto | % churners capturados | Lift acumulado |
|----------|----------------------:|---------------:|
| Top ~10% | ~28% | **2.85×** |
| Top ~20% | ~51% | **2.55×** |
| Top ~30% | ~69% | **2.30×** |

**Interpretación:** con capacidad limitada de visitas, el ranking multiplica la efectividad vs contacto aleatorio.

---

## Case 2 — Potencial comercial

| Modelo | MAE | RMSE | R² |
|--------|----:|-----:|---:|
| Ridge | 2.64 | 3.25 | 0.896 |
| **XGBoost Regressor** | **0.33** | **0.58** | **0.997** |

Output: `output/reports/case2_commercial_scoring.csv`  
Figura: `output/figures/case2_potential_distribution.png`

---

## Case 3 — Anomalías

- Contaminación 5% → ~353 anomalías
- Precision@K ≈ base rate de churn (~0.26)  
  → el modelo detecta **rareza de facturación**, no churn directo (esperado)

Output: `output/reports/case3_anomaly_scoring.csv`  
Figura: `output/figures/case3_anomaly_distribution.png`

---

## Case 4 — Survival

| Resultado | Valor |
|-----------|------:|
| Concordance (Cox) | ~0.83 |
| HR `Month-to-month` | ~9.2× |
| Mediana supervivencia Month-to-month | ~35 meses |
| One/Two year | mediana no alcanzada en ventana (`inf`) |

Output por cliente: `P(churn en 6/12/24m)`  
Archivos:

- `output/reports/survival_cox_hazard_ratios.csv`
- `output/reports/survival_risk_scoring.csv`
- `output/figures/survival_kaplan_meier.png`
- `output/figures/survival_cox_hazard_ratios.png`

---

## Score unificado

Pesos: churn 0.50 / potencial 0.30 / anomalía 0.20  
Archivo: `output/reports/unified_commercial_scoring.csv`  
Figura: `output/figures/unified_commercial_scoring.png`

Segmentos típicos: `Maintain`, `Retain_Urgent`, `Retain_HighValue`, `Grow_Upsell`, …

---

## Limitaciones (honestas)

1. Snapshot transversal → survival aproximado (no panel mes a mes)
2. Anomalías ≠ predictor de churn
3. Sin A/B: el lift es potencial de targeting, no impacto causal medido
4. Dominio telecom: revalidar features en otros sectores

---

## Siguiente lectura

→ [04 — Business playbook](04_business_playbook.md)
