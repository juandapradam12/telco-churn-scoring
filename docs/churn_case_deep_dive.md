# Caso de uso técnico (Deep Dive) — Churn & Priorización Comercial

Este documento resume la versión “deep” del caso de uso implementado en el repo:
un problema de **clasificación binaria** para predecir churn y convertirlo en un **scoring calibrado** con umbrales de negocio (tiers).

---

## 1) Objetivo

Predecir, para cada cliente, la probabilidad de abandono (**churn**) en el dataset Telco Customer Churn.
El output final es un **riesgo calibrado (0–1)** que se usa para priorizar acciones:

- **High**: visita urgente (segmento más pequeño y más riesgoso)
- **Medium**: contacto proactivo
- **Low**: mantenimiento / comunicación periódica

---

## 2) Datos y validación

- Se carga `data/telco_churn.csv`.
- `TotalCharges` se fuerza a numérico (`pd.to_numeric(..., errors="coerce")`) e imputación a `0.0` cuando corresponde.
- Se valida:
  - shape, nulls totales, duplicados, distribución de churn y churn rate.

---

## 3) Feature engineering (más “CRM-like”)

### 3.1 Codificación

- `Churn`: `"Yes"` → `1`, `"No"` → `0`
- columnas *Yes/No*: se binarizan a `0/1`
- `gender`: `Male → 1`
- variables categóricas multi-valor: `One-Hot Encoding (drop_first=True)`

### 3.2 Variables nuevas clave

Antes del one-hot se generan indicadores binarios de servicios:

- `has_onlinesecurity`, `has_onlinebackup`, `has_deviceprotection`, `has_techsupport`
- `has_streamingtv`, `has_streamingmovies`
- agregados:
  - `num_additional_services`
  - `num_protection_services`
  - `num_streaming_services`

Además se incorporan transformaciones útiles para patrones de facturación:

- `charge_ratio` = `TotalCharges / (tenure * MonthlyCharges)`
- `expected_total_charges` = `tenure * MonthlyCharges`
- `deviation_from_expected` = `TotalCharges - expected_total_charges`
- `avg_monthly_charge` = `TotalCharges / tenure` (robusta para tenure=0)
- `log_total_charges`, `log_monthly_charges`

---

## 4) Validación correcta: train / val / test

Se usa partición estratificada:

- `test_size = 0.2`
- `val_size = 0.2` sobre el resto (train+val)

Así se evita el sesgo de “tuning en test”.

---

## 5) Modelos y desbalance

Modelos comparados (tres):

- **Logistic Regression** (baseline interpretable)
- **Random Forest** (no linealidades + interacciones)
- **XGBoost** (cuando está disponible)

Se maneja el desbalance:

- `class_weight="balanced"` (sklearn)
- `scale_pos_weight` para XGBoost

---

## 6) Probabilities calibradas (Sigmoid / Platt scaling)

El repo prioriza que el score sea interpretable como probabilidad:

- se entrena el modelo base en `train`
- se calibra con `val` usando `CalibratedClassifierCV` y un split explícito train→val
- método por defecto: `sigmoid`

Métricas de calibración:

- **Brier score**
- **ECE** (Expected Calibration Error)

---

## 7) Thresholding y tiers con objetivos de negocio (recall)

En `val`:

1. Se elige `threshold_opt` para maximizar **F1** con constraint de **Recall >= 0.8**
2. Se calculan cortes para tiers buscando umbrales alineados a recall:
   - `risk_threshold_medium`: objetivo recall ~ 0.75
   - `risk_threshold_high`: objetivo recall ~ 0.60

En el scoring final:

- **High** si `churn_score >= risk_threshold_high`
- **Medium** si `risk_threshold_medium <= churn_score < risk_threshold_high`
- **Low** si `churn_score < risk_threshold_medium`

---

## 8) Resultados (última ejecución del repo)

Genera `output/reports/model_comparison.csv`.

| Model | F1 | ROC-AUC | PR-AUC | Precision | Recall | Brier | ECE | threshold_opt | medium | high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.6601 | 0.8560 | 0.6757 | 0.5582 | 0.8075 | 0.1315 | 0.0244 | 0.25 | 0.31 | 0.45 |
| LogisticRegression | 0.6440 | 0.8535 | 0.6780 | 0.5344 | 0.8102 | 0.1313 | 0.0274 | 0.25 | 0.32 | 0.47 |
| XGBoost | 0.6416 | 0.8525 | 0.6645 | 0.5358 | 0.7995 | 0.1338 | 0.0263 | 0.23 | 0.29 | 0.44 |

**Mejor modelo por F1**: `RandomForest`.

### Lift / Gains (negocio)

Sobre el holdout de test, el scoring se evalúa también en deciles:

- `output/reports/lift_table_<Model>.csv`
- `output/figures/lift_gains.png`

Interpretación típica: el top 10%/20% del score captura mucho más churn que un contacto aleatorio (lift > 1).

### Score comercial unificado

Combina churn + potencial + anomalía en:

- `output/reports/unified_commercial_scoring.csv`
- `output/figures/unified_commercial_scoring.png`

Pesos por defecto: churn 0.50 / potential 0.30 / anomaly 0.20.
Segmentos de playbook: `Retain_HighValue`, `Retain_InvestigateBill`, `Retain_Urgent`, `Nurture_Upsell`, `Grow_Upsell`, `Investigate_Billing`, `Maintain`.

---

## 9) Artefactos generados

Carpetas:

- Modelos:
  - `output/models/*_base.pkl`
  - `output/models/*_calibrated.pkl`
- Reportes:
  - `output/reports/model_comparison.csv`
  - `output/reports/churn_scoring.csv`
  - `output/reports/lift_table_*.csv`
  - `output/reports/case2_*.csv`
  - `output/reports/case3_*.csv`
  - `output/reports/unified_commercial_scoring.csv`
- Figuras:
  - `output/figures/roc_curves.png`
  - `output/figures/pr_curves.png`
  - `output/figures/calibration_<Model>.png`
  - `output/figures/cost_curve.png`
  - `output/figures/lift_gains.png`
  - `output/figures/shap_summary_<Model>.png` (si aplica)
  - `output/figures/churn_score_distribution.png`
  - `output/figures/unified_commercial_scoring.png`

---

## 10) Cómo ejecutar

```bash
python3 main.py
```



---

## 11) Survival analysis — ¿hay estructura?

**Si, de forma aproximada**, sin datos externos:

| Campo | Uso en survival |
|-------|-----------------|
| `tenure` | tiempo hasta evento (meses) |
| `Churn` | evento (1) / censurado (0) |

Se puede implementar Kaplan–Meier (curvas por contrato/servicios) y Cox PH (hazard ratios).

**Caveats importantes:**
- El dataset es un **snapshot transversal**, no un panel longitudinal con fechas de calendario.
- Los no-churners estan **right-censored** en su tenure actual.
- No hay left-truncation / historico de cambios de plan mes a mes.
- Por tanto survival aqui responde "riesgo relativo en el tiempo de vida observado", no forecasting calendario puro.

Conclusión: es la siguiente capa natural del repo; no bloquea por falta de columnas, pero hay que narrar bien las limitaciones.
