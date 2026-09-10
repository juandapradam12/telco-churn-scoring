# 02 — Methodology

## Datos

| Campo clave | Uso |
|-------------|-----|
| `Churn` | Target binario (`Yes`/`No`) |
| `tenure` | Antigüedad (meses) y **tiempo** en survival |
| `MonthlyCharges` / `TotalCharges` | Facturación |
| Servicios / contrato / pago | Predictores categóricos |

Notas de calidad:

- `TotalCharges` llega como string; se convierte a numérico.
- Clientes con `tenure = 0` tienen `TotalCharges` vacío → imputación a `0`.
- Churn rate ≈ **26.5%** (desbalanceo moderado).

## Feature engineering

### Encoding

- Yes/No → `0/1`
- Multi-categoría → one-hot (`drop_first=True`)
- `customerID` se preserva para scoring (no entra al modelo)

### Features de negocio

- Indicadores `has_*` de servicios adicionales
- Agregados: `num_additional_services`, protección, streaming
- Facturación:
  - `charge_ratio`
  - `deviation_from_expected`
  - `avg_monthly_charge`
  - `log_monthly_charges` / `log_total_charges`

## Validación: train / val / test

```text
100% datos
 ├── 60% train   → fit + hyperparameter tuning
 ├── 20% val     → calibración + umbrales
 └── 20% test    → métricas finales + lift
```

Todo estratificado por churn.  
Así se evita el error clásico de “tunear mirando test”.

## Case 1 — Clasificación de churn

### Modelos

- Logistic Regression (baseline interpretable, con scaler)
- Random Forest
- XGBoost (si disponible)

### Desbalanceo

- `class_weight="balanced"` (sklearn)
- `scale_pos_weight ≈ n_neg / n_pos` (XGBoost)
- **No SMOTE** por defecto: desbalanceo moderado + muchas categóricas

### Calibración

- `CalibratedClassifierCV` (método sigmoid / Platt)
- Métricas de calidad probabilística: **Brier**, **ECE**

### Umbrales

1. **Óptimo F1** (con constraint de recall en val)
2. **Por costes:** minimiza `FP × cost_fp + FN × cost_fn`  
   (default: `cost_fn = 5 × cost_fp`)
3. **Tiers de riesgo** alineados a objetivos de recall en val  
   (no cortes fijos 0.3/0.6)

### Métricas principales

| Métrica | Por qué |
|---------|---------|
| F1 | Equilibra FP/FN con desbalanceo |
| PR-AUC | Mejor que ROC cuando la clase positiva es minoritaria |
| ROC-AUC | Discriminación global |
| Recall | Captura de churners (negocio) |
| Brier / ECE | Fiabilidad del score como probabilidad |

## Case 2 — Potencial comercial (regresión)

Target construido:

```text
monthly_potential = P75(MonthlyCharges | Contract, InternetService) − MonthlyCharges
```

(clip a ≥ 0)

Modelos: Ridge (baseline) + XGBoost Regressor.  
Métricas: MAE, RMSE, R².

## Case 3 — Anomalías de facturación

- **Isolation Forest** (no supervisado; no usa `Churn` como input)
- Features orientadas a rareza de facturación / precio vs servicios
- Validación exploratoria: Precision@K vs base rate de churn  
  (útil como chequeo, **no** como objetivo principal)

## Case 4 — Survival analysis

| Campo | Rol |
|-------|-----|
| `tenure` | duración (tiempo hasta evento) |
| `Churn` | evento (1) / censurado (0) |

### Modelos

- **Kaplan–Meier:** curva de supervivencia global y por `Contract` (+ log-rank)
- **Cox PH:** hazard ratios + concordance
- Scoring: `P(churn antes de t) = 1 − S(t | X)` para t = 6, 12, 24 meses

### Caveats (importante narrarlos)

- Dataset **snapshot**, no panel longitudinal con fechas de calendario
- No-churners están **right-censored** en su tenure actual
- Interpreta riesgo relativo en el tiempo de vida observado

## Score comercial unificado

Pesos por defecto:

| Señal | Peso |
|-------|-----:|
| Churn score | 0.50 |
| Potencial upsell | 0.30 |
| Anomalía | 0.20 |

Genera:

- `commercial_priority_score`
- `commercial_segment` (playbook)
- `recommended_action`

Detalle de segmentos en [04 — Business playbook](04_business_playbook.md).

## Interpretabilidad

- Feature importance (árboles)
- SHAP summary (cuando aplica)

## Siguiente lectura

→ [03 — Results](03_results.md)
