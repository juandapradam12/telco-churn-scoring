# One-pager — Telco Churn Scoring

## El gancho

**Planteamiento:** no puedes visitar a los 7k clientes; llamar al azar quema presupuesto de retencion.  
**Resultado:** un ranking comercial donde el **top 20%** captura ~**51%** de los churners (**2.5×** vs aleatorio).

Resumen ejecutivo del caso. Detalle tecnico en [`churn_case_deep_dive.md`](churn_case_deep_dive.md).

---

## Problema

Priorizar la fuerza comercial sobre ~7k clientes telecom con **26.5% churn**.
Accuracy engaña: predecir siempre “No Churn” da ~73% sin detectar a nadie.

## Enfoque

| Capa | Que hace |
|------|----------|
| Case 1 — Clasificacion | Probabilidad de churn calibrada + tiers dinamicos |
| Case 2 — Regresion | Potencial de upsell (€/mes) |
| Case 3 — Anomalias | Rareza de facturacion (Isolation Forest) |
| Unificado | Ranking comercial con playbooks |

Validacion: **train / val / test**. Tuning solo en train. Calibracion y umbrales en val. Metricas finales en test.

## Resultados (holdout)

**Mejor modelo:** RandomForest — F1 **0.66** | ROC-AUC **0.86** | PR-AUC **0.68** | ECE **0.016**

**Lift (negocio):**

| Si contactas… | Capturas… de churners | vs aleatorio |
|---------------|----------------------:|-------------:|
| Top 10% | 28% | **2.85x** |
| Top 20% | 51% | **2.55x** |
| Top 30% | 69% | **2.30x** |

**Costes:** con `cost(FN)=5 × cost(FP)`, umbral optimo ~**0.21** (no 0.5).

**Case 2:** XGBoost Regressor MAE ≈ 0.33 | R² ≈ 0.997  
**Survival:** Cox concordance ≈ 0.83; Month-to-month HR ≈ 9.2x; riesgo a 12m por cliente  
**Unificado:** segmentos `Retain_HighValue`, `Retain_Urgent`, `Grow_Upsell`, etc.

## Por que importa

El output no es solo un modelo: es una **cola accionable** (retencion, upsell, revision de facturacion) con score interpretable como probabilidad — y survival añade *cuándo* (horizonte 6/12/24m), no solo *si*.

## Limitaciones

- Datos de snapshot (no historico mes a mes) → survival usa `tenure` + censoring
- Anomalias ≠ churn (Precision@K ~ base rate)
- Sin experimento A/B de impacto causal

## Ya implementado (capa survival)

Kaplan–Meier (global + por Contract) y Cox PH sobre `tenure` / `Churn`.
Artefactos: `survival_kaplan_meier.png`, `survival_cox_hazard_ratios.csv`, `survival_risk_scoring.csv`.
