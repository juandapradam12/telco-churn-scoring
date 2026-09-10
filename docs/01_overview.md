# 01 — Overview

## El problema

Una fuerza comercial no puede visitar a todos los clientes. Llamar al azar:

- desperdicia visitas en clientes de bajo riesgo,
- llega tarde a los que se van a ir,
- no distingue retención vs upsell vs error de facturación.

El dataset [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) tiene **7,043 clientes** y **~26.5% churn**.  
Un modelo naive (“siempre No Churn”) acierta ~73% de accuracy **sin detectar a nadie**. Accuracy no sirve.

## El gancho (planteamiento → resultado)

**Planteamiento:** no puedes llamar a todo el mundo.  
**Resultado:** un ranking accionable donde contactar el **top 20%** captura ~**51%** de los churners (**2.5×** vs aleatorio).

## Qué construye este proyecto

No es “un modelo de clasificación”. Es un **sistema de priorización comercial** con varias capas:

| Capa | Pregunta que responde | Output |
|------|------------------------|--------|
| **Case 1 — Clasificación** | ¿Se va? | `churn_score` calibrado + tiers |
| **Case 2 — Regresión** | ¿Cuánto más podría facturar? | potencial €/mes + prioridad upsell |
| **Case 3 — Anomalías** | ¿Hay rareza de facturación? | `anomaly_score` |
| **Case 4 — Survival** | ¿Cuándo se va? | P(churn en 6/12/24 meses) |
| **Unificado** | ¿Qué hago yo ahora? | segmento + acción recomendada |

## Principios de diseño

1. **Validación honesta:** train / val / test. Tuning en train, calibración/umbrales en val, métricas en test.
2. **Probabilidades útiles:** calibración (sigmoid) + Brier/ECE.
3. **Decisión de negocio:** umbral por costes + lift/gains (no solo F1).
4. **Tiempo:** survival añade urgencia, no solo riesgo estático.
5. **Acción:** el score unificado traduce ML en playbooks (`Retain_HighValue`, `Grow_Upsell`, …).

## Para quién es este repo

- Portfolio / caso técnico de ML aplicado a negocio
- Data scientists que quieren ver un end-to-end defendible
- Perfiles de negocio que necesitan una cola priorizada, no un AUC aislado

## Siguiente lectura

→ [02 — Methodology](02_methodology.md)
