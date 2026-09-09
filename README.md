# Telco Churn Scoring — Deep Dive

**Autor:** Juan Prada

---

## The story (GitHub About)

> **Planteamiento:** la fuerza comercial no puede llamar a todo el mundo — y llamar al azar desperdicia el presupuesto de retencion.  
> **Resultado:** un ranking accionable (retener / upsell / revisar facturacion) donde contactar el **top 20%** captura ~**51%** de los churners (**2.5×** vs aleatorio).

**Description para GitHub** (About → ⚙️):
```text
Most sales teams call the wrong customers. This project turns telco data into a ranked playbook—retain, upsell, or investigate—where the top 20% captures ~51% of churners (2.5× random).
```

Alternativa mas corta:
```text
Stop calling customers at random. Rank who to retain, upsell, or investigate—top 20% of the list captures ~51% of churners (2.5× better than chance).
```

---

## Contexto

Proyecto de machine learning aplicado a negocio sobre el dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn):

- **26.5% churn** → desbalanceo moderado (accuracy no es valida)
- Objetivo: scoring calibrado + urgencia temporal para priorizar acciones comerciales
- Tres casos + survival + un score unificado con playbooks operativos

Brief del proyecto: [`docs/Enunciado_Proyecto_ML.pdf`](docs/Enunciado_Proyecto_ML.pdf)  
One-pager: [`docs/results_one_pager.md`](docs/results_one_pager.md)  
Ejemplo Cliente A vs B: [`docs/example_client_a_vs_b.md`](docs/example_client_a_vs_b.md)  
Detalle tecnico: [`docs/churn_case_deep_dive.md`](docs/churn_case_deep_dive.md)

---

## Como cambiar la descripcion del repo en GitHub

1. Abre el repo en GitHub.
2. En la portada, a la derecha, bloque **About**.
3. Pulsa el icono de **engranaje** (Edit repository details).
4. Campo **Description** → pega el texto de arriba → **Save changes**.

(Alternativa: `Settings` → `General` → `Repository name` / Description.)

---

## Estructura

```
telco-churn-scoring/
├── docs/
│   ├── Enunciado_Proyecto_ML.pdf
│   ├── results_one_pager.md
│   └── churn_case_deep_dive.md
├── data/telco_churn.csv
├── src/
│   ├── data/loader.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── train.py      # calibracion, costes, scoring
│   │   ├── tuning.py     # RandomizedSearchCV
│   │   └── lift.py       # lift / gains / deciles
│   ├── cases/
│   │   ├── commercial_potential.py
│   │   ├── anomaly_detection.py
│   │   ├── survival_analysis.py      # Kaplan-Meier + Cox PH
│   │   └── unified_scoring.py
│   └── visualization/plots.py
├── notebooks/churn_analysis.ipynb
├── output/{models,figures,reports}/
├── main.py
└── requirements.txt
```

---

## Como ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Notebook narrativo:

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### Artefactos principales

| Artefacto | Contenido |
|-----------|-----------|
| `output/reports/model_comparison.csv` | F1, ROC/PR-AUC, Brier, ECE, umbrales |
| `output/reports/lift_table_*.csv` | Lift y gains por decil |
| `output/reports/churn_scoring.csv` | Riesgo de churn + tiers |
| `output/reports/case2_commercial_scoring.csv` | Potencial de upsell |
| `output/reports/case3_anomaly_scoring.csv` | Anomalias de facturacion |
| `output/reports/unified_commercial_scoring.csv` | Prioridad comercial unificada |
| `output/figures/` | ROC, PR, calibracion, costes, lift, SHAP, etc. |

---

## Pipeline (que hace `main.py`)

1. EDA + feature engineering (servicios, ratios de facturacion, logs)
2. Split estratificado **train / val / test**
3. **Hyperparameter tuning** solo en train (`RandomizedSearchCV`)
4. **Calibracion** de probabilidades en val (sigmoid) + metricas en test
5. **Threshold por costes** (`cost FN = 5 × cost FP`) y **lift/gains**
6. Scoring de churn (tiers dinamicos desde val)
7. Case 2 (regresion de potencial) + Case 3 (Isolation Forest)
8. **Survival analysis** (Kaplan–Meier + Cox PH) con riesgo a 6/12/24 meses
9. **Score comercial unificado** (pesos 0.50 / 0.30 / 0.20)

---

## Resultados clave (ultima ejecucion)

| Modelo | F1 | ROC-AUC | PR-AUC | Recall | ECE |
|--------|---:|--------:|-------:|-------:|----:|
| RandomForest | 0.661 | 0.856 | 0.680 | 0.781 | 0.016 |
| XGBoost | 0.659 | 0.858 | 0.669 | 0.797 | 0.025 |
| LogisticRegression | 0.643 | 0.853 | 0.678 | 0.810 | 0.026 |

Lift (RandomForest, test):

| Contacto | Churners capturados | Lift acumulado |
|----------|--------------------:|---------------:|
| Top 10% (decil 1) | 28.3% | 2.85x |
| Top 20% | 50.8% | 2.55x |
| Top 30% | 69.0% | 2.30x |

Desbalanceo: **26.5% churn**. Umbral optimo por costes (~0.21) baja respecto a 0.5.

Survival (Cox PH):
- Concordance ≈ **0.83**
- `Month-to-month` HR ≈ **9.2x** vs contratos largos
- Output: `P(churn en 6/12/24 meses)` por cliente

---

## Tratamiento del desbalanceo

- `class_weight="balanced"` / `scale_pos_weight` (~2.8)
- Metricas: F1, PR-AUC, Recall (no accuracy)
- Calibracion + threshold por matriz de costes
- No SMOTE: desbalanceo moderado; variables categoricas mayoritarias

---

## Limitaciones y siguientes pasos

- Snapshot transversal (no panel temporal completo) → survival es factible con `tenure` + evento `Churn`, con caveats
- Case 3 detecta rareza de facturacion, no churn directo
- Reentrenar periodicamente; validar acciones con A/B

Proximas mejoras naturales: capacity-aware thresholding, enriquecimiento de anomalias, y survival con historico longitudinal real (si aparece panel temporal).
