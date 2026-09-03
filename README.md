# Telco Churn Scoring — Clasificación y Priorización Comercial

**Autor:** Juan Prada · **Fecha:** Abril 2026

---

## Contexto

Proyecto de machine learning aplicado a negocio: predicción de churn (clasificación binaria) sobre un dataset de clientes de telecomunicaciones. El objetivo es generar un scoring de riesgo por cliente que permita priorizar las visitas de la fuerza comercial en campo.

El brief original del proyecto se conserva en [`docs/Enunciado_Proyecto_ML.pdf`](docs/Enunciado_Proyecto_ML.pdf).

---

## Estructura del proyecto

```
telco-churn-scoring/
├── docs/
│   └── Enunciado_Proyecto_ML.pdf   # Brief del proyecto
├── data/
│   └── telco_churn.csv             # Dataset Telco Customer Churn (Kaggle)
├── src/
│   ├── data/
│   │   └── loader.py               # Carga y validacion de datos
│   ├── features/
│   │   └── engineering.py          # Preprocesamiento y feature engineering
│   ├── models/
│   │   └── train.py                # Entrenamiento, evaluacion y serializacion
│   └── visualization/
│       └── plots.py                # Visualizaciones reutilizables
├── notebooks/
│   └── churn_analysis.ipynb        # Notebook narrativo con explicaciones
├── output/                         # Artefactos generados (no versionados)
│   ├── models/                     # Modelos serializados (.pkl)
│   ├── figures/                    # Graficas (.png)
│   └── reports/                    # Metricas y scoring (.csv)
├── main.py                         # Pipeline ejecutable end-to-end
├── requirements.txt
└── README.md
```

---

## Como ejecutar

### 1. Crear entorno virtual e instalar dependencias

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
python3.11 main.py
```

Genera en `output/`:
- `models/` — modelos serializados (LogisticRegression, RandomForest, XGBoost)
- `figures/` — graficas de EDA, curvas ROC, matriz de confusion, SHAP, scoring
- `reports/model_comparison.csv` — tabla comparativa de metricas (incluye PR-AUC y umbral optimo)
- `reports/churn_scoring.csv` — ranking de clientes por riesgo de churn

### 3. Ver el analisis narrativo

Con el entorno virtual activado:

```bash
source .venv/bin/activate
jupyter notebook notebooks/churn_analysis.ipynb
```

Si `jupyter` no está en el PATH (instalación con `pip install --user`), añade `~/.local/bin`:

```bash
export PATH="$HOME/.local/bin:$PATH"
jupyter notebook notebooks/churn_analysis.ipynb
# equivalente: ~/.local/bin/jupyter-notebook notebooks/churn_analysis.ipynb
```

En Cursor: abre `notebooks/churn_analysis.ipynb` como **Jupyter Notebook** (no como texto/JSON). Si aparece el JSON crudo, usa “Open With → Jupyter Notebook” o el icono de notebook en la esquina superior derecha.

Contiene el analisis completo con explicaciones de cada decision tecnica.

---

## Modelos comparados

| Modelo | Descripcion |
|--------|-------------|
| Logistic Regression | Baseline lineal interpretable |
| Random Forest | Ensemble de arboles, captura no-linealidades |
| XGBoost | Gradient boosting, mejor rendimiento en datos tabulares |

**Metrica principal:** F1-Score  
**Metricas complementarias:** AUC-ROC y PR-AUC (esta ultima mas informativa con desbalanceo de clases)

**Justificacion:** Con un desbalanceo del 26% de churn, la accuracy no es apropiada. El F1 penaliza tanto falsos negativos (clientes que se van sin detectar) como falsos positivos (recursos desperdiciados). El AUC-ROC complementa midiendo la capacidad discriminativa general; el PR-AUC es mas sensible al desbalanceo.

El pipeline aplica **threshold tuning** sobre el conjunto de test: barre umbrales de 0.1 a 0.9 y selecciona el que maximiza F1 para la evaluacion final y el scoring comercial.

---

## Output de negocio: Scoring de riesgo

El modelo produce un score (0-1) por cliente, segmentado en tres niveles:

| Nivel | Score | Accion recomendada |
|-------|-------|-------------------|
| High | > 0.6 | Visita urgente — oferta de retencion personalizada |
| Medium | 0.3 – 0.6 | Contacto proactivo — revision de contrato |
| Low | < 0.3 | Mantenimiento — comunicacion periodica |

---

## Tratamiento del desbalanceo de clases

El dataset tiene un 26.5% de churn — desbalanceo moderado pero suficiente para que un modelo naive aprenda a predecir siempre "No Churn" y obtenga 73% de accuracy sin detectar ningun cliente en riesgo.

La estrategia adoptada es **ponderacion de clases**:

- `LogisticRegression` y `RandomForest` usan `class_weight="balanced"`, que calcula automaticamente un peso inversamente proporcional a la frecuencia de cada clase. Con la distribucion del dataset, la clase churn recibe un peso ~2.8x mayor.
- `XGBoost` usa `scale_pos_weight = n_negativos / n_positivos ≈ 2.83`, que tiene el mismo efecto dentro del framework de gradient boosting.

Esto obliga a los modelos a penalizar mas los falsos negativos (clientes que se van sin ser detectados), que es el error mas costoso desde el punto de vista de negocio.

**Por que no SMOTE u otras tecnicas:**  
Con un desbalanceo del 26.5% (no extremo), la ponderacion de clases es suficiente y mas interpretable. SMOTE genera muestras sinteticas que pueden introducir ruido con variables categoricas, que son mayoritarias en este dataset.

---

## Limitaciones y mejoras futuras

### Limitaciones actuales

- El dataset es de telecomunicaciones. Aplicarlo a otros sectores requiere revalidar el feature engineering.
- No se modelan efectos temporales ni estacionalidad del churn.
- El scoring asume que la distribucion de clientes es estable. Se recomienda reentrenamiento periodico.
- El umbral optimo se calcula sobre el conjunto de test; en produccion conviene validarlo con datos out-of-time o cross-validation.

### Que se haria con mas tiempo

**Sobre los mismos datos:**

- **Optimizacion de hiperparametros:** usando Optuna (busqueda bayesiana) en lugar de grid search, especialmente para XGBoost y Random Forest.
- **Calibracion de probabilidades:** aplicar Platt scaling o isotonic regression para asegurar que un score de 0.7 signifique realmente un 70% de probabilidad de churn.
- **SMOTE** u otras tecnicas de resampling si el desbalanceo aumenta en datos reales.

**Con datos temporales:**

- **Survival analysis** (Cox Proportional Hazards, Kaplan-Meier): predecir *cuando* se va el cliente, no solo si.
- **Features de comportamiento temporal:** variacion de `MonthlyCharges` mes a mes, incidencias de soporte, tendencia de uso.
- **Validacion temporal correcta:** usar los ultimos N meses como test para evitar data leakage.

**Con datos de negocio adicionales:**

- **Customer Lifetime Value (CLV)** como peso en la funcion de perdida.
- **Calibracion del umbral por segmento** segun capacidad de visitas y coste de retencion.
- **Experimentos A/B** para medir el impacto real de las acciones de retencion.
