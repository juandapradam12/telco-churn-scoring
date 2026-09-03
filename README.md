# H&K Prueba Tecnica — ML Engineer
### Clasificacion de Churn y Priorizacion Comercial

**Autor:** Juan Prada · **Fecha:** Abril 2026

---

## Contexto

Prueba tecnica para el rol de ML Engineer en H&K. El caso implementado es prediccion de churn (clasificacion binaria) sobre un dataset de clientes de telecomunicaciones. El objetivo es generar un scoring de riesgo por cliente que permita priorizar las visitas de la fuerza comercial en campo.

---

## Estructura del proyecto

```
hk-prueba-tecnica-ml/
├── docs/
│   └── Prueba_Tecnica_ML_Engineer.pdf  # Enunciado original de la prueba
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
├── output/
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
- `figures/` — graficas de EDA, curvas ROC, SHAP, scoring
- `reports/model_comparison.csv` — tabla comparativa de metricas
- `reports/churn_scoring.csv` — ranking de clientes por riesgo de churn

### 3. Ver el analisis narrativo

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

Contiene el analisis completo con explicaciones de cada decision tecnica.

---

## Deep Dive técnico (versión mejorada)

El pipeline del repo ya incluye una versión “deep” del caso de churn con:
- partición **train/val/test** (tuning sin sesgo en test),
- **calibración de probabilidades** (sigmoid/Platt scaling) con métricas como **Brier** y **ECE**,
- **umbral tuning** y `risk_tier` con cortes alineados a objetivos de recall.

Resumen técnico y resultados en:
[`docs/churn_case_deep_dive.md`](docs/churn_case_deep_dive.md)

---

## Modelos comparados

| Modelo | Descripcion |
|--------|-------------|
| Logistic Regression | Baseline lineal interpretable |
| Random Forest | Ensemble de arboles, captura no-linealidades |
| XGBoost | Gradient boosting, mejor rendimiento en datos tabulares |

**Metrica principal:** F1-Score  
**Justificacion:** Con un desbalanceo del 26% de churn, la accuracy no es apropiada. El F1 penaliza tanto falsos negativos (clientes que se van sin detectar) como falsos positivos (recursos desperdiciados). El AUC-ROC complementa midiendo la capacidad discriminativa general.

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
Con un desbalanceo del 26.5% (no extremo), la ponderacion de clases es suficiente y mas interpretable. SMOTE genera muestras sinteticas que pueden introducir ruido con variables categoricas, que son mayoritarias en este dataset. El threshold tuning y la calibracion de probabilidades son mejoras validas para una siguiente iteracion.

---

## Limitaciones y mejoras futuras

### Limitaciones actuales

- El dataset es de telecomunicaciones. Aplicarlo a otros sectores requiere revalidar el feature engineering.
- No se modelan efectos temporales ni estacionalidad del churn.
- El scoring asume que la distribucion de clientes es estable. Se recomienda reentrenamiento periodico.
- Como mejoras al tratamiento del desbalanceo: optimizacion del threshold de clasificacion, SMOTE, o calibracion de probabilidades (Platt scaling).

### Que se haria con mas tiempo

**Sobre los mismos datos:**

- **Threshold tuning:** el umbral de clasificacion (0.5 por defecto) es arbitrario. Se buscaria el umbral optimo barriendo de 0.1 a 0.9 y seleccionando el que maximiza F1, o el que garantiza un Recall minimo del X% segun criterio de negocio.
- **Optimizacion de hiperparametros:** usando Optuna (busqueda bayesiana) en lugar de grid search, especialmente para XGBoost y Random Forest. Es probable que XGBoost bien afinado supere a Random Forest.
- **Calibracion de probabilidades:** aplicar Platt scaling o isotonic regression para asegurar que un score de 0.7 signifique realmente un 70% de probabilidad de churn, no solo "mas probable que 0.6". Importante para que el scoring sea interpretable por negocio.
- **PR-AUC** como metrica adicional: el area bajo la curva Precision-Recall es mas informativa que ROC-AUC cuando el desbalanceo es relevante.

**Con datos temporales:**

- **Survival analysis** (Cox Proportional Hazards, Kaplan-Meier): en lugar de predecir si el cliente se va, predecir *cuando* se va. Permite planificar visitas con mayor anticipacion y priorizar clientes cuyo riesgo aumenta en las proximas semanas.
- **Features de comportamiento temporal:** variacion de `MonthlyCharges` mes a mes, numero de incidencias de soporte recientes, tendencia de uso de servicios.
- **Validacion temporal correcta:** en lugar de split aleatorio, usar los ultimos N meses como test para evitar data leakage temporal — el modelo no puede "ver el futuro" durante el entrenamiento.

**Con datos de negocio adicionales:**

- **Customer Lifetime Value (CLV)** como peso en la funcion de perdida: no todos los churners valen igual. Un cliente con alto CLV deberia tener mayor prioridad de retencion aunque su probabilidad de churn sea similar a la de otro de bajo valor.
- **Calibracion del umbral por segmento:** segun la capacidad de visitas del equipo comercial y el coste de retencion por tipo de cliente, el umbral optimo puede variar entre segmentos.
- **Experimentos A/B:** para medir el impacto real de las acciones de retencion y separar el efecto causal del modelo del simple comportamiento natural del cliente.
